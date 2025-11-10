"""
Training loop con GRPO (Group Relative Policy Optimization) para el orquestador
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
from typing import List, Dict, Optional, Tuple

from config import Config
from orchestrator import orchestrator_generate_graph_with_retry, compute_log_probs_from_sequences, ORCH_SYSTEM
from runner import run_two_agent_graph, set_agent_llm
from verifier import compute_reward
from transformers import PreTrainedModel, PreTrainedTokenizer


class LiveTable:
    """Tabla en vivo para mostrar métricas de entrenamiento."""
    def __init__(self, title: str = "GRPO Training", max_rows: int = 200):
        self.title = title
        self.max_rows = max_rows
        self.rows = []
        empty = pd.DataFrame(columns=["step", "reward_mean", "reward_std", "loss", "kl_penalty", "val_acc"])
        try:
            from IPython.display import display
            self.handle = display(self._styled(empty), display_id=True)
            self.use_ipython = True
        except ImportError:
            self.use_ipython = False
            print(f"{title} (no IPython display available)")

    def _styled(self, df: pd.DataFrame):
        styler = df.style.set_caption(self.title).format({
            "reward_mean": "{:.3f}",
            "reward_std": "{:.3f}",
            "loss": "{:.4f}",
            "kl_penalty": "{:.4f}",
            "val_acc": (lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        })
        try:
            styler = styler.hide(axis="index")
        except Exception:
            pass
        return styler

    def update(self, *, step, reward_mean, reward_std, loss, kl_penalty, val_acc=None):
        self.rows.append(dict(
            step=int(step),
            reward_mean=float(reward_mean),
            reward_std=float(reward_std),
            loss=float(loss),
            kl_penalty=float(kl_penalty),
            val_acc=(None if val_acc is None else float(val_acc)),
        ))
        rows = self.rows[-self.max_rows:]
        df = pd.DataFrame(rows, columns=["step", "reward_mean", "reward_std", "loss", "kl_penalty", "val_acc"])
        if self.use_ipython:
            self.handle.update(self._styled(df))
        else:
            print(f"\nStep {step}: reward={reward_mean:.3f}±{reward_std:.3f}, loss={loss:.4f}, kl={kl_penalty:.4f}")


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_coef: float,
    kl_target: float
) -> Tuple[torch.Tensor, float]:
    """
    Calcula la penalización KL y el coeficiente adaptativo.
    
    Args:
        log_probs: Log-probs del modelo actual [B, T]
        ref_log_probs: Log-probs del modelo de referencia [B, T]
        kl_coef: Coeficiente base de KL
        kl_target: Target de divergencia KL
    
    Returns:
        (kl_penalty_tensor, kl_value)
    """
    kl = (log_probs - ref_log_probs).mean()
    kl_value = kl.item()
    
    # Adaptive KL coefficient (similar to PPO)
    if kl_value > kl_target * 1.5:
        adaptive_coef = kl_coef * 1.5
    elif kl_value < kl_target * 0.5:
        adaptive_coef = kl_coef * 0.5
    else:
        adaptive_coef = kl_coef
    
    # KL penalty es negativo de KL (queremos minimizar la divergencia)
    kl_penalty = -adaptive_coef * kl
    
    return kl_penalty, kl_value


def run_grpo_training(
    cfg: Config,
    orchestrator: PreTrainedModel,
    orchestrator_tokenizer: PreTrainedTokenizer,
    agent_model: PreTrainedModel,
    agent_tokenizer: PreTrainedTokenizer,
    ds_train,
    ds_val,
    ds_test,
    seed: int = 42
):
    """
    Loop principal de entrenamiento GRPO para el orquestador.
    
    Args:
        cfg: Configuración
        orchestrator: Modelo orquestador (se entrena con LoRA)
        orchestrator_tokenizer: Tokenizer del orquestador
        agent_model: Modelo usado por los agentes (frozen)
        agent_tokenizer: Tokenizer de los agentes
        ds_train: Dataset de entrenamiento
        ds_val: Dataset de validación
        ds_test: Dataset de test
        seed: Seed para reproducibilidad
    """
    os.makedirs(cfg.run_root, exist_ok=True)
    
    # Configurar modelo de agentes globalmente
    set_agent_llm(agent_model, agent_tokenizer)
    
    # Optimizador solo para el orquestador
    optimizer = torch.optim.AdamW(
        orchestrator.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Preparar problemas
    problems = [ex["question"] for ex in ds_train]
    gold_answers = [ex["answer"] for ex in ds_train]
    
    logs = []
    table = LiveTable(title="GRPO Training")
    
    # Log-probs de referencia (para KL) - se actualizan periódicamente
    ref_log_probs_batch = None
    
    pbar = tqdm(range(cfg.steps), desc=f"GRPO [{os.path.basename(cfg.run_root)}]")
    for step in pbar:
        # Seleccionar batch de problemas
        rng = np.random.default_rng(seed + step)
        problem_indices = rng.choice(len(problems), size=cfg.batch_problems, replace=False)
        
        # FASE 1: Generar todas las trayectorias SIN gradientes (solo inferencia)
        all_sequences = []
        all_prompt_lengths = []
        all_graph_specs = []
        all_problem_texts = []
        all_gold_answers = []
        
        for prob_idx in problem_indices:
            problem_text = problems[prob_idx]
            gold_answer = gold_answers[prob_idx]
            
            # Preparar prompt para obtener su longitud
            messages = [
                {"role": "system", "content": ORCH_SYSTEM},
                {"role": "user", "content": problem_text},
            ]
            try:
                prompt = orchestrator_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                prompt = f"System: {ORCH_SYSTEM}\nUser: {problem_text}\nAssistant: "
            enc_temp = orchestrator_tokenizer([prompt], return_tensors="pt")
            prompt_len = enc_temp['input_ids'].shape[1]
            
            # Generar múltiples grafos para este problema (sin gradientes) -> no training mode
            successful_trajs = 0
            for traj_idx in range(cfg.trajectories_per_problem):
                # Reintentar hasta 3 veces, si falla saltar esta trayectoria
                result = orchestrator_generate_graph_with_retry(
                    orchestrator_model=orchestrator,
                    orchestrator_tokenizer=orchestrator_tokenizer,
                    problem_text=problem_text,
                    temperature=cfg.orch_temperature,
                    max_tokens=cfg.orch_max_tokens,
                    max_retries=3,
                    base_seed=seed + step * 1000 + prob_idx * 100 + traj_idx
                )
                
                # Si falló después de reintentos, saltar esta trayectoria
                if result is None:
                    continue
                
                successful_trajs += 1
                
                graph_spec, sequence = result
                all_sequences.append(sequence[0])  # [seq_len]
                all_prompt_lengths.append(prompt_len)
                all_graph_specs.append(graph_spec)
                all_problem_texts.append(problem_text)
                all_gold_answers.append(gold_answer)
        
        # Verificar que tenemos al menos una trayectoria válida
        if len(all_sequences) == 0:
            print(f"⚠️ Step {step}: Todas las trayectorias fallaron al generar JSON válido. Saltando este step.")
            print(f"   Total trayectorias intentadas: {cfg.batch_problems * cfg.trajectories_per_problem}")
            continue
        
        print(f"✓ Step {step}: {len(all_sequences)}/{cfg.batch_problems * cfg.trajectories_per_problem} trayectorias exitosas")
        
        # FASE 2: Ejecutar todos los grafos y calcular rewards (sin gradientes)
        all_rewards = []
        for graph_spec, problem_text, gold_answer in zip(all_graph_specs, all_problem_texts, all_gold_answers):
            result = run_two_agent_graph(
                graph_spec=graph_spec,
                problem_text=problem_text,
                temperature=cfg.agent_temperature,
                max_tokens=cfg.agent_max_tokens
            )
            reward = compute_reward(result["final_text"], gold_answer)
            all_rewards.append(reward)
        
        # FASE 3: Calcular log-probs CON gradientes sobre todas las secuencias
        sequences_tensor = torch.stack(all_sequences)  # [B, seq_len]
        prompt_lengths_tensor = torch.tensor(all_prompt_lengths, device=sequences_tensor.device)
        
        log_probs = compute_log_probs_from_sequences(
            orchestrator_model=orchestrator,
            orchestrator_tokenizer=orchestrator_tokenizer,
            sequences=sequences_tensor,
            prompt_lengths=prompt_lengths_tensor
        )  # [B]
        
        # Convertir rewards a tensor
        rewards = torch.tensor(all_rewards, device=next(orchestrator.parameters()).device, dtype=torch.float32)
        
        # Calcular log-probs de referencia si no existen o actualizar periódicamente
        if ref_log_probs_batch is None or (step % 10 == 0):
            # Usar log-probs actuales como referencia (detached)
            ref_log_probs = log_probs.detach().clone()
            ref_log_probs_batch = ref_log_probs
        else:
            ref_log_probs = ref_log_probs_batch
        
        # GRPO: calcular advantages relativos dentro del grupo
        # Para cada problema, normalizar rewards dentro de sus trayectorias
        advantages = []
        for prob_idx in range(cfg.batch_problems):
            start = prob_idx * cfg.trajectories_per_problem
            end = start + cfg.trajectories_per_problem
            group_rewards = rewards[start:end]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            # Normalizar rewards dentro del grupo
            normalized_rewards = (group_rewards - group_mean) / group_std
            advantages.extend(normalized_rewards.tolist())
        
        advantages = torch.tensor(advantages, device=rewards.device)
        
        # Calcular loss: -log_prob * advantage
        policy_loss = -(log_probs * advantages).mean()
        
        # KL penalty
        kl_penalty, kl_value = compute_kl_penalty(
            log_probs.unsqueeze(0),  # [1, B]
            ref_log_probs.unsqueeze(0),  # [1, B]
            cfg.kl_coef,
            cfg.kl_target
        )
        
        # Loss total
        loss = policy_loss + kl_penalty
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(orchestrator.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # ref_log_probs_batch ya se actualiza arriba si es necesario
        torch.cuda.empty_cache()
        
        # Métricas
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        
        # Validación periódica
        val_acc = None
        if (step % cfg.val_every == 0) or (step == cfg.steps - 1):
            val_acc = evaluate_on_validation(
                orchestrator, orchestrator_tokenizer,
                agent_model, agent_tokenizer,
                ds_val, cfg, num_examples=cfg.val_sample_n
            )
        
        row = dict(
            step=int(step),
            reward_mean=float(reward_mean),
            reward_std=float(reward_std),
            loss=float(loss.item()),
            policy_loss=float(policy_loss.item()),
            kl_penalty=float(kl_penalty.item()),
            kl_value=float(kl_value),
            val_acc=(None if val_acc is None else float(val_acc))
        )
        logs.append(row)
        
        if (step % cfg.log_every == 0) or (val_acc is not None):
            table.update(
                step=row["step"],
                reward_mean=row["reward_mean"],
                reward_std=row["reward_std"],
                loss=row["loss"],
                kl_penalty=row["kl_penalty"],
                val_acc=row.get("val_acc", None)
            )
        
        pbar.set_postfix({
            "reward": f"{reward_mean:.3f}",
            "loss": f"{loss.item():.3f}",
            "kl": f"{kl_value:.3f}"
        })
        
        torch.cuda.empty_cache()
    
    # Guardar logs
    try:
        pd.DataFrame(logs).to_csv(os.path.join(cfg.run_root, "train_logs.csv"), index=False)
    except Exception:
        with open(os.path.join(cfg.run_root, "train_logs.jsonl"), "w") as f:
            for r in logs:
                f.write(json.dumps(r) + "\n")
    
    # Guardar adapters
    save_dir = os.path.join(cfg.run_root, "adapters_lora")
    os.makedirs(save_dir, exist_ok=True)
    orchestrator.save_pretrained(save_dir)
    
    return logs


def evaluate_on_validation(
    orchestrator: PreTrainedModel,
    orchestrator_tokenizer: PreTrainedTokenizer,
    agent_model: PreTrainedModel,
    agent_tokenizer: PreTrainedTokenizer,
    ds_val,
    cfg: Config,
    num_examples: Optional[int] = None
) -> float:
    """
    Evalúa el orquestador en el validation set.
    Genera un grafo por problema y mide exact-match accuracy.
    """
    from orchestrator import orchestrator_generate_graph_with_retry
    
    set_agent_llm(agent_model, agent_tokenizer)
    
    n = len(ds_val) if num_examples is None else min(num_examples, len(ds_val))
    correct = 0
    
    for i in range(n):
        ex = ds_val[i]
        problem_text = ex["question"]
        gold_answer = ex["answer"]
        
        try:
            # Generar grafo (greedy para evaluación) con reintentos
            result = orchestrator_generate_graph_with_retry(
                orchestrator_model=orchestrator,
                orchestrator_tokenizer=orchestrator_tokenizer,
                problem_text=problem_text,
                temperature=0.0,  # Greedy
                max_tokens=cfg.orch_max_tokens,
                max_retries=3,
                base_seed=i * 1000
            )
            
            # Si falló después de reintentos, contar como incorrecto
            if result is None:
                continue
            
            graph_spec, _ = result
            
            # Ejecutar grafo
            result = run_two_agent_graph(
                graph_spec=graph_spec,
                problem_text=problem_text,
                temperature=cfg.agent_temperature,
                max_tokens=cfg.agent_max_tokens
            )
            
            # Computar reward
            reward = compute_reward(result["final_text"], gold_answer)
            correct += reward
            
        except Exception as e:
            # Si falla la generación o ejecución, reward = 0
            pass
    
    return correct / n

