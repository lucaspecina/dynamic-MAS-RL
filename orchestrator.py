"""
Orquestador: genera grafos de agentes dinámicamente
"""
from typing import Dict, Any, List, Optional
import json
import re
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel


# Prompt del orquestador
ORCH_SYSTEM = (
    "You are designing a two-agent system to solve math word problems -> Agent A and Agent B.\n"
    "The information flow is automatic: problem → Agent A → (problem + draft) → Agent B → final answer.\n\n"
    "Your task: Design appropriate system prompts for each agent that will guide them in their roles.\n"
    "Output ONLY a valid JSON object in a ```json block with this exact structure:\n"
    "{\n"
    '  "agents": [\n'
    '    {"id": "agentA", "type": "llm", "system_prompt": "<prompt for planning/drafting>"},\n'
    '    {"id": "agentB", "type": "llm", "system_prompt": "<prompt for solving/finalizing>"}\n'
    '  ]\n'
    "}\n\n"
    "Do not include anything else, only the structure above.\n"
    "The last agent is free to use any reasoning or format (including step-by-step) when working, "
    "but when giving the final_answer, it must write ONLY ONE final numeric answer (no units) inside square brackets.\n"
)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Busca el primer bloque ```json ... ``` o el primer {...} parseable.
    """
    # Try fenced block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        json_str = fence.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON del bloque ```json```: {e}")
    
    # Try first balanced-ish { ... }
    brace = re.search(r"(\{.*\})", text, re.S)
    if brace:
        json_str = brace.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON del bloque {{...}}: {e}")
    
    raise ValueError("No se pudo extraer JSON válido del texto del orquestador.")


def validate_graph_spec(spec: Dict[str, Any]) -> None:
    """
    Validación para estructura simplificada (solo agents con system prompts).
    """
    assert isinstance(spec, dict), "spec debe ser dict"
    assert "agents" in spec and isinstance(spec["agents"], list), "agents faltante"
    assert len(spec["agents"]) == 2, "v0: exactamente 2 agentes"
    ids = [a.get("id") for a in spec["agents"]]
    assert all(isinstance(i, str) for i in ids), "ids inválidos"
    assert len(set(ids)) == 2, "ids de agentes deben ser únicos"
    assert ids == ["agentA", "agentB"], "ids deben ser exactamente 'agentA' y 'agentB'"

    for a in spec["agents"]:
        assert a.get("type") == "llm", "v0: type debe ser 'llm'"
        assert isinstance(a.get("system_prompt"), str) and len(a["system_prompt"]) > 0, "system_prompt vacío"


def orchestrator_generate_graph(
    orchestrator_model: PreTrainedModel,
    orchestrator_tokenizer: PreTrainedTokenizer,
    problem_text: str,
    temperature: float = 0.7,
    max_tokens: int = 800,
    seed: Optional[int] = None
) -> tuple[Dict[str, Any], torch.Tensor]:
    """
    Genera un grafo sin gradientes (solo inferencia).
    Patrón estándar GRPO/PPO: primero generar rollouts sin gradientes.
    
    Args:
        seed: Seed opcional para reproducibilidad
    
    Returns:
        (graph_spec, full_sequence)
        - graph_spec: Dict con la especificación del grafo
        - full_sequence: Secuencia completa [prompt + generated] sin gradientes
    
    Raises:
        ValueError: Si no se puede extraer JSON válido
    """
    import random
    import numpy as np
    
    # Set seed si se proporciona
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    messages = [
        {"role": "system", "content": ORCH_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    
    orchestrator_model.eval()
    orchestrator_model.config.use_cache = True
    model_device = next(orchestrator_model.parameters()).device
    
    try:
        prompt = orchestrator_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except (AttributeError, ValueError, TypeError):
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
        prompt = "".join(prompt_parts) + "Assistant: "
    
    enc = orchestrator_tokenizer([prompt], return_tensors="pt").to(model_device)
    
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        use_cache=True,
        pad_token_id=orchestrator_tokenizer.pad_token_id,
        eos_token_id=orchestrator_tokenizer.eos_token_id
    )
    
    if temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    
    # Generar sin gradientes (solo inferencia)
    with torch.no_grad():
        gen_out = orchestrator_model.generate(**enc, **gen_kwargs)
    
    # Manejar diferentes formatos de retorno de generate()
    if isinstance(gen_out, torch.Tensor):
        sequences = gen_out
    else:
        sequences = gen_out.sequences
    
    # Verificar que sequences tenga el formato correcto
    if sequences.dim() != 2:
        raise ValueError(f"sequences tiene dimensión incorrecta: {sequences.dim()}, esperado 2")
    
    # Decodificar y extraer JSON
    input_ids = enc['input_ids']
    prompt_len = input_ids.shape[1]
    
    # Verificar que la secuencia generada tenga al menos el prompt
    if sequences.shape[1] < prompt_len:
        raise ValueError(f"Secuencia generada ({sequences.shape[1]}) es más corta que el prompt ({prompt_len})")
    
    # Extraer solo los tokens generados (después del prompt)
    generated_tokens = sequences[0][prompt_len:]
    
    # Verificar que se generaron tokens
    if len(generated_tokens) == 0:
        raise ValueError("No se generaron tokens (secuencia vacía)")
    
    raw = orchestrator_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Verificar que raw no esté vacío
    if not raw or not raw.strip():
        raise ValueError("Texto generado está vacío después de decodificar")
    
    # Debug: imprimir el texto generado si falla (solo en el último intento)
    try:
        spec = extract_json_from_text(raw)
        validate_graph_spec(spec)
    except (ValueError, AssertionError) as e:
        # Re-lanzar la excepción con el texto generado para debugging
        raise ValueError(f"Error al extraer/validar JSON. Texto generado (primeros 500 chars):\n{raw[:500]}\n\nError original: {e}")
    
    return spec, sequences


def orchestrator_generate_graph_with_retry(
    orchestrator_model: PreTrainedModel,
    orchestrator_tokenizer: PreTrainedTokenizer,
    problem_text: str,
    temperature: float = 0.7,
    max_tokens: int = 800,
    max_retries: int = 3,
    base_seed: Optional[int] = None
) -> Optional[tuple[Dict[str, Any], torch.Tensor]]:
    """
    Genera un grafo con reintentos. Si falla después de max_retries intentos, retorna None.
    
    Returns:
        (graph_spec, full_sequence) si tiene éxito, None si falla después de max_retries
    """
    for attempt in range(max_retries):
        # Usar seed diferente en cada intento para diversidad
        seed = None if base_seed is None else (base_seed + attempt) & 0x7FFFFFFF
        
        try:
            spec, sequences = orchestrator_generate_graph(
                orchestrator_model=orchestrator_model,
                orchestrator_tokenizer=orchestrator_tokenizer,
                problem_text=problem_text,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            return spec, sequences
        except Exception as e:
            # Capturar cualquier excepción (ValueError, AssertionError, etc.)
            # Si es el último intento, retornar None silenciosamente
            if attempt == max_retries - 1:
                # Solo imprimir en el último intento para no saturar logs
                return None
            # Si no, continuar al siguiente intento sin imprimir nada
            continue
    
    return None


def compute_log_probs_from_sequences(
    orchestrator_model: PreTrainedModel,
    orchestrator_tokenizer: PreTrainedTokenizer,
    sequences: torch.Tensor,
    prompt_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Calcula log-probs CON gradientes sobre secuencias ya generadas.
    Patrón estándar GRPO/PPO: forward pass con gradientes después de generar rollouts.
    
    Args:
        sequences: Tensor [B, seq_len] con secuencias completas (prompt + generated)
        prompt_lengths: Tensor [B] con longitudes de prompt para cada secuencia
    
    Returns:
        log_probs_sum: Tensor [B] con suma de log-probs de tokens generados (con gradientes)
    """
    orchestrator_model.train()
    orchestrator_model.config.use_cache = False
    
    B = sequences.shape[0]
    log_probs_sums = []
    
    for b in range(B):
        seq = sequences[b:b+1]  # [1, seq_len]
        prompt_len = prompt_lengths[b].item()
        
        # Forward pass con gradientes
        outputs = orchestrator_model(
            input_ids=seq[:, :-1],  # Todos excepto el último token
            attention_mask=(seq[:, :-1] != orchestrator_tokenizer.pad_token_id)
        )
        logits = outputs.logits  # [1, seq_len-1, vocab_size]
        
        # Log-probs de los tokens generados (excluyendo el prompt)
        target_ids = seq[:, prompt_len:]  # Solo tokens generados
        log_probs = F.log_softmax(logits[:, prompt_len-1:-1, :], dim=-1)  # Alineado con target_ids
        
        # Gather log-probs de los tokens realmente generados
        target_ids_for_gather = target_ids[:, :-1]  # Excluir último (no tiene siguiente)
        log_probs_selected = log_probs.gather(
            dim=-1,
            index=target_ids_for_gather.unsqueeze(-1)
        ).squeeze(-1)  # [1, gen_len-1]
        
        log_probs_sum = log_probs_selected.sum()
        log_probs_sums.append(log_probs_sum)
    
    return torch.stack(log_probs_sums)  # [B]

