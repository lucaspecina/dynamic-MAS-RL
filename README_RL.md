# Entrenamiento RL del Orquestador MAS

Módulos Python para entrenar el orquestador con **GRPO (Group Relative Policy Optimization)** y **RLVR (Reward Learning with Verifiable Rewards)**.

## Flujo del Sistema

1. **Orquestador** genera dinámicamente un JSON con el grafo de agentes (2 agentes con system prompts)
2. **Runner** ejecuta ese grafo y obtiene la respuesta final del Agent B
3. **Verificador (RLVR)** compara la respuesta con el gold → reward 0/1
4. **Entrenamiento GRPO**: genera múltiples trayectorias por problema, calcula advantages relativos dentro del grupo, y actualiza el orquestador con regularización KL

## Estructura de Módulos

- **`config.py`**: Configuración (modelos, hiperparámetros GRPO, etc.)
- **`models.py`**: Carga de modelos (orquestador con LoRA, modelo de agentes frozen)
- **`data.py`**: Carga del dataset GSM8K
- **`orchestrator.py`**: Generación de grafos dinámicos (con/sin log-probs)
- **`runner.py`**: Ejecución del grafo de agentes
- **`verifier.py`**: Cálculo de rewards verificables (0/1)
- **`training_grpo.py`**: Loop de entrenamiento GRPO
- **`train_rl.py`**: Script principal

## Uso

### Ejecutar entrenamiento

```bash
python train_rl.py
```

### Usar desde Python

```python
from config import Config
from models import load_tokenizers, make_lora_orchestrator, load_agent_model
from data import load_gsm8k
from training_grpo import run_grpo_training

# Configuración
cfg = Config()
cfg.steps = 100
cfg.batch_problems = 4
cfg.trajectories_per_problem = 4  # Múltiples grafos por problema para GRPO

# Cargar datos y modelos
ds_train, ds_val, ds_test = load_gsm8k(cfg.val_rows)
tok_orch, tok_agent = load_tokenizers(cfg.orchestrator_id, cfg.agent_model_id)
orchestrator = make_lora_orchestrator(cfg.orchestrator_id)
agent_model = load_agent_model(cfg.agent_model_id)

# Entrenar
logs = run_grpo_training(
    cfg=cfg,
    orchestrator=orchestrator,
    orchestrator_tokenizer=tok_orch,
    agent_model=agent_model,
    agent_tokenizer=tok_agent,
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    seed=42
)
```

## Métricas

Durante el entrenamiento se muestran:
- **reward_mean**: Promedio de rewards en el batch
- **reward_std**: Desviación estándar de rewards
- **loss**: Loss total (policy loss + KL penalty)
- **kl_penalty**: Penalización KL para regularización
- **val_acc**: Exact-match accuracy en validación

## GRPO

Group Relative Policy Optimization:
- Genera múltiples trayectorias (grafos) por problema
- Normaliza rewards dentro de cada grupo de trayectorias del mismo problema
- Calcula advantages relativos: `advantage = (reward - group_mean) / group_std`
- Loss: `-log_prob * advantage + KL_penalty`

## Outputs

El entrenamiento guarda en `cfg.run_root`:
- `train_logs.csv`: Logs de entrenamiento
- `adapters_lora/`: Adapters LoRA del orquestador entrenado

