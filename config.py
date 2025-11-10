"""
Configuración para entrenamiento RL del orquestador MAS
"""
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Models
    # orchestrator_id: str = "Qwen/Qwen3-0.6B-Base"  # Modelo que genera los grafos (se entrena con LoRA)
    orchestrator_id: str = "Qwen/Qwen3-4B-Instruct-2507"  # Modelo que genera los grafos (se entrena con LoRA)
    agent_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"  # Modelo usado por los agentes del grafo
    
    # Prompting
    prompt_template: str = (
        "Solve step by step.\n"
        "Give ONLY ONE final numeric answer (no units), inside square brackets.\n"
        "Problem: {question}\n\nSolution:"
    )
    
    # Generation params para orquestador
    orch_max_tokens: int = 800  # Tokens máximos para generar el JSON del grafo
    agent_max_tokens: int = 512  # Tokens máximos para respuestas de agentes
    
    # Generation temps
    orch_temperature: float = 0.7  # Temperatura para generar grafos (sampling)
    agent_temperature: float = 0.2  # Temperatura para agentes (más determinista)
    
    # Training schedule (GRPO)
    steps: int = 100
    batch_problems: int = 4  # Número de problemas por batch
    trajectories_per_problem: int = 4  # Número de grafos a generar por problema (para GRPO)
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_accum: int = 1
    
    # GRPO params
    kl_coef: float = 0.1  # Coeficiente de regularización KL
    kl_target: float = 6.0  # Target KL divergence
    
    # Micro-batching
    orch_mb: int = 2  # Micro-batch size para el orquestador
    
    # Monitoring
    log_every: int = 10
    val_every: int = 10
    val_sample_n: int = 50
    ema_momentum: float = 0.9
    
    # Validation size
    val_rows: Optional[int] = None  # if None, uses min(200, len(train))
    
    # Output dir
    run_root: str = f"./run_rl_{int(time.time())}"

    def __post_init__(self):
        os.makedirs(self.run_root, exist_ok=True)

