"""
Script principal para entrenar el orquestador con GRPO/RLVR
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch

from config import Config
from models import load_tokenizers, make_lora_orchestrator, load_agent_model
from data import load_gsm8k
from training_grpo import run_grpo_training


def main(args=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train orchestrator with GRPO/RLVR")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--batch-problems", type=int, default=None, help="Batch size (problems per step)")
    parser.add_argument("--trajectories", type=int, default=None, help="Trajectories per problem")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (fewer steps, more prints)")
    parser.add_argument("--run-dir", type=str, default=None, help="Output directory")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Seed para reproducibilidad
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Configuración
    cfg = Config()
    
    # Override config with command line arguments
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch_problems is not None:
        cfg.batch_problems = args.batch_problems
    if args.trajectories is not None:
        cfg.trajectories_per_problem = args.trajectories
    if args.lr is not None:
        cfg.lr = args.lr
    if args.run_dir is not None:
        cfg.run_root = args.run_dir
    
    # Debug mode: reduce steps and increase logging
    if args.debug:
        cfg.steps = min(cfg.steps, 5)
        cfg.batch_problems = min(cfg.batch_problems, 2)
        cfg.trajectories_per_problem = min(cfg.trajectories_per_problem, 2)
        cfg.val_every = 1
        cfg.log_every = 1
        print("🐛 DEBUG MODE ENABLED")
    
    print(f"Run directory: {cfg.run_root}")
    print(f"Configuration: steps={cfg.steps}, batch_problems={cfg.batch_problems}, "
          f"trajectories_per_problem={cfg.trajectories_per_problem}, lr={cfg.lr}")

    # Cargar datos
    ds_train, ds_val, ds_test = load_gsm8k(cfg.val_rows)

    # Cargar tokenizers y modelos
    print("\n== Loading tokenizers ==")
    tok_orch, tok_agent = load_tokenizers(cfg.orchestrator_id, cfg.agent_model_id)
    print("Padding sides:", tok_orch.padding_side, tok_agent.padding_side)

    print("\n== Loading models ==")
    orchestrator = make_lora_orchestrator(cfg.orchestrator_id)
    agent_model = load_agent_model(cfg.agent_model_id)
    print("✓ Models loaded")

    # Training
    print("\n== Training (GRPO/RLVR) ==")
    logs = run_grpo_training(
        cfg=cfg,
        orchestrator=orchestrator,
        orchestrator_tokenizer=tok_orch,
        agent_model=agent_model,
        agent_tokenizer=tok_agent,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        seed=SEED
    )

    print("\n== Training Complete ==")
    print(f"Logs saved to: {cfg.run_root}")
    print(f"Final logs: {len(logs)} steps")


if __name__ == "__main__":
    # Breakpoint útil para debugging
    # import pdb; pdb.set_trace()  # Descomentar para debug interactivo
    
    main()

