"""
Utilidades para cargar modelos y tokenizers
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def load_tokenizers(orchestrator_id: str, agent_model_id: str):
    """
    Carga los tokenizers para orquestador y agentes.
    """
    tok_orch = AutoTokenizer.from_pretrained(orchestrator_id, use_fast=True)
    tok_agent = AutoTokenizer.from_pretrained(agent_model_id, use_fast=True)
    for tok in (tok_orch, tok_agent):
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # decoder-only: left padding
    return tok_orch, tok_agent


def make_lora_orchestrator(model_id: str) -> torch.nn.Module:
    """
    Crea un modelo orquestador con LoRA adapters (el que se entrena).
    """
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base.config.use_cache = False  # off for training
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    return get_peft_model(base, lora_cfg)


def load_agent_model(model_id: str) -> torch.nn.Module:
    """
    Carga el modelo usado por los agentes (frozen, no se entrena).
    """
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m

