"""
Verificador RLVR: compara respuesta final con gold y devuelve reward 0/1
"""
import re
from typing import Optional


def parse_gold(answer_text: str) -> Optional[str]:
    """
    Extrae la respuesta numérica del texto gold (formato GSM8K).
    """
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return nums[-1].strip() if nums else None


def parse_pred(text: str) -> Optional[str]:
    """
    Extrae la respuesta numérica del texto generado (busca entre brackets).
    """
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1].strip() if nums else None


def compute_reward(predicted_text: str, gold_answer: str) -> float:
    """
    Computa reward verificable (0/1) comparando respuesta predicha con gold.
    
    Args:
        predicted_text: Texto generado por el Agent B (debe contener [número])
        gold_answer: Respuesta gold del dataset (formato GSM8K con #### número)
    
    Returns:
        Reward: 1.0 si coincide, 0.0 si no
    """
    pred = parse_pred(predicted_text)
    gold = parse_gold(gold_answer)
    
    if pred is None or gold is None:
        return 0.0
    
    return 1.0 if pred == gold else 0.0

