"""
Runner: ejecuta el grafo de agentes generado por el orquestador
"""
from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


# Variable global para el modelo y tokenizer de los agentes
_agent_model = None
_agent_tokenizer = None


def set_agent_llm(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Configura el modelo y tokenizer para los agentes."""
    global _agent_model, _agent_tokenizer
    _agent_model = model
    _agent_tokenizer = tokenizer


@torch.no_grad()
def llm_complete(messages: list[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Usa el modelo local cargado en memoria para generar respuestas.
    """
    global _agent_model, _agent_tokenizer
    
    if _agent_model is None or _agent_tokenizer is None:
        raise ValueError("Modelo de agentes no configurado. Llama a set_agent_llm(model, tokenizer) primero.")
    
    _agent_model.eval()
    _agent_model.config.use_cache = True
    model_device = next(_agent_model.parameters()).device
    
    # Convertir mensajes a prompt
    try:
        prompt = _agent_tokenizer.apply_chat_template(
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
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        prompt = "".join(prompt_parts) + "Assistant: "
    
    # Tokenizar y generar
    enc = _agent_tokenizer([prompt], return_tensors="pt").to(model_device)
    
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        use_cache=True,
        pad_token_id=_agent_tokenizer.pad_token_id,
        eos_token_id=_agent_tokenizer.eos_token_id
    )
    
    if temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    
    with torch.no_grad():
        out = _agent_model.generate(**enc, **gen_kwargs)
    
    # Decodificar solo los tokens generados
    generated_tokens = out[0][enc['input_ids'].shape[1]:]
    text = _agent_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return text.strip()


def _render_user_inputs(input_map: Dict[str, Any]) -> str:
    """Render simple para el mensaje del 'user' a cada agente."""
    lines = []
    for k, v in input_map.items():
        lines.append(f"{k.upper()}: {v}")
    return "\n".join(lines)


def _agent_call_llm(agent: Dict[str, Any], input_payload: Dict[str, Any], temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Construye mensajes (system + user) y llama al LLM."""
    sys_prompt = agent["system_prompt"]
    user_text = _render_user_inputs(input_payload)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text}
    ]
    return llm_complete(messages, temperature=temperature, max_tokens=max_tokens)


def run_two_agent_graph(
    graph_spec: Dict[str, Any],
    problem_text: str,
    temperature: float = 0.2,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Ejecuta el grafo de 2 agentes en serie con flujo fijo.
    Flujo: problem → Agent A → (problem + draft) → Agent B → final answer
    
    Args:
        graph_spec: Dict con solo "agents" (lista de 2 agentes con id, type, system_prompt)
        problem_text: El problema matemático a resolver
        temperature: Temperatura para generación de agentes
        max_tokens: Máximo de tokens a generar por agente
    
    Returns:
        Dict con 'trace' y 'final_text'
    """
    assert isinstance(graph_spec, dict), "graph_spec debe ser dict"
    assert "agents" in graph_spec and isinstance(graph_spec["agents"], list), "agents faltante"
    assert len(graph_spec["agents"]) == 2, "v0: exactamente 2 agentes"
    
    agent_a = graph_spec["agents"][0]
    agent_b = graph_spec["agents"][1]
    assert agent_a["id"] == "agentA", "Primer agente debe ser agentA"
    assert agent_b["id"] == "agentB", "Segundo agente debe ser agentB"
    
    trace: Dict[str, Any] = {"agents": {}}
    
    # Ejecutar Agent A: recibe solo el problema
    payload_a = {"problem": problem_text}
    draft = _agent_call_llm(agent_a, payload_a, temperature=temperature, max_tokens=max_tokens)
    
    trace["agents"]["agentA"] = {
        "system_prompt": agent_a["system_prompt"],
        "input_payload": payload_a,
        "output_text": draft
    }
    
    # Ejecutar Agent B: recibe problema + draft de A
    payload_b = {
        "problem": problem_text,
        "draft": draft
    }
    final_answer = _agent_call_llm(agent_b, payload_b, temperature=temperature, max_tokens=max_tokens)
    
    trace["agents"]["agentB"] = {
        "system_prompt": agent_b["system_prompt"],
        "input_payload": payload_b,
        "output_text": final_answer
    }
    
    return {"trace": trace, "final_text": final_answer}

