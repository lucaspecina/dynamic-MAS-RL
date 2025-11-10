LLM que cree dinamicamente sistema multi-agentes, entrenado con RL.

# Proyecto

**Orquestador de sistemas multi-agente entrenado con RL (reward verificable).**
La idea es que un LLM “orquestador” no solo responda, sino **diseñe** (en texto/JSON) el **grafo de agentes** adecuado para cada tarea: qué roles hay, qué prompt usa cada uno y cómo se conectan. Entrenamos al orquestador con **RLVR** (recompensas verificables, 0/1) para que aprenda a elegir estructuras que realmente resuelven el problema, evitando “reward hacking”. Esto escala luego a **multi-turn**, **tool-calling** y **LLM-as-judge** como señal auxiliar.

# Mini ejemplo (toy) que estamos corriendo ahora

* **Tarea:** problemas simples tipo **GSM8K**.
* **Grafo mínimo (2 agentes en serie):**

  * **Agent A (planner):** escribe un borrador/plan breve.
  * **Agent B (solver):** usa problema + borrador y devuelve `FINAL: <número>`.
* **Orquestador (el que entrenamos con RL):** **genera en TEXTO** un JSON con los dos agentes, sus prompts, y el cableado (`A → B`).
* **Runner:** ejecuta ese JSON con nuestros modelos locales y obtiene la respuesta final.
* **Verificador (RLVR):** extrae el número final y compara con el gold → **reward 0/1**.
* **Entrenamiento (siguiente paso inmediato):** **GRPO** con varias trayectorias por problema y **LoRA** sobre el orquestador (regularizado con KL).

## Por qué importa

* Fuerza al sistema a **aprender a estructurar** la solución (no solo a “adivinar”).
* Es **escalable**: del 2-agentes pasamos a debates, herramientas y criterios múltiples (rubrics/judge), manteniendo el reward principal verificable.