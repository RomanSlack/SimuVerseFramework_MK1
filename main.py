import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Set your OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

app = FastAPI()

# In-memory session storage per agent
sessions: Dict[str, List[Dict[str, str]]] = {}

# -------------------------------------------------------------------------
# STRONGER GOALS in the system prompt:
# We instruct the agent to collaborate to find a missing part ("O2 regulator").
# Also remind them to move, do nothing, or converse.
# -------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """\
You are a game agent. You have a primary goal: collaborate with other agents to find the missing O2 regulator part on this Mars base.
You can MOVE to exactly one of these four locations: park, library, 02_Regulator_Room, gym (or move to another agent).
You can choose to do NOTHING, or you can CONVERSE with another agent. 
You must always provide at least one sentence of reasoning before your final line. 
Your final line must be in one of the forms:
MOVE: <location or agent_name>
NOTHING: do nothing
CONVERSE: <agent_name>
"""

def get_or_create_session(agent_id: str, system_prompt: str) -> List[Dict[str, str]]:
    if agent_id not in sessions:
        sessions[agent_id] = [{"role": "system", "content": system_prompt}]
    return sessions[agent_id]

def build_prompt(conversation: List[Dict[str, str]]) -> str:
    """
    Build a single prompt string from the conversation history.
    Append a mandatory instruction to force the final line.
    """
    prompt_lines = []
    for msg in conversation:
        prompt_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    prompt_lines.append("Assistant:")
    prompt_lines.append(
        "Remember to provide at least one sentence of reasoning, "
        "then end with exactly one line that starts with MOVE:, NOTHING:, or CONVERSE:."
    )
    return "\n".join(prompt_lines)

# =============================================================================
# OpenAI ChatGPT Wrapper (with higher temperature)
# =============================================================================

class OpenAIChatGPT:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        from openai import OpenAI
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=1.0,  # HIGHER TEMPERATURE for more creativity
        )
        return response.choices[0].message.content

# =============================================================================
# Pydantic Models
# =============================================================================

class GenerateRequest(BaseModel):
    agent_id: str
    user_input: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

class GenerateResponse(BaseModel):
    agent_id: str
    text: str    # Full response text from the LLM
    action: str  # "move", "nothing", or "converse"
    location: str  # for "move": one of "park", "library", "02_Regulator_Room", "gym" or agent name; for "converse": agent name

# =============================================================================
# FastAPI Endpoint
# =============================================================================

@app.post("/generate", response_model=GenerateResponse)
def generate_response(data: GenerateRequest):
    # Get or create conversation
    conversation = get_or_create_session(data.agent_id, data.system_prompt)

    # Add user input
    conversation.append({"role": "user", "content": data.user_input})

    # Build prompt
    prompt = build_prompt(conversation)

    # Generate LLM response
    llm = OpenAIChatGPT(api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18")
    assistant_text = llm.generate(prompt)

    # Basic validation: ensure there's at least one line of reasoning
    lines = assistant_text.strip().split("\n")
    if len(lines) < 2:
        # Not enough lines => no reasoning
        assistant_text = "Your response is invalid. You must provide at least one sentence of reasoning.\nNOTHING: do nothing"
    else:
        # Check final line is one of [MOVE:, NOTHING:, CONVERSE:]
        final_line = lines[-1].strip().lower()
        if not (final_line.startswith("move:") or final_line.startswith("nothing:") or final_line.startswith("converse:")):
            # Force a retry
            assistant_text = "Your final line did not start with MOVE:, NOTHING:, or CONVERSE:. Invalid response.\nNOTHING: do nothing"

    # Add the assistant's text to conversation
    conversation.append({"role": "assistant", "content": assistant_text})

    # Parse out the action and location
    action = "none"
    location = ""
    for line in assistant_text.splitlines():
        l = line.strip().lower()
        if l.startswith("move:"):
            action = "move"
            loc = line.split(":", 1)[1].strip().lower()
            location = loc
            break
        elif l.startswith("nothing:"):
            action = "nothing"
            break
        elif l.startswith("converse:"):
            action = "converse"
            loc = line.split(":", 1)[1].strip()
            location = loc
            break

    return GenerateResponse(
        agent_id=data.agent_id,
        text=assistant_text,
        action=action,
        location=location
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)
