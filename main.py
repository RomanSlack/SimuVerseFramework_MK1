import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

app = FastAPI()

# Store conversation per agent
sessions: Dict[str, List[Dict[str, str]]] = {}

def get_or_create_session(agent_id: str, system_prompt: str) -> List[Dict[str, str]]:
    """
    Creates a conversation session for each agent if not present,
    starting with the system prompt from Unity.
    """
    if agent_id not in sessions:
        sessions[agent_id] = [{"role": "system", "content": system_prompt}]
    return sessions[agent_id]

def build_prompt(conversation: List[Dict[str, str]]) -> str:
    """
    Convert the conversation list into a single prompt string.
    We'll tack on a final message instructing the LLM to
    finish with exactly one line that starts with:
      MOVE:, NOTHING:, or CONVERSE:
    """
    prompt_lines = []
    for msg in conversation:
        # "role" is either system, user, assistant
        prompt_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    # Add a final line instructing the LLM to produce one final line
    prompt_lines.append(
        "Assistant:\nRemember you must provide at least one sentence of reasoning "
        "and end with a line that starts with MOVE:, NOTHING:, or CONVERSE:."
    )
    return "\n".join(prompt_lines)

# ----------------------------------------------------------------------------
# OpenAI Wrapper
# ----------------------------------------------------------------------------

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
            temperature=1.0
        )
        return response.choices[0].message.content

# ----------------------------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    agent_id: str
    user_input: str
    system_prompt: str  # No default – must come from Unity.

class GenerateResponse(BaseModel):
    agent_id: str
    text: str
    action: str   # "move", "nothing", "converse", or "none" if invalid
    location: str # location or agent name

# ----------------------------------------------------------------------------
# FastAPI Endpoint
# ----------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
def generate_response(data: GenerateRequest):
    # 1) Retrieve or create a conversation list
    conversation = get_or_create_session(data.agent_id, data.system_prompt)

    # 2) Append user input to the conversation
    conversation.append({"role": "user", "content": data.user_input})

    # 3) Build the prompt
    prompt = build_prompt(conversation)

    # 4) Call the LLM
    llm = OpenAIChatGPT(api_key=OPENAI_API_KEY)
    assistant_text = llm.generate(prompt)

    # 5) Basic validation
    lines = assistant_text.strip().split("\n")
    if len(lines) < 2:
        # Not enough lines => no reasoning
        assistant_text = (
            "Your response is invalid. You must provide at least one sentence of reasoning.\n"
            "NOTHING: do nothing"
        )
    else:
        # Check final line
        final_line = lines[-1].strip().lower()
        valid_starts = ["move:", "nothing:", "converse:"]
        if not any(final_line.startswith(x) for x in valid_starts):
            assistant_text = (
                "Your final line did not start with MOVE:, NOTHING:, or CONVERSE:. "
                "Invalid response.\nNOTHING: do nothing"
            )

    # 6) Append the final assistant text
    conversation.append({"role": "assistant", "content": assistant_text})

    # 7) Parse out the action + location
    action = "none"
    location = ""
    for line in assistant_text.splitlines():
        l = line.strip().lower()
        if l.startswith("move:"):
            action = "move"
            location = line.split(":", 1)[1].strip()
            break
        elif l.startswith("nothing:"):
            action = "nothing"
            break
        elif l.startswith("converse:"):
            action = "converse"
            location = line.split(":", 1)[1].strip()
            break

    return GenerateResponse(
        agent_id=data.agent_id,
        text=assistant_text,
        action=action,
        location=location.lower()  # unify to lower if you want
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)
