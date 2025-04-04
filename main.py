import os
import uvicorn
import json
import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

app = FastAPI()

# ----------------------------------------------------------------------------
# In-memory conversation sessions and logs
# ----------------------------------------------------------------------------
sessions: Dict[str, List[Dict[str, str]]] = {}
logs: Dict[str, List[Dict[str, Any]]] = {}
logs_file = "agent_logs.json"

def save_logs():
    with open(logs_file, "w") as f:
        json.dump(logs, f, indent=2)

def log_event(agent_id: str, event_type: str, details: Dict[str, Any]):
    if agent_id not in logs:
        logs[agent_id] = []
    logs[agent_id].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "details": details
    })
    save_logs()

# ----------------------------------------------------------------------------
# Utility: Get or create a new session for the agent.
# Only add the system prompt if the session is new.
# ----------------------------------------------------------------------------
def get_or_create_session(agent_id: str, system_prompt: str) -> List[Dict[str, str]]:
    if agent_id not in sessions:
        sessions[agent_id] = []
        if system_prompt.strip():
            sessions[agent_id].append({"role": "system", "content": system_prompt})
    return sessions[agent_id]

# ----------------------------------------------------------------------------
# Utility: Build the LLM prompt from conversation
# ----------------------------------------------------------------------------
def build_prompt(conversation: List[Dict[str, str]]) -> str:
    """
    Combine conversation messages into a single string. 
    Then add a final instruction about at least one line of reasoning 
    and ending with MOVE:, NOTHING:, or CONVERSE:.
    """
    lines = []
    for msg in conversation:
        lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    lines.append("Assistant:")
    lines.append("Remember: Provide at least one sentence of reasoning then end your answer with MOVE:, NOTHING:, or CONVERSE:.")
    return "\n".join(lines)

# ----------------------------------------------------------------------------
# Simple OpenAI LLM wrapper
# ----------------------------------------------------------------------------
class OpenAIChatGPT:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=1.0
        )
        return resp.choices[0].message.content

# ----------------------------------------------------------------------------
# Request and Response models
# ----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    agent_id: str
    user_input: str
    system_prompt: str

class GenerateResponse(BaseModel):
    agent_id: str
    text: str
    action: str  # move, nothing, converse, or none
    location: str

# ----------------------------------------------------------------------------
# The main /generate endpoint
# ----------------------------------------------------------------------------
@app.post("/generate", response_model=GenerateResponse)
def generate_response(data: GenerateRequest):
    # Log incoming user input
    log_event(data.agent_id, "user_input", {
        "input": data.user_input,
        "system_prompt": data.system_prompt
    })

    # 1) Get or create session
    conversation = get_or_create_session(data.agent_id, data.system_prompt)

    # 2) Append user's new message
    conversation.append({"role": "user", "content": data.user_input})

    # 3) Build the prompt
    prompt = build_prompt(conversation)
    log_event(data.agent_id, "prompt_built", {"prompt": prompt})

    # 4) Call LLM
    llm = OpenAIChatGPT(api_key=OPENAI_API_KEY)
    assistant_text = llm.generate(prompt)

    # 5) Validate final line
    lines = assistant_text.strip().split("\n")
    if len(lines) < 2:
        assistant_text = (
            "Your response is invalid. You must provide at least one sentence of reasoning.\n"
            "NOTHING: do nothing"
        )
        log_event(data.agent_id, "validation_failure", {"reason": "Not enough lines"})
    else:
        final_line = lines[-1].strip().lower()
        if not (final_line.startswith("move:") or final_line.startswith("nothing:") or final_line.startswith("converse:")):
            assistant_text = (
                "Your final line did not start with MOVE:, NOTHING:, or CONVERSE:. Invalid response.\n"
                "NOTHING: do nothing"
            )
            log_event(data.agent_id, "validation_failure", {"reason": "Bad final line"})

    # 6) Append assistant's final text to conversation
    conversation.append({"role": "assistant", "content": assistant_text})

    # 7) Parse out action & location
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

    # If the action is CONVERSE, we forward the text to the target agent's session
    if action == "converse" and location:
        # ensure the target agent session exists
        get_or_create_session(location, "")
        # forward a "user" message from the original agent
        fwd_text = f"[Forwarded from {data.agent_id}]: {assistant_text}"
        sessions[location].append({"role": "user", "content": fwd_text})
        log_event(data.agent_id, "conversation_forwarded", {
            "to_agent": location,
            "fwd_text": fwd_text
        })

    # 8) Log final response
    log_event(data.agent_id, "response", {
        "assistant_text": assistant_text,
        "action": action,
        "location": location
    })

    return GenerateResponse(
        agent_id=data.agent_id,
        text=assistant_text,
        action=action,
        location=location.lower()
    )

# ----------------------------------------------------------------------------
# Endpoint to reset system
# ----------------------------------------------------------------------------
@app.post("/reset")
def reset_system():
    global sessions, logs
    sessions = {}
    logs = {}
    with open(logs_file, "w") as f:
        json.dump(logs, f)
    return {"status": "ok", "message": "All sessions & logs cleared."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)
