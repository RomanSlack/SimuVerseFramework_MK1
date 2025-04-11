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

# In-memory session storage per agent.
sessions: Dict[str, List[Dict[str, str]]] = {}

# Simple logging system.
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
# The system prompt is added only if the session is new; also append current task.
# ----------------------------------------------------------------------------
def get_or_create_session(agent_id: str, system_prompt: str, task: str) -> List[Dict[str, str]]:
    if agent_id not in sessions:
        full_prompt = system_prompt
        if task.strip():
            full_prompt += "\nCurrent Task: " + task
        sessions[agent_id] = [{"role": "system", "content": full_prompt}]
    return sessions[agent_id]

# ----------------------------------------------------------------------------
# Utility: Build the LLM prompt from conversation.
# If there are forwarded messages, include a "Conversation Context:" section.
# ----------------------------------------------------------------------------
def build_prompt(conversation: List[Dict[str, str]]) -> str:
    context_lines = [msg["content"] for msg in conversation if msg["content"].startswith("[Forwarded")]
    normal_lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation if not msg["content"].startswith("[Forwarded")]
    prompt_lines = []
    if context_lines:
        prompt_lines.append("Conversation Context:")
        prompt_lines.extend(context_lines)
    prompt_lines.extend(normal_lines)
    prompt_lines.append("Assistant:")
    prompt_lines.append("Remember: Provide at least one sentence of reasoning and then end your response with MOVE:, NOTHING:, or CONVERSE: (with no extra text).")
    return "\n".join(prompt_lines)

# ----------------------------------------------------------------------------
# OpenAI ChatGPT Wrapper with high temperature.
# ----------------------------------------------------------------------------
class OpenAIChatGPT:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
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
    system_prompt: str  # Provided only on first request.
    task: str           # Current task from the agent.

class GenerateResponse(BaseModel):
    agent_id: str
    text: str         # Full AI response.
    action: str       # "move", "nothing", "converse", or "none"
    location: str     # For MOVE: a location or agent name; for CONVERSE: target agent's name.

# ----------------------------------------------------------------------------
# FastAPI Endpoint for generating a response.
# ----------------------------------------------------------------------------
@app.post("/generate", response_model=GenerateResponse)
def generate_response(data: GenerateRequest):
    log_event(data.agent_id, "user_input", {
        "input": data.user_input,
        "system_prompt": data.system_prompt,
        "task": data.task
    })
    
    # Get or create session (inject system prompt and task only if new)
    conversation = get_or_create_session(data.agent_id, data.system_prompt, data.task)
    conversation.append({"role": "user", "content": data.user_input})
    prompt = build_prompt(conversation)
    log_event(data.agent_id, "prompt_built", {"prompt": prompt})
    
    llm = OpenAIChatGPT(api_key=OPENAI_API_KEY)
    assistant_text = llm.generate(prompt)
    
    # Validate: Ensure at least one reasoning line and a proper final command.
    lines = assistant_text.strip().split("\n")
    if len(lines) < 2:
        assistant_text = ("Your response is invalid. You must provide at least one sentence of reasoning.\n"
                          "NOTHING: do nothing")
        log_event(data.agent_id, "validation_failure", {"reason": "Not enough lines"})
    else:
        final_line = lines[-1].strip().lower()
        if not (final_line.startswith("move:") or final_line.startswith("nothing:") or final_line.startswith("converse:")):
            assistant_text = ("Your final line did not start with MOVE:, NOTHING:, or CONVERSE:. Invalid response.\n"
                              "NOTHING: do nothing")
            log_event(data.agent_id, "validation_failure", {"reason": "Bad final line"})
    
    conversation.append({"role": "assistant", "content": assistant_text})
    
    # Parse the final command.
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

    # If action is CONVERSE, forward the conversation to the target agent's session.
    if action == "converse" and location:
        get_or_create_session(location, "", "")  # Ensure target agent session exists.
        fwd_text = f"[Forwarded from {data.agent_id}]: {assistant_text}"
        sessions[location].append({"role": "user", "content": fwd_text})
        log_event(data.agent_id, "conversation_forwarded", {"to_agent": location, "fwd_text": fwd_text})
    
    log_event(data.agent_id, "response", {"assistant_text": assistant_text, "action": action, "location": location})
    return GenerateResponse(agent_id=data.agent_id, text=assistant_text, action=action, location=location.lower())

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
