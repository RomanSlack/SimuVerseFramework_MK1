import os
import uvicorn
import threading
import time
import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

# Logging system
logs: Dict[str, List[Dict[str, Any]]] = {}

def log_event(agent_id: str, event_type: str, details: Dict[str, Any]):
    """
    Log an event for a specific agent with timestamp
    """
    if agent_id not in logs:
        logs[agent_id] = []
    
    logs[agent_id].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "details": details
    })

# Setup templates directory for the chat interface
templates = Jinja2Templates(directory="templates")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create the HTML template for the chat interface
with open("templates/logs.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>SimuVerse Logs</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }
        #sidebar {
            width: 200px;
            background-color: #f0f0f0;
            padding: 20px;
            overflow-y: auto;
        }
        #content {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
        }
        .agent-btn {
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .log-entry {
            border-bottom: 1px solid #ddd;
            padding: 10px 0;
        }
        .timestamp {
            color: #888;
            font-size: 0.8em;
        }
        .event-type {
            font-weight: bold;
            color: #4CAF50;
        }
        .details {
            margin-top: 5px;
            white-space: pre-wrap;
        }
        #refresh-btn {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <button id="refresh-btn" onclick="location.reload()">Refresh</button>
        <h3>Agents</h3>
        {% for agent_id in logs %}
            <button class="agent-btn" onclick="location.href='?agent={{ agent_id }}'">{{ agent_id }}</button>
        {% endfor %}
    </div>
    <div id="content">
        {% if selected_agent %}
            <h2>Logs for Agent: {{ selected_agent }}</h2>
            {% for log in agent_logs %}
                <div class="log-entry">
                    <div class="timestamp">{{ log.timestamp }}</div>
                    <div class="event-type">{{ log.type }}</div>
                    <div class="details">{{ log.details|string }}</div>
                </div>
            {% endfor %}
        {% else %}
            <h2>Select an agent from the sidebar to view logs</h2>
        {% endif %}
    </div>
</body>
</html>
    """)

# Add a route to clear logs
@app.post("/clear_logs")
def clear_logs():
    global logs
    logs = {}
    return {"status": "success", "message": "All logs cleared"}

def get_or_create_session(agent_id: str, system_prompt: str) -> List[Dict[str, str]]:
    """
    Create or retrieve a conversation session for the given agent,
    starting with the provided system prompt.
    """
    if agent_id not in sessions:
        sessions[agent_id] = [{"role": "system", "content": system_prompt}]
    return sessions[agent_id]

def build_prompt(conversation: List[Dict[str, str]]) -> str:
    """
    Convert the conversation history to a single prompt.
    Append an instruction that the final line must begin with MOVE:, NOTHING:, or CONVERSE:.
    """
    prompt_lines = []
    for msg in conversation:
        prompt_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    prompt_lines.append("Assistant:")
    prompt_lines.append("Remember: Provide at least one sentence of reasoning and end with a final line that starts with MOVE:, NOTHING:, or CONVERSE: (with no extra text).")
    return "\n".join(prompt_lines)

# ----------------------------------------------------------------------------
# OpenAI ChatGPT Wrapper with higher temperature for more creativity.
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
# Pydantic Models for Request/Response.
# ----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    agent_id: str
    user_input: str
    system_prompt: str  # Sent from Unity (includes complete instructions & personality)

class GenerateResponse(BaseModel):
    agent_id: str
    text: str         # Full AI response
    action: str       # "move", "nothing", "converse" or "none"
    location: str     # For MOVE: one of the valid locations or an agent name; for CONVERSE: the target agent's name

# ----------------------------------------------------------------------------
# FastAPI Endpoint.
# ----------------------------------------------------------------------------
@app.post("/generate", response_model=GenerateResponse)
def generate_response(data: GenerateRequest):
    # Log user input
    log_event(data.agent_id, "user_input", {
        "input": data.user_input,
        "system_prompt": data.system_prompt
    })
    
    conversation = get_or_create_session(data.agent_id, data.system_prompt)
    conversation.append({"role": "user", "content": data.user_input})
    prompt = build_prompt(conversation)
    
    # Log that we're generating a response
    log_event(data.agent_id, "generating_response", {
        "prompt": prompt
    })
    
    llm = OpenAIChatGPT(api_key=OPENAI_API_KEY)
    assistant_text = llm.generate(prompt)
    
    # Validate that there's at least one reasoning line and a proper final line.
    lines = assistant_text.strip().split("\n")
    if len(lines) < 2:
        assistant_text = (
            "Your response is invalid. You must provide at least one sentence of reasoning.\n"
            "NOTHING: do nothing"
        )
        # Log validation failure
        log_event(data.agent_id, "validation_failure", {
            "reason": "Not enough lines",
            "response": assistant_text
        })
    else:
        final_line = lines[-1].strip().lower()
        valid_starts = ["move:", "nothing:", "converse:"]
        if not any(final_line.startswith(x) for x in valid_starts):
            assistant_text = (
                "Your final line did not start with MOVE:, NOTHING:, or CONVERSE:. Invalid response.\n"
                "NOTHING: do nothing"
            )
            # Log validation failure
            log_event(data.agent_id, "validation_failure", {
                "reason": "Invalid final line",
                "response": assistant_text
            })
    
    conversation.append({"role": "assistant", "content": assistant_text})
    
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
    
    # Log the final response
    log_event(data.agent_id, "response", {
        "text": assistant_text,
        "action": action,
        "location": location
    })
    
    return GenerateResponse(
        agent_id=data.agent_id,
        text=assistant_text,
        action=action,
        location=location.lower()
    )

# Add a route for the logs view
@app.get("/logs", response_class=HTMLResponse)
async def view_logs(request: Request):
    selected_agent = request.query_params.get("agent")
    agent_logs = []
    
    if selected_agent and selected_agent in logs:
        agent_logs = logs[selected_agent]
    
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs": logs,
        "selected_agent": selected_agent,
        "agent_logs": agent_logs
    })

# Add a route to reset sessions and logs
@app.post("/reset")
def reset_system():
    global sessions, logs
    sessions = {}
    logs = {}
    return {"status": "success", "message": "System reset successfully"}

if __name__ == "__main__":
    # Run the app on a separate port (3001) for the logs interface
    # This allows the main API to run on port 3000
    import threading
    
    def run_main_app():
        uvicorn.run(app, host="127.0.0.1", port=3000)
    
    def run_logs_app():
        uvicorn.run(app, host="127.0.0.1", port=3001)
    
    # Start both servers in separate threads
    main_thread = threading.Thread(target=run_main_app, daemon=True)
    logs_thread = threading.Thread(target=run_logs_app, daemon=True)
    
    main_thread.start()
    logs_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers...")
        # The daemon threads will be terminated when the main thread exits
