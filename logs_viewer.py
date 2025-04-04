import os
import uvicorn
import json
import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, List

app = FastAPI(title="SimuVerse Logs Viewer")

# Setup templates directory
templates_dir = "log_templates"
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Create static files directory for CSS, JS, etc.
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create CSS file
with open(f"{static_dir}/style.css", "w") as f:
    f.write("""
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f8f9fa;
    color: #333;
}

.container {
    display: flex;
    height: 100vh;
}

.sidebar {
    width: 250px;
    min-width: 250px; /* Prevent sidebar from shrinking */
    background-color: #2c3e50;
    color: white;
    padding: 20px;
    overflow-y: auto;
    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    flex-shrink: 0; /* Prevent sidebar from shrinking */
}

.content {
    flex-grow: 1;
    padding: 20px;
    overflow-y: auto;
    position: relative;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 15px;
}

.header h1 {
    margin: 0;
    color: #2c3e50;
}

.agent-list {
    margin-top: 20px;
    flex-grow: 1;
}

.agent-btn {
    display: block;
    width: 100%;
    padding: 12px;
    margin-bottom: 10px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    text-align: left;
    transition: background-color 0.2s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.agent-btn:hover {
    background-color: #2980b9;
}

.agent-btn.active {
    background-color: #e74c3c;
    font-weight: bold;
}

.log-entry {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 15px;
    margin-bottom: 15px;
    transition: transform 0.2s;
    border-left: 4px solid #3498db;
}

.log-entry:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.timestamp {
    color: #7f8c8d;
    font-size: 0.8em;
    margin-bottom: 5px;
}

.event-type {
    font-weight: bold;
    color: #3498db;
    margin-bottom: 10px;
    font-size: 1.1em;
}

.event-type.user_input {
    color: #e74c3c;
}

.event-type.response {
    color: #2ecc71;
}

.event-type.validation_failure {
    color: #f39c12;
}

.details {
    margin-top: 10px;
    white-space: pre-wrap;
    font-family: monospace;
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
}

.button {
    padding: 10px 15px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.2s;
}

.button:hover {
    background-color: #2980b9;
}

.button.refresh {
    background-color: #2ecc71;
}

.button.refresh:hover {
    background-color: #27ae60;
}

.button.clear {
    background-color: #e74c3c;
    margin-left: 10px;
}

.button.clear:hover {
    background-color: #c0392b;
}

.no-agent-selected {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 70vh;
    color: #7f8c8d;
}

.no-agent-selected i {
    font-size: 5em;
    margin-bottom: 20px;
    color: #bdc3c7;
}

.loading {
    text-align: center;
    padding: 40px;
    color: #7f8c8d;
}

.footer {
    margin-top: auto;
    font-size: 0.8em;
    text-align: center;
    padding-top: 20px;
    color: #ecf0f1;
}

.toggle-view {
    display: flex;
    margin-bottom: 15px;
}

.toggle-button {
    padding: 8px 15px;
    background-color: #ecf0f1;
    border: none;
    cursor: pointer;
}

.toggle-button.active {
    background-color: #3498db;
    color: white;
}

.toggle-button:first-child {
    border-radius: 4px 0 0 4px;
}

.toggle-button:last-child {
    border-radius: 0 4px 4px 0;
}

.chat-view .log-entry {
    max-width: 80%;
    margin-bottom: 15px;
}

.chat-view .event-type.user_input,
.chat-view .log-entry.user_input {
    background-color: #e8f4fd;
    border-left-color: #3498db;
    margin-right: auto;
}

.chat-view .event-type.response,
.chat-view .log-entry.response {
    background-color: #eafaf1;
    border-left-color: #2ecc71;
    margin-left: auto;
}

.actions-bar {
    position: -webkit-sticky;
    position: sticky;
    top: 0;
    background-color: rgba(248, 249, 250, 0.9);
    z-index: 100;
    padding: 10px 0;
    backdrop-filter: blur(5px);
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 20px;
}

@media (max-width: 768px) {
    .container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        max-height: 200px;
    }
    
    .chat-view .log-entry {
        max-width: 95%;
    }
}
""")

# Create HTML template for the chat interface
with open(f"{templates_dir}/logs.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>SimuVerse Logs</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <script>
        // Auto-refresh every 5 seconds
        let autoRefresh = false;
        let refreshInterval;
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('auto-refresh-btn');
            
            if (autoRefresh) {
                btn.innerHTML = '<i class="fas fa-pause"></i> Pause Auto-Refresh';
                btn.classList.add('active');
                refreshInterval = setInterval(() => {
                    fetchData();
                }, 5000);
            } else {
                btn.innerHTML = '<i class="fas fa-sync"></i> Enable Auto-Refresh';
                btn.classList.remove('active');
                clearInterval(refreshInterval);
            }
        }
        
        function fetchData() {
            const currentUrl = window.location.href;
            fetch(currentUrl)
                .then(response => response.text())
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    
                    // Update agent list
                    const oldSidebar = document.querySelector('.agent-list');
                    const newSidebar = doc.querySelector('.agent-list');
                    if (oldSidebar && newSidebar) {
                        oldSidebar.innerHTML = newSidebar.innerHTML;
                    }
                    
                    // Update content
                    const oldContent = document.querySelector('.log-entries');
                    const newContent = doc.querySelector('.log-entries');
                    if (oldContent && newContent) {
                        oldContent.innerHTML = newContent.innerHTML;
                    }
                });
        }
        
        function clearLogs() {
            if (confirm('Are you sure you want to clear all logs?')) {
                fetch('/clear_logs', {
                    method: 'POST'
                }).then(response => {
                    window.location.reload();
                });
            }
        }
        
        function toggleView(view) {
            const logEntries = document.querySelector('.log-entries');
            const standardBtn = document.getElementById('standard-view-btn');
            const chatBtn = document.getElementById('chat-view-btn');
            
            if (view === 'chat') {
                logEntries.classList.add('chat-view');
                chatBtn.classList.add('active');
                standardBtn.classList.remove('active');
                localStorage.setItem('viewMode', 'chat');
            } else {
                logEntries.classList.remove('chat-view');
                standardBtn.classList.add('active');
                chatBtn.classList.remove('active');
                localStorage.setItem('viewMode', 'standard');
            }
        }
        
        window.onload = function() {
            // Set up view mode based on localStorage
            const viewMode = localStorage.getItem('viewMode') || 'standard';
            toggleView(viewMode);
            
            // Scroll to bottom for chat view
            if (viewMode === 'chat') {
                const logEntries = document.querySelector('.log-entries');
                if (logEntries) {
                    logEntries.scrollTop = logEntries.scrollHeight;
                }
            }
        };
    </script>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>SimuVerse Logs</h2>
            <button class="button refresh" onclick="window.location.reload()">
                <i class="fas fa-sync"></i> Refresh
            </button>
            <div class="agent-list">
                <h3>Agents</h3>
                {% for agent_id in agents %}
                    <button class="agent-btn {% if agent_id == selected_agent %}active{% endif %}" 
                            onclick="window.location.href='?agent={{ agent_id }}'">
                        <i class="fas fa-user-astronaut"></i> {{ agent_id }}
                    </button>
                {% endfor %}
            </div>
            <div class="footer">
                SimuVerse Framework v1.0
            </div>
        </div>
        
        <div class="content">
            {% if selected_agent %}
                <div class="header">
                    <h1><i class="fas fa-user-astronaut"></i> Agent: {{ selected_agent }}</h1>
                    <div>
                        <button id="auto-refresh-btn" class="button" onclick="toggleAutoRefresh()">
                            <i class="fas fa-sync"></i> Enable Auto-Refresh
                        </button>
                        <button class="button clear" onclick="clearLogs()">
                            <i class="fas fa-trash"></i> Clear Logs
                        </button>
                    </div>
                </div>
                
                <div class="actions-bar">
                    <div class="toggle-view">
                        <button id="standard-view-btn" class="toggle-button active" onclick="toggleView('standard')">
                            <i class="fas fa-list"></i> Standard View
                        </button>
                        <button id="chat-view-btn" class="toggle-button" onclick="toggleView('chat')">
                            <i class="fas fa-comments"></i> Chat View
                        </button>
                    </div>
                </div>
                
                <div class="log-entries">
                    {% for log in agent_logs %}
                        <div class="log-entry {{ log.type }}">
                            <div class="timestamp">{{ log.timestamp }}</div>
                            <div class="event-type {{ log.type }}">{{ log.type }}</div>
                            {% if log.type == 'user_input' %}
                                <div><strong>User:</strong> {{ log.details.input }}</div>
                            {% elif log.type == 'response' %}
                                <div><strong>Assistant:</strong> {{ log.details.text }}</div>
                                <div><strong>Action:</strong> {{ log.details.action }} {% if log.details.location %}({{ log.details.location }}){% endif %}</div>
                            {% else %}
                                <pre class="details">{{ log.details|string }}</pre>
                            {% endif %}
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-agent-selected">
                    <i class="fas fa-user-astronaut"></i>
                    <h2>Select an agent from the sidebar to view logs</h2>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
""")

# Logs file path
logs_file = "agent_logs.json"

def load_logs():
    """Load logs from file"""
    if os.path.exists(logs_file):
        try:
            with open(logs_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading logs: {e}")
    return {}

@app.get("/", response_class=HTMLResponse)
async def view_logs(request: Request):
    logs = load_logs()
    selected_agent = request.query_params.get("agent")
    agent_logs = []
    
    if selected_agent and selected_agent in logs:
        agent_logs = logs[selected_agent]
    
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "agents": logs.keys(),
        "selected_agent": selected_agent,
        "agent_logs": agent_logs
    })

@app.post("/clear_logs")
async def clear_logs():
    # Clear logs file
    with open(logs_file, "w") as f:
        json.dump({}, f)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3001)