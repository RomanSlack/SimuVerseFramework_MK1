# SimuVerse Framework

The framework powering the SimuVerse simulation.

## Setup

1. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Install requirements:
```bash
pip install fastapi uvicorn openai python-dotenv
```

## Running the Framework

1. Start the main API server:
```bash
python main.py
```

This will start a FastAPI server on http://127.0.0.1:3000 that provides the agent interaction API.

2. Start the logs viewer interface (in a separate terminal):
```bash
python logs_viewer.py
```

This will start a separate web interface on http://127.0.0.1:3001 that shows logs for all agents in a nice, user-friendly interface.

## API Endpoints

### Main API (port 3000)

- `POST /generate` - Generate a response from an agent
- `GET /api/logs` - Get all logs in JSON format
- `GET /api/logs/{agent_id}` - Get logs for a specific agent in JSON format
- `POST /clear_logs` - Clear all logs
- `POST /reset` - Reset the entire system (clear sessions and logs)

### Logs Viewer (port 3001)

- `GET /` - Web interface for viewing logs
- `POST /clear_logs` - Clear all logs

## Logs Viewer Features

- View logs for all agents in a clean, user-friendly interface
- Switch between standard and chat views
- Auto-refresh functionality to see logs in real-time
- Clear logs with a single click
- Mobile-responsive design

## How Logs Work

1. The system logs all agent interactions to an `agent_logs.json` file
2. Logs are organized by agent ID
3. Each log entry includes:
   - Timestamp
   - Event type (user_input, generating_response, response, validation_failure)
   - Detailed information specific to the event type
4. Logs persist between server restarts but can be cleared via the API or UI

## Example Agent Interaction

```python
import requests

# Generate a response from an agent
response = requests.post(
    "http://127.0.0.1:3000/generate", 
    json={
        "agent_id": "agent1",
        "user_input": "Hello, how are you?",
        "system_prompt": "You are a helpful assistant."
    }
)

print(response.json())
```

## Model

Current model: gpt-4o-mini-2024-07-18