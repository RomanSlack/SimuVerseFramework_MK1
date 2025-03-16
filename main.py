import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Set your OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

app = FastAPI()

# Store conversation per agent (in-memory)
sessions: Dict[str, List[Dict[str, str]]] = {}

DEFAULT_SYSTEM_PROMPT = """\
You are a game agent. You have a mood (provided by the user).
You can only choose to move to exactly one of these four locations:
- park
- library
- home
- gym

Your response should contain some brief explanation in natural language,
but MUST end with a line of the form:
MOVE: <location>

Where <location> is exactly one of [park, library, home, gym].
Example:
"I feel like reading, so the library is best."
MOVE: library
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
        # Capitalize the role for clarity (System, User, Assistant)
        prompt_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
    # Append a final instruction to guarantee the format.
    prompt_lines.append("Assistant:") 
    prompt_lines.append("Please ensure your final line is exactly in the format: MOVE: <location>")
    return "\n".join(prompt_lines)


# =============================================================================
# New OpenAI ChatGPT Class using updated client interface
# =============================================================================

class OpenAIChatGPT:
    """
    A wrapper for the new OpenAI ChatCompletion interface.
    This class builds a single prompt string and returns the assistant response.
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        # Import the OpenAI client from the new API package
        from openai import OpenAI
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content


# =============================================================================
# Pydantic Models for the Endpoint
# =============================================================================

class GenerateRequest(BaseModel):
    agent_id: str
    user_input: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

class GenerateResponse(BaseModel):
    agent_id: str
    text: str    # Full LLM text response
    action: str  # e.g. "move"
    location: str  # one of "park", "library", "home", "gym"


# =============================================================================
# FastAPI Endpoint
# =============================================================================

@app.post("/generate", response_model=GenerateResponse)
def generate_response(data: GenerateRequest):
    # Retrieve or create the conversation history for this agent
    conversation = get_or_create_session(data.agent_id, data.system_prompt)
    # Append the new user input to the conversation history
    conversation.append({"role": "user", "content": data.user_input})

    # Build a single prompt string from the conversation
    prompt = build_prompt(conversation)

    # Use our new OpenAIChatGPT interface (which uses the new API client)
    llm = OpenAIChatGPT(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    assistant_text = llm.generate(prompt)

    # Append the assistant's response to the conversation
    conversation.append({"role": "assistant", "content": assistant_text})

    # Parse the assistant response for a "MOVE: <location>" command
    action = "none"
    location = ""
    for line in assistant_text.splitlines():
        if line.strip().lower().startswith("move:"):
            action = "move"
            loc = line.split(":", 1)[1].strip().lower()
            if loc in ["park", "library", "home", "gym"]:
                location = loc
            break

    return GenerateResponse(
        agent_id=data.agent_id,
        text=assistant_text,
        action=action,
        location=location
    )

if __name__ == "__main__":
    # Run on localhost:3000
    uvicorn.run(app, host="127.0.0.1", port=3000)
