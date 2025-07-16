import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --- Load environment variables ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Initialize Hugging Face Inference Client ---
client = InferenceClient(
    provider="nebius",
    api_key=HF_TOKEN
)

# --- FastAPI App ---
app = FastAPI(
    title="Movie Recommender Chatbot",
    description="An LLM-powered conversational agent for movie recommendations",
    version="1.0"
)

# --- Data Models ---
class ChatRequest(BaseModel):
    user_message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    bot_reply: str
    session_id: str

# --- Session-Based Conversation History ---
conversation_history: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return conversation_history.get(session_id, [])

def add_to_history(session_id: str, role: str, content: str):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"role": role, "content": content})

# --- LLM Inference Function ---
def get_bot_reply(user_message: str, session_id: Optional[str] = None) -> Tuple[str, str]:
    system_prompt = """You are a friendly and knowledgeable movie recommendation assistant.
Your responses should be:
- Conversational and engaging
- Focused on movie suggestions
- Limited to 2-3 sentences
- Ask clarifying questions when needed
- Never recommend inappropriate content"""

    # Create new session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Build message history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": user_message})

    try:
        # Chat completion
        response = client.chat.completions.create(
            model="google/gemma-2-2b-it",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        reply = response.choices[0].message.content.strip()

        # Save to history
        add_to_history(session_id, "user", user_message)
        add_to_history(session_id, "assistant", reply)

        return reply, session_id

    except Exception as e:
        print(f"Error: {e}")
        return "I'm having trouble connecting to my movie database. Please try again later.", session_id

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    reply, session_id = get_bot_reply(request.user_message, request.session_id)
    return ChatResponse(bot_reply=reply, session_id=session_id)

@app.get("/sample_chats")
async def get_sample_chats():
    return {
        "conversation_1": [
            {"role": "user", "content": "Can you recommend a good comedy movie?"},
            {"role": "assistant", "content": "Sure! You might enjoy 'The Grand Budapest Hotel' — it's quirky and hilarious."},
            {"role": "user", "content": "Is it like any other movies?"},
            {"role": "assistant", "content": "Yes, if you like Wes Anderson films, try 'Moonrise Kingdom' or 'Fantastic Mr. Fox'."}
        ]
    }

@app.get("/training_sample")
async def get_training_sample():
    return {
        "format": "JSONL",
        "samples": [
            {
                "input": "Can you recommend a good comedy movie?",
                "output": "Sure! You might enjoy 'The Grand Budapest Hotel' — it's quirky and hilarious."
            },
            {
                "input": "Is it like any other movies?",
                "output": "Yes, if you like Wes Anderson films, try 'Moonrise Kingdom' or 'Fantastic Mr. Fox'."
            }
        ]
    }

# --- Run app (for manual run) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
