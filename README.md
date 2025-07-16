# Movie Recommender Chatbot (LLM-powered)

This is a FastAPI-based chatbot powered by Hugging Face’s hosted LLM (`google/gemma-2-2b-it`) via the Inference API using `provider="nebius"`. The bot provides personalized movie recommendations based on user queries and context.

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | FastAPI app with Hugging Face LLM integration |
| `sample_chats.txt` | 3 multi-turn movie conversations |
| `train_sample.jsonl` | Fine-tuning-ready format (JSONL) |
| `requirements.txt` | All dependencies needed |

---

## Running the App

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Create a `.env` file
```
HF_TOKEN=hf_Your_Hugging_Face_Token
```

### Step 3: Start the server
```bash
uvicorn main:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test via Swagger UI.

---

## Features

- Multi-turn chat support with `session_id`
- Natural language movie recommendations
- Hugging Face hosted inference (no large models locally)
- Minimal API with `/chat`, `/sample_chats`, `/training_sample`

---

## Assignment Requirements Checklist

- ✅ 3 realistic multi-turn dialogues  
- ✅ One chat converted to fine-tuning dataset (JSONL)  
- ✅ Inference using Hugging Face  
- ✅ FastAPI wrapper with POST `/chat`  
- ✅ Session tracking supported  

---

## Requirements

```txt
fastapi
uvicorn
python-dotenv
huggingface_hub
pydantic
```

