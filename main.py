from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import List, Optional
import base64
import os
import uvicorn
import json
import psycopg2 
from dotenv import load_dotenv

# 1. Load the hidden .env file into memory first
load_dotenv()

# 2. Grab the URL from the .env file (or use dummy if it fails)
DATABASE_URL = os.getenv("DATABASE_URL", "dummy_local_url")

# 3. Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLOUD DATABASE SETUP ---
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS user_states
                     (user_id TEXT PRIMARY KEY, state_data TEXT)''')
        conn.commit()
        conn.close()
        print("Successfully connected to Supabase Cloud DB!")
    except Exception as e:
        print(f"Database connection error: {e}")

# Initialize the cloud table when the server starts
init_db()

class SyncRequest(BaseModel):
    user_id: str
    state_data: dict

# --- UPGRADED DATABASE ENDPOINTS ---
@app.post("/api/sync")
async def sync_state(request: SyncRequest):
    """Saves the user's entire session state to the Cloud Database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_states (user_id, state_data) 
        VALUES (%s, %s)
        ON CONFLICT (user_id) 
        DO UPDATE SET state_data = EXCLUDED.state_data
    """, (request.user_id, json.dumps(request.state_data)))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/load/{user_id}")
async def load_state(user_id: str):
    """Retrieves the user's session state from the Cloud."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT state_data FROM user_states WHERE user_id = %s", (user_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0])
    return {"activeSessionId": None, "sessions": {}}

# --- CLOUD API ENDPOINTS (GEMINI 24/7) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "missing_key")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = "gemini-2.5-flash"

class ChatRequest(BaseModel):
    message: str
    images_base64: List[str]
    history: list = []

class SummarizeRequest(BaseModel):
    caption: Optional[str] = None
    history: list = []

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/upload-and-caption")
async def generate_caption(files: List[UploadFile] = File(...)):
    content_list = [{"type": "text", "text": "Analyze these images as a single cohesive set and provide a detailed, combined summary."}]
    image_data_urls = []

    for file in files:
        image_bytes = await file.read()
        base64_img = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:{file.content_type};base64,{base64_img}"
        
        image_data_urls.append(image_data_url)
        content_list.append({"type": "image_url", "image_url": {"url": image_data_url}})
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content_list}]
    )
    
    return {"caption": response.choices[0].message.content, "images_base64": image_data_urls}

@app.post("/api/chat")
async def chat_with_image(request: ChatRequest):
    messages = request.history.copy()
    
    content_list = [{"type": "text", "text": request.message}]
    for img_b64 in request.images_base64:
        content_list.append({"type": "image_url", "image_url": {"url": img_b64}})

    messages.append({"role": "user", "content": content_list})
    
    response = await client.chat.completions.create(model=MODEL_NAME, messages=messages)
    return {"reply": response.choices[0].message.content}

@app.post("/api/summarize")
async def summarize_session(request: SummarizeRequest):
    prompt = f"Please provide a highly professional, one-paragraph executive summary of the following diagnostic session.\n\n"
    if request.caption:
        prompt += f"Initial Diagnostic Report:\n{request.caption}\n\n"
    
    prompt += "Follow-up Conversation:\n"
    for msg in request.history:
        # Ignore system notifications and old diagnostic headers in the final PDF
        if not msg['content'].startswith('*Diagnostics completed') and not msg['content'].startswith('*[System:'):
            prompt += f"{msg['role'].upper()}: {msg['content']}\n"

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"summary": response.choices[0].message.content}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)