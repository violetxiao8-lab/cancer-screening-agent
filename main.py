from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

app = FastAPI(title="AgentT Cancer Screening API")

# ── Allow Lovable frontend to call this API ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Lovable URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once on startup ──────────────────────────
@app.on_event("startup")
async def startup_event():
    global vectorstore, llm
    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma(persist_directory="./knowledge_base", embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0.4)
    print("✅ Models loaded successfully!")

# ── Request & Response models ────────────────────────────
class ChatRequest(BaseModel):
    message: str
    age: Optional[int] = None
    gender: Optional[str] = None
    race: Optional[str] = "Chinese American"
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    reply: str
    status: str = "success"

# ── Health check ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AgentT is online! 🎓", "message": "Cancer Screening Educator API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ── Main chat endpoint ────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Retrieve relevant docs from knowledge base
        docs = vectorstore.similarity_search(request.message, k=6)
        context = "\n\n".join(d.page_content for d in docs)

        # Build personalized system prompt
        age    = request.age or "unknown"
        gender = request.gender or "unknown"
        race   = request.race or "Chinese American"

        system_prompt = f"""You are AgentT, a warm and knowledgeable cancer screening educator 
for {race} and broader Asian communities.

User profile: Age {age}, {gender}, {race}.

Instructions:
- Use ONLY the provided context to answer questions
- Be warm, conversational, encouraging, and easy to understand
- Tailor information to the user's age and gender when relevant
- Use bullet points for lists, keep responses clear and readable
- If answer is not in context say: "I don't have that specific info — please speak with your doctor."
- Do NOT add disclaimers at the end — shown separately on the page
- Respond naturally and conversationally as AgentT

Context from knowledge base:
{context}"""

        # Build conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add last 6 messages of history (3 turns)
        for msg in request.conversation_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current message
        messages.append({"role": "user", "content": request.message})

        # Get response from OpenAI
        response = llm.invoke(messages)
        
        return ChatResponse(reply=response.content, status="success")

    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again in a moment!",
            status="error"
        )