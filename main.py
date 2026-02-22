from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()

app = FastAPI(title="AgentT Cancer Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None
llm = None
sheet = None

def init_google_sheets():
    global sheet
    try:
        # Load credentials from environment variable (JSON string)
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not creds_json:
            print("⚠️ GOOGLE_CREDENTIALS_JSON not set, skipping Sheets integration")
            return

        creds_dict = json.loads(creds_json)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        spreadsheet = client.open("AgentT User Data")
        sheet = spreadsheet.sheet1

        # Add headers if sheet is empty
        if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
            headers = [
                "Timestamp", "Age", "Gender", "Ethnicity",
                "Family History", "Prior Screening", "Smoking",
                "Alcohol", "Activity", "Community",
                "User Question", "AgentT Answer"
            ]
            sheet.append_row(headers)

        print("✅ Google Sheets connected successfully!")
    except Exception as e:
        print(f"⚠️ Google Sheets connection failed: {e}")
        sheet = None

def log_to_sheets(request, reply):
    global sheet
    if not sheet:
        return
    try:
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            request.age or "",
            request.gender or "",
            request.ethnicity or "",
            request.family_history or "",
            request.prior_screening or "",
            request.smoking or "",
            request.alcohol or "",
            request.activity or "",
            request.community or "",
            request.message,
            reply
        ]
        sheet.append_row(row)
    except Exception as e:
        print(f"⚠️ Failed to log to Sheets: {e}")

@app.on_event("startup")
async def startup_event():
    global vectorstore, llm
    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0.4)
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        vectorstore = Chroma(persist_directory="./knowledge_base", embedding_function=embeddings)
        print("✅ Models and knowledge base loaded successfully!")
    except Exception as e:
        print(f"⚠️ ChromaDB not found, will answer without context: {e}")
        vectorstore = None

    init_google_sheets()

class ChatRequest(BaseModel):
    message: str
    age: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = "Chinese American"
    family_history: Optional[str] = None
    prior_screening: Optional[str] = None
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    activity: Optional[str] = None
    community: Optional[str] = None
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    reply: str
    status: str = "success"

@app.get("/")
def root():
    return {"status": "AgentT is online! 🎓", "message": "Cancer Screening Educator API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        context = ""
        if vectorstore:
            docs = vectorstore.similarity_search(request.message, k=6)
            context = "\n\n".join(d.page_content for d in docs)

        age = request.age or "unknown"
        gender = request.gender or "unknown"
        ethnicity = request.ethnicity or "Chinese American"

        system_prompt = f"""You are AgentT, a warm and knowledgeable cancer screening educator
for {ethnicity} and broader Asian communities.

User profile: Age {age}, {gender}, {ethnicity}.

Instructions:
- Be warm, conversational, encouraging, and easy to understand
- Tailor information to the user's age and gender when relevant
- Use bullet points for lists, keep responses clear and readable
- If answer is not in context say: "I don't have that specific info — please speak with your doctor."
- Do NOT add disclaimers at the end
- Respond naturally and conversationally as AgentT

Context from knowledge base:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]

        for msg in request.conversation_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": request.message})

        response = llm.invoke(messages)
        reply = response.content if hasattr(response, 'content') else str(response)

        # Log to Google Sheets
        log_to_sheets(request, reply)

        return ChatResponse(reply=reply, status="success")

    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again in a moment!",
            status="error"
        )