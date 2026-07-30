from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import datetime
import random
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import gspread
from google.oauth2.service_account import Credentials
import jwt

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

# ---------------------------------------------------------------------------
# OTP AUTH CONFIG
# ---------------------------------------------------------------------------

ALLOWED_EMAILS = [
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
    if e.strip()
]

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-secret-in-production")
OTP_EXPIRY_MINUTES = 10
TOKEN_EXPIRY_HOURS = 24

# In-memory OTP store: { email: {"code": "123456", "expires_at": datetime} }
# NOTE: this resets if the server restarts, and won't work across multiple
# replicas. Fine for a demo / single-instance deploy.
otp_store = {}


def send_otp_email(to_email: str, code: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise RuntimeError("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not configured")

    subject = "Your AgentT login code"
    body = f"""Hi,

Your 5-digit one-time login code for AgentT is:

    {code}

This code expires in {OTP_EXPIRY_MINUTES} minutes. If you didn't request this, you can ignore this email.
"""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, [to_email], msg.as_string())


def create_access_token(email: str) -> str:
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    email = payload.get("email")
    if not email or email.lower() not in ALLOWED_EMAILS:
        raise HTTPException(status_code=401, detail="Not authorized")

    return email


class RequestOtpBody(BaseModel):
    email: str


class VerifyOtpBody(BaseModel):
    email: str
    code: str


@app.post("/auth/request-otp")
def request_otp(body: RequestOtpBody):
    email = body.email.strip().lower()

    if email not in ALLOWED_EMAILS:
        raise HTTPException(status_code=403, detail="This email is not authorized to access AgentT")

    code = f"{random.randint(0, 99999):05d}"
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=OTP_EXPIRY_MINUTES)
    otp_store[email] = {"code": code, "expires_at": expires_at}

    try:
        send_otp_email(email, code)
    except Exception as e:
        print(f"⚠️ Failed to send OTP email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send OTP email")

    return {"status": "sent", "message": f"A login code was sent to {email}"}


@app.post("/auth/verify-otp")
def verify_otp(body: VerifyOtpBody):
    email = body.email.strip().lower()
    entry = otp_store.get(email)

    if not entry:
        raise HTTPException(status_code=400, detail="No OTP requested for this email")

    if datetime.datetime.utcnow() > entry["expires_at"]:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="Code expired, please request a new one")

    if body.code.strip() != entry["code"]:
        raise HTTPException(status_code=400, detail="Incorrect code")

    del otp_store[email]
    token = create_access_token(email)
    return {"status": "success", "token": token}


# ---------------------------------------------------------------------------
# EXISTING APP LOGIC
# ---------------------------------------------------------------------------

def init_google_sheets():
    global sheet
    try:
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

        spreadsheet = client.open_by_key("16AhEs5OlDGYl3eu36Ls1yEPFS8FkrtkH0EVPqbzpXts")
        sheet = spreadsheet.sheet1

        if not sheet.cell(1, 1).value:
            headers = [
                "Session ID", "Timestamp", "Age", "Gender", "Ethnicity",
                "Family History", "Prior Screening", "Smoking",
                "Alcohol", "Community",
                "Q1", "A1"
            ]
            sheet.append_row(headers)

        print("✅ Google Sheets connected successfully!")
    except Exception as e:
        import traceback
        print(f"⚠️ Google Sheets connection failed: {e}")
        print(traceback.format_exc())
        sheet = None

def log_to_sheets(request, reply):
    global sheet
    if not sheet:
        return
    try:
        session_id = request.session_id or "unknown"
        all_values = sheet.get_all_values()

        row_index = None
        for i, row in enumerate(all_values[1:], start=2):
            if row and row[0] == session_id:
                row_index = i
                break

        if row_index is None:
            new_row = [
                session_id,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                request.age or "",
                request.gender or "",
                request.ethnicity or "",
                request.family_history or "",
                request.prior_screening or "",
                request.smoking or "",
                request.alcohol or "",
                request.community or "",
                request.message,
                reply
            ]
            sheet.append_row(new_row)
        else:
            existing_row = all_values[row_index - 1]
            next_col = len(existing_row) + 1
            sheet.update_cell(row_index, next_col, request.message)
            sheet.update_cell(row_index, next_col + 1, reply)

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
    session_id: Optional[str] = None
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
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
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

        log_to_sheets(request, reply)

        return ChatResponse(reply=reply, status="success")

    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again in a moment!",
            status="error"
        )
