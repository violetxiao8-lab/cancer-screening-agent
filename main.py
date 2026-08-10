from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import datetime
import random
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import gspread
from google.oauth2.service_account import Credentials
import jwt

from analytics_core import compute_engagement_summary

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

# Admins must ALSO be in ALLOWED_EMAILS to log in at all (same OTP flow).
# ADMIN_EMAILS is just an extra permission flag checked on top of that.
ADMIN_EMAILS = [
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
]

SENDGRID_API_KEY = (os.environ.get("SENDGRID_API_KEY") or "").strip()
SENDGRID_FROM_EMAIL = (os.environ.get("SENDGRID_FROM_EMAIL") or "mashiatjamal91@gmail.com").strip()
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-secret-in-production")
OTP_EXPIRY_MINUTES = 10
TOKEN_EXPIRY_HOURS = 24

# In-memory OTP store: { email: {"code": "123456", "expires_at": datetime} }
# NOTE: this resets if the server restarts, and won't work across multiple
# replicas. Fine for a demo / single-instance deploy.
otp_store = {}


def send_otp_email(to_email: str, code: str):
    if not SENDGRID_API_KEY:
        raise RuntimeError("SENDGRID_API_KEY not configured")

    body_html = f"""
    <p>Hi,</p>
    <p>Your 5-digit one-time login code for AgentT is:</p>
    <h2>{code}</h2>
    <p>This code expires in {OTP_EXPIRY_MINUTES} minutes. If you didn't request this, you can ignore this email.</p>
    """

    response = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": SENDGRID_FROM_EMAIL, "name": "AgentT"},
            "subject": "Your AgentT login code",
            "content": [{"type": "text/html", "value": body_html}],
        },
        timeout=15,
    )

    if response.status_code >= 300:
        raise RuntimeError(f"SendGrid API error {response.status_code}: {response.text}")


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


def get_current_admin(current_user: str = Depends(get_current_user)) -> str:
    if current_user.lower() not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


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

SHEET_HEADERS = [
    "Session ID", "Participant Email", "Participant ID (Prolific)", "Timestamp",
    "Message Index", "Age", "Gender", "Ethnicity", "Family History",
    "Prior Screening", "Smoking", "Alcohol", "Community", "Question", "Answer",
]

# Tracks how many messages we've seen per session_id during this server's
# runtime, so "Message Index" is meaningful without re-reading the sheet.
# NOTE: like otp_store, this resets on restart. If you need Message Index to
# survive restarts, derive it from the sheet later during analysis instead
# of relying on this counter.
session_message_counts = {}


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

        existing_header = sheet.row_values(1)
        if existing_header != SHEET_HEADERS:
            # Only overwrite the header row itself — never touch existing data rows.
            sheet.update("A1", [SHEET_HEADERS])

        print("✅ Google Sheets connected successfully!")
    except Exception as e:
        import traceback
        print(f"⚠️ Google Sheets connection failed: {e}")
        print(traceback.format_exc())
        sheet = None


def log_to_sheets(request, reply, user_email):
    """Append ONE row per message. No reads, no cell-by-cell updates —
    just a fast, append-only write so the sheet stays a clean flat log
    that's trivial to pivot/filter for engagement analysis."""
    global sheet
    if not sheet:
        return
    try:
        session_id = request.session_id or "unknown"
        session_message_counts[session_id] = session_message_counts.get(session_id, 0) + 1
        message_index = session_message_counts[session_id]

        new_row = [
            session_id,
            user_email or "",
            request.participant_id or "",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message_index,
            request.age or "",
            request.gender or "",
            request.ethnicity or "",
            request.family_history or "",
            request.prior_screening or "",
            request.smoking or "",
            request.alcohol or "",
            request.community or "",
            request.message,
            reply,
        ]
        sheet.append_row(new_row)

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
    participant_id: Optional[str] = None  # e.g. Prolific ID, if collected client-side
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

        log_to_sheets(request, reply, current_user)

        return ChatResponse(reply=reply, status="success")

    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again in a moment!",
            status="error"
        )


@app.get("/admin/analytics")
def admin_analytics(current_admin: str = Depends(get_current_admin)):
    """Admin-only: returns the same engagement summary as analytics_summary.py,
    as JSON, so an admin panel button can call this directly instead of
    someone having to run the script by hand."""
    if not sheet:
        raise HTTPException(status_code=503, detail="Sheet not connected")

    try:
        rows = sheet.get_all_records()
        summary = compute_engagement_summary(rows, ALLOWED_EMAILS)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute analytics: {e}")
