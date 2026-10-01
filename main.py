from fastapi import FastAPI, Depends, HTTPException, Header







from fastapi.middleware.cors import CORSMiddleware







from pydantic import BaseModel







from typing import Optional







import os







import json







import datetime







import hmac

import uuid

import threading















from dotenv import load_dotenv







from langchain_openai import OpenAIEmbeddings, ChatOpenAI







from langchain_community.vectorstores import Chroma







import gspread







from google.oauth2.service_account import Credentials







import jwt















from analytics_core import compute_engagement_summary























load_dotenv()















app = FastAPI(title="AgentT Cancer Screening API")























# ---------------------------------------------------------------------------







# CORS







# ---------------------------------------------------------------------------















app.add_middleware(







    CORSMiddleware,







    allow_origins=["*"],







    allow_credentials=True,







    allow_methods=["*"],







    allow_headers=["*"],







)























# ---------------------------------------------------------------------------







# GLOBALS







# ---------------------------------------------------------------------------















vectorstore = None







llm = None







sheet = None























# ---------------------------------------------------------------------------







# AUTH CONFIG







# ---------------------------------------------------------------------------















ALLOWED_EMAILS = [







    e.strip().lower()







    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")







    if e.strip()







]















ADMIN_EMAILS = [







    e.strip().lower()







    for e in os.environ.get("ADMIN_EMAILS", "").split(",")







    if e.strip()







]















JWT_SECRET = os.environ.get(







    "JWT_SECRET",







    "change-this-secret-in-production"







)















ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")















TOKEN_EXPIRY_HOURS = 24























# ---------------------------------------------------------------------------







# JWT







# ---------------------------------------------------------------------------















def create_access_token(email: str, role: str = "user") -> str:







    payload = {







        "email": email.lower(),







        "role": role,







        "exp": datetime.datetime.utcnow()







        + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),







    }















    return jwt.encode(







        payload,







        JWT_SECRET,







        algorithm="HS256"







    )























def get_token_payload(







    authorization: Optional[str] = Header(None)







) -> dict:















    if not authorization or not authorization.startswith("Bearer "):







        raise HTTPException(







            status_code=401,







            detail="Missing or invalid Authorization header"







        )















    token = authorization.split(" ", 1)[1]















    try:







        payload = jwt.decode(







            token,







            JWT_SECRET,







            algorithms=["HS256"]







        )















        return payload















    except jwt.ExpiredSignatureError:







        raise HTTPException(







            status_code=401,







            detail="Token expired, please log in again"







        )















    except jwt.InvalidTokenError:







        raise HTTPException(







            status_code=401,







            detail="Invalid token"







        )























# ---------------------------------------------------------------------------







# USER AUTHORIZATION







# ---------------------------------------------------------------------------















def get_current_user(







    authorization: Optional[str] = Header(None)







) -> str:















    payload = get_token_payload(authorization)















    email = (payload.get("email") or "").lower()







    role = payload.get("role")















    if not email:







        raise HTTPException(







            status_code=401,







            detail="Not authorized"







        )















    if role not in ("user", "admin"):







        raise HTTPException(







            status_code=401,







            detail="Not authorized"







        )















    if email not in ALLOWED_EMAILS:







        raise HTTPException(







            status_code=401,







            detail="Not authorized"







        )















    return email























# ---------------------------------------------------------------------------







# ADMIN AUTHORIZATION







# ---------------------------------------------------------------------------















def get_current_admin(







    authorization: Optional[str] = Header(None)







) -> str:















    payload = get_token_payload(authorization)















    email = (payload.get("email") or "").lower()







    role = payload.get("role")















    if not email:







        raise HTTPException(







            status_code=403,







            detail="Admin access required"







        )















    if role != "admin":







        raise HTTPException(







            status_code=403,







            detail="Admin access required"







        )















    if email not in ADMIN_EMAILS:







        raise HTTPException(







            status_code=403,







            detail="Admin access required"







        )















    return email























# ---------------------------------------------------------------------------







# LOGIN REQUEST MODELS







# ---------------------------------------------------------------------------















class LoginBody(BaseModel):







    email: str























class AdminLoginBody(BaseModel):







    email: str







    password: str























# ---------------------------------------------------------------------------







# REGULAR USER LOGIN







# ---------------------------------------------------------------------------















@app.post("/auth/login")







def login(body: LoginBody):















    email = body.email.strip().lower()















    if not email:







        raise HTTPException(







            status_code=400,







            detail="Email is required"







        )















    if email not in ALLOWED_EMAILS:







        raise HTTPException(







            status_code=403,







            detail="This email is not authorized to access AgentT"







        )















    # Admin accounts must authenticate using the admin password.







    if email in ADMIN_EMAILS:







        return {







            "status": "password_required",







            "requires_password": True







        }















    token = create_access_token(







        email=email,







        role="user"







    )



    create_participant_session(email)















    return {







        "status": "success",







        "requires_password": False,







        "token": token,







        "role": "user"







    }























# ---------------------------------------------------------------------------







# ADMIN LOGIN







# ---------------------------------------------------------------------------















@app.post("/auth/admin-login")







def admin_login(body: AdminLoginBody):















    email = body.email.strip().lower()















    if email not in ADMIN_EMAILS:







        raise HTTPException(







            status_code=403,







            detail="Admin access required"







        )















    if not ADMIN_PASSWORD:







        raise HTTPException(







            status_code=500,







            detail="Admin authentication is not configured"







        )















    # Constant-time password comparison.







    if not hmac.compare_digest(







        body.password,







        ADMIN_PASSWORD







    ):







        raise HTTPException(







            status_code=401,







            detail="Incorrect password"







        )















    token = create_access_token(







        email=email,







        role="admin"







    )















    return {







        "status": "success",







        "token": token,







        "role": "admin"







    }























# ---------------------------------------------------------------------------







# GOOGLE SHEETS







# ---------------------------------------------------------------------------















SHEET_HEADERS = [







    "Session ID",







    "Participant Email",







    "Participant ID (Prolific)",







    "Timestamp",







    "Message Index",







    "Age",







    "Gender",







    "Ethnicity",







    "Family History",







    "Prior Screening",







    "Smoking",







    "Alcohol",







    "Community",







    "Question",







    "Answer",







]























# Tracks message count per session during the current server runtime.







session_message_counts = {}



























# Participant login/session tracking is intentionally kept OUT of Google Sheets.

# It exists only in the running API process and is exposed through /admin/sessions.

# A Railway restart/redeploy starts a fresh session list.

participant_sessions = {}

participant_sessions_lock = threading.Lock()

ACTIVE_SESSION_WINDOW_SECONDS = 15 * 60





def utc_now_iso():

    return datetime.datetime.now(datetime.timezone.utc).isoformat()





def parse_iso_datetime(value):

    if not value:

        return None

    try:

        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    except (TypeError, ValueError):

        return None





def create_participant_session(email: str):

    """Create an in-memory participant login session and return its ID."""

    normalized_email = (email or "").strip().lower()

    if not normalized_email or normalized_email in ADMIN_EMAILS:

        return None

    session_id = str(uuid.uuid4())

    now = utc_now_iso()

    with participant_sessions_lock:

        participant_sessions[session_id] = {

            "email": normalized_email, "session_id": session_id,

            "login_time": now, "last_activity": now, "message_count": 0,

        }

    return session_id





def update_participant_session(email: str):

    """Update the participant's most recent login session after a chat message."""

    normalized_email = (email or "").strip().lower()

    if not normalized_email or normalized_email in ADMIN_EMAILS:

        return

    now = utc_now_iso()

    with participant_sessions_lock:

        matching = [x for x in participant_sessions.values() if x.get("email") == normalized_email]

        if matching:

            session = max(matching, key=lambda x: x.get("login_time", ""))

        else:

            session_id = str(uuid.uuid4())

            session = {"email": normalized_email, "session_id": session_id, "login_time": now, "last_activity": now, "message_count": 0}

            participant_sessions[session_id] = session

        session["last_activity"] = now

        session["message_count"] = int(session.get("message_count", 0) or 0) + 1





def end_participant_session(email: str):
    """Mark the participant's most recent in-memory session as explicitly ended."""
    normalized_email = (email or "").strip().lower()
    if not normalized_email or normalized_email in ADMIN_EMAILS:
        return
    now = utc_now_iso()
    with participant_sessions_lock:
        matching = [x for x in participant_sessions.values() if x.get("email") == normalized_email]
        if not matching:
            return
        session = max(matching, key=lambda x: x.get("login_time", ""))
        session["last_activity"] = now
        session["ended_at"] = now


def init_google_sheets():















    global sheet















    try:















        creds_json = os.environ.get(







            "GOOGLE_CREDENTIALS_JSON"







        )















        if not creds_json:







            print(







                "⚠️ GOOGLE_CREDENTIALS_JSON not set, "







                "skipping Sheets integration"







            )







            return















        creds_dict = json.loads(creds_json)















        scopes = [







            "https://www.googleapis.com/auth/spreadsheets",







            "https://www.googleapis.com/auth/drive"







        ]















        creds = Credentials.from_service_account_info(







            creds_dict,







            scopes=scopes







        )















        client = gspread.authorize(creds)















        spreadsheet = client.open_by_key(







            "16AhEs5OlDGYl3eu36Ls1yEPFS8FkrtkH0EVPqbzpXts"







        )















        sheet = spreadsheet.sheet1















        existing_header = sheet.row_values(1)















        if existing_header != SHEET_HEADERS:















            # Only update the header row.







            # Existing participant data is preserved.







            sheet.update(







                "A1",







                [SHEET_HEADERS]







            )















        print(







            "✅ Google Sheets connected successfully!"







        )















    except Exception as e:















        import traceback















        print(







            f"⚠️ Google Sheets connection failed: {e}"







        )















        print(







            traceback.format_exc()







        )















        sheet = None























def log_to_sheets(







    request,







    reply,







    user_email







):















    global sheet















    if not sheet:







        return















    try:















        session_id = (







            request.session_id or "unknown"







        )















        session_message_counts[session_id] = (







            session_message_counts.get(







                session_id,







                0







            ) + 1







        )















        message_index = (







            session_message_counts[







                session_id







            ]







        )















        new_row = [







            session_id,







            user_email or "",







            request.participant_id or "",







            datetime.datetime.now().strftime(







                "%Y-%m-%d %H:%M:%S"







            ),







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















        sheet.append_row(







            new_row







        )















    except Exception as e:















        print(







            f"⚠️ Failed to log to Sheets: {e}"







        )























# ---------------------------------------------------------------------------







# STARTUP







# ---------------------------------------------------------------------------















@app.on_event("startup")







async def startup_event():















    global vectorstore, llm















    api_key = os.environ.get(







        "OPENAI_API_KEY"







    )















    llm = ChatOpenAI(







        model="gpt-4o",







        openai_api_key=api_key,







        temperature=0.4







    )















    try:















        embeddings = OpenAIEmbeddings(







            model="text-embedding-3-small",







            openai_api_key=api_key







        )















        vectorstore = Chroma(







            persist_directory="./knowledge_base",







            embedding_function=embeddings







        )















        print(







            "✅ Models and knowledge base loaded successfully!"







        )















    except Exception as e:















        print(







            "⚠️ ChromaDB not found, "







            f"will answer without context: {e}"







        )















        vectorstore = None















    init_google_sheets()























# ---------------------------------------------------------------------------







# CHAT MODELS







# ---------------------------------------------------------------------------















class ChatRequest(BaseModel):















    message: str















    session_id: Optional[str] = None















    participant_id: Optional[str] = None















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























# ---------------------------------------------------------------------------







# HEALTH ENDPOINTS







# ---------------------------------------------------------------------------















@app.get("/")







def root():















    return {







        "status": "AgentT is online! 🎓",







        "message": "Cancer Screening Educator API"







    }























@app.get("/health")







def health():















    return {







        "status": "healthy"







    }























# ---------------------------------------------------------------------------







# CHAT







# ---------------------------------------------------------------------------















@app.post(







    "/chat",







    response_model=ChatResponse







)







async def chat(







    request: ChatRequest,







    current_user: str = Depends(







        get_current_user







    )







):















    try:















        context = ""















        if vectorstore:















            docs = vectorstore.similarity_search(







                request.message,







                k=6







            )















            context = "\n\n".join(







                d.page_content







                for d in docs







            )















        age = (







            request.age or "unknown"







        )















        gender = (







            request.gender or "unknown"







        )















        ethnicity = (







            request.ethnicity







            or "Chinese American"







        )















        system_prompt = f"""







You are AgentT, a warm and knowledgeable cancer screening educator







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







{context}







"""















        messages = [







            {







                "role": "system",







                "content": system_prompt







            }







        ]















        for msg in (







            request.conversation_history[-6:]







        ):















            if msg.get("role") in [







                "user",







                "assistant"







            ]:















                messages.append(







                    {







                        "role": msg["role"],







                        "content": msg["content"]







                    }







                )















        messages.append(







            {







                "role": "user",







                "content": request.message







            }







        )















        response = llm.invoke(







            messages







        )















        reply = (







            response.content







            if hasattr(







                response,







                "content"







            )







            else str(response)







        )















        # Authenticated email is recorded







        # with every interaction.







        log_to_sheets(







            request,







            reply,







            current_user







        )



        update_participant_session(current_user)















        return ChatResponse(







            reply=reply,







            status="success"







        )















    except Exception as e:















        print(







            f"Error: {e}"







        )















        return ChatResponse(







            reply=(







                "I'm having trouble connecting "







                "right now. Please try again "







                "in a moment!"







            ),







            status="error"







        )























# ---------------------------------------------------------------------------







# ADMIN ANALYTICS







# ---------------------------------------------------------------------------















@app.get("/admin/analytics")







def admin_analytics(







    current_admin: str = Depends(







        get_current_admin







    )







):















    if not sheet:















        raise HTTPException(







            status_code=503,







            detail="Sheet not connected"







        )















    try:















        rows = sheet.get_all_records()















        summary = compute_engagement_summary(







            rows,







            ALLOWED_EMAILS







        )















        return summary















    except Exception as e:















        raise HTTPException(







            status_code=500,







            detail=(







                "Failed to compute analytics: "







                f"{e}"







            )







        )







# ---------------------------------------------------------------------------



# ADMIN PARTICIPANT SESSIONS



# ---------------------------------------------------------------------------







@app.get("/admin/sessions")

def admin_sessions(current_admin: str = Depends(get_current_admin)):

    now = datetime.datetime.now(datetime.timezone.utc)

    sessions = []

    with participant_sessions_lock:

        records = [dict(item) for item in participant_sessions.values()]

    for record in records:

        login_time = str(record.get("login_time", "")).strip()

        last_activity = str(record.get("last_activity", "")).strip()

        login_dt = parse_iso_datetime(login_time)

        activity_dt = parse_iso_datetime(last_activity)

        duration_seconds = max(0, int((activity_dt - login_dt).total_seconds())) if login_dt and activity_dt else 0

        status = "Ended"
        explicitly_ended = bool(record.get("ended_at"))
        if activity_dt and not explicitly_ended:
            if activity_dt.tzinfo is None:
                activity_dt = activity_dt.replace(tzinfo=datetime.timezone.utc)
            if max(0, (now - activity_dt).total_seconds()) <= ACTIVE_SESSION_WINDOW_SECONDS:
                status = "Active"

        sessions.append({

            "email": str(record.get("email", "")).strip().lower(),

            "session_id": str(record.get("session_id", "")).strip(),

            "login_time": login_time, "last_activity": last_activity,

            "duration_seconds": duration_seconds,

            "message_count": int(record.get("message_count", 0) or 0),

            "status": status,

        })

    sessions.sort(key=lambda item: item.get("login_time", ""), reverse=True)

    return {"sessions": sessions}


