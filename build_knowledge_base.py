# Run this ONCE to build your ChromaDB vector store

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR       = "./data"
VIDEOS_DIR     = "./data/videos"
VECTOR_DB_PATH = "./knowledge_base"

VIDEOS = {
    "epidemiology":          "j2M0tZz6LI8",
    "colorectal_screening":  "Ltz4Tb1I7kQ",
    "lung_cancer_disparity": "LHjnKMx3OTY",
}

os.makedirs(VIDEOS_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD PDFs
# ============================================================
print("\n" + "="*60)
print("STEP 1: Loading PDF documents")
print("="*60)

all_documents = []

slides_path = os.path.join(DATA_DIR, "slides.pdf")
if os.path.exists(slides_path):
    loader = PyPDFLoader(slides_path)
    slides_docs = loader.load()
    for doc in slides_docs:
        doc.metadata["source_type"] = "presentation_slides"
        doc.metadata["source_name"] = "Cancer Screening for Chinese Americans (Slides)"
    all_documents.extend(slides_docs)
    print(f"  ✅ Slides PDF: {len(slides_docs)} pages loaded")
else:
    print(f"  ❌ slides.pdf not found in data/ folder")

acs_path = os.path.join(DATA_DIR, "aanhpi_cff.pdf")
if os.path.exists(acs_path):
    loader = PyPDFLoader(acs_path)
    acs_docs = loader.load()
    for doc in acs_docs:
        doc.metadata["source_type"] = "research_document"
        doc.metadata["source_name"] = "ACS AANHPI Cancer Facts and Figures"
    all_documents.extend(acs_docs)
    print(f"  ✅ ACS PDF: {len(acs_docs)} pages loaded")
else:
    print(f"  ❌ aanhpi_cff.pdf not found in data/ folder")

# ============================================================
# STEP 2: FETCH YOUTUBE TRANSCRIPTS
# ============================================================
print("\n" + "="*60)
print("STEP 2: Fetching YouTube transcripts")
print("="*60)

ytt_api = YouTubeTranscriptApi()

for topic, video_id in VIDEOS.items():
    saved_path = os.path.join(VIDEOS_DIR, f"{topic}.txt")

    if os.path.exists(saved_path):
        print(f"  📂 {topic}: Loading from saved file...")
        with open(saved_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        print(f"      ✅ Loaded ({len(transcript_text):,} chars)")
    else:
        print(f"  🌐 {topic} ({video_id}): Fetching from YouTube...")
        try:
            fetched = ytt_api.fetch(video_id, languages=("en", "en-US"))
            snippets = [s.text.strip() for s in fetched if s.text.strip()]
            transcript_text = " ".join(snippets)

            if not transcript_text:
                raise ValueError("Empty transcript")

            with open(saved_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"      ✅ Fetched & saved ({len(transcript_text):,} chars)")

        except (NoTranscriptFound, TranscriptsDisabled) as e:
            print(f"      ❌ No transcript available: {e}")
            transcript_text = None
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            transcript_text = None

    if transcript_text:
        doc = Document(
            page_content=transcript_text,
            metadata={
                "source_type": "video_transcript",
                "source_name": f"YouTube: {topic.replace('_', ' ').title()}",
                "video_id": video_id,
                "topic": topic
            }
        )
        all_documents.append(doc)

# ============================================================
# STEP 3: CHUNK DOCUMENTS
# ============================================================
print("\n" + "="*60)
print("STEP 3: Splitting into chunks")
print("="*60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

chunks = splitter.split_documents(all_documents)
print(f"  ✅ Created {len(chunks)} chunks from {len(all_documents)} documents")

source_counts = Counter(c.metadata.get("source_type", "unknown") for c in chunks)
for src, count in source_counts.items():
    print(f"     {src}: {count} chunks")

# ============================================================
# STEP 4: EMBED + STORE IN CHROMADB
# ============================================================
print("\n" + "="*60)
print("STEP 4: Creating embeddings and building ChromaDB")
print("="*60)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("  ❌ OPENAI_API_KEY not found in .env file!")
    print("     Add your key to .env and run again.")
    exit(1)

print("  ⏳ Embedding chunks with OpenAI text-embedding-3-small...")
print("     This takes 1-2 minutes...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_PATH
)

print(f"  ✅ ChromaDB saved to: {VECTOR_DB_PATH}")

# ============================================================
# STEP 5: TEST RETRIEVAL
# ============================================================
print("\n" + "="*60)
print("STEP 5: Testing retrieval")
print("="*60)

test_queries = [
    "What is the most common cancer in Chinese American women?",
    "Why does early cancer screening matter?",
    "What are colorectal cancer screening guidelines?",
    "How does lung cancer affect Asian Americans?",
]

for query in test_queries:
    results = vectorstore.similarity_search(query, k=2)
    print(f"\n  ❓ {query}")
    print(f"  📄 Source: {results[0].metadata.get('source_name', 'unknown')}")
    print(f"  💬 Preview: {results[0].page_content[:120]}...")

print("\n" + "="*60)
print("✅ KNOWLEDGE BASE READY!")
print("="*60)