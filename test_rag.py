
# Quick test to verify RAG is working before building the app

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Load the knowledge base
print("Loading knowledge base...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

vectorstore = Chroma(
    persist_directory="./knowledge_base",
    embedding_function=embeddings
)

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3
)

print("✅ Knowledge base loaded!\n")

# Test function
def ask(question):
    print(f"❓ Question: {question}")
    
    # Search knowledge base
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    
    # Ask GPT with context
    prompt = f"""You are a friendly cancer screening educator helping Chinese-American and Asian communities.
Use ONLY the context below to answer. If the answer is not in the context, say so.
Always end with: "Remember, I'm an educator not a doctor — please consult a healthcare professional for personal advice."

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    print(f"💬 Answer: {response.content}\n")
    print("-" * 60)

# Run 3 test questions
ask("What is the most common cancer in Chinese American women?")
ask("At what age should I start getting screened for colorectal cancer?")
ask("What are the main risk factors for lung cancer in Asian Americans?")