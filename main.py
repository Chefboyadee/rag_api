from fastapi import FastAPI, Query
import pandas as pd
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# Load the data
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "cleaned_neo_text_chunks_and_embeddings_df.csv")

df = pd.read_csv(csv_path)
df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))

# Build FAISS index
d = len(df["embedding"].iloc[0])
index = faiss.IndexFlatL2(d)
index.add(np.stack(df["embedding"].values))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI API Key (expects it to be set as an env variable or manually passed)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this for security in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history
session_histories = {}

@app.get("/query")
def query_rag(
    query: str = Query(..., description="Your search query"),
    session_id: str = Query(None, description="Session ID for context persistence")
):
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Search in FAISS
    D, I = index.search(query_embedding, k=5)
    retrieved_texts = df.iloc[I[0]]["sentence_chunk"].tolist()
    context = "\n".join(retrieved_texts)

    # Maintain conversation history
    if session_id:
        if session_id not in session_histories:
            session_histories[session_id] = []
        session_histories[session_id].append({"role": "user", "content": query})

        # Add past messages to context
        conversation_history = session_histories[session_id][-5:]  # Limit to last 5 exchanges
    else:
        conversation_history = []

    # Construct messages for OpenAI
    messages = [{"role": "system", "content": "You are an assistant that provides answers based on retrieved documents."}]
    messages.extend(conversation_history)  # Include past exchanges
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuery: {query}"})

    # Generate response
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    ai_response = response.choices[0].message.content

    # Store AI response in history
    if session_id:
        session_histories[session_id].append({"role": "assistant", "content": ai_response})

    return {
        "query": query,
        "retrieved_texts": retrieved_texts,
        "generated_response": ai_response,
        "session_id": session_id
    }
