from fastapi import FastAPI, Query
import pandas as pd
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

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
    query_embedding = model.encode(query).reshape(1, -1)

    # Search in FAISS
    D, I = index.search(np.float32(query_embedding), k=5)
    retrieved_texts = df.iloc[I[0]]["sentence_chunk"].tolist()
    retrieved_embeddings = np.stack(df.iloc[I[0]]["embedding"].values)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, retrieved_embeddings)[0]
    max_similarity = max(similarities) if similarities.size > 0 else 0

    # Check similarity threshold
    threshold = 0.4  # Adjust based on experimentation
    if max_similarity < threshold:
        return {
            "query": query,
            "retrieved_texts": retrieved_texts,
            "generated_response": "Sorry, I don't know how to answer that based on the available information.",
            "session_id": session_id
        }

    # Prepare context and call OpenAI if relevant context is found
    context = "\n".join(retrieved_texts)

    # Maintain conversation history
    if session_id:
        if session_id not in session_histories:
            session_histories[session_id] = []
        session_histories[session_id].append({"role": "user", "content": query})

        conversation_history = session_histories[session_id][-5:]
    else:
        conversation_history = []

    messages = [
        {"role": "system", "content": "You are an assistant that answers based only on the provided context. If the context is irrelevant, say you don't know."},
        *conversation_history,
        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    ai_response = response.choices[0].message.content

    if session_id:
        session_histories[session_id].append({"role": "assistant", "content": ai_response})

    return {
        "query": query,
        "retrieved_texts": retrieved_texts,
        "similarity_scores": similarities.tolist(),
        "generated_response": ai_response,
        "session_id": session_id
    }

