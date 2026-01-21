from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(
    api_key="pcsk_brWKn_RXN6269HJFsiyn7c3Xoe2kZWA7cgoD5iCoamriEN2y78fRUP41rSsE7WbXchzDw"
)

index = pc.Index("jarvis-index")

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(q: Query):
    embedding = model.encode(q.question).tolist()

    result = index.query(
        vector=embedding,
        top_k=2,
        include_metadata=True
    )

    answer = ""
    for match in result["matches"]:
        answer += match["metadata"]["text"] + "\n"

    return {"answer": answer.strip()}

