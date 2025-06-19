from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
from pinecone import Pinecone

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legally-index")

app = FastAPI()

class EmbedRequest(BaseModel):
    user_id: str
    text: str
    metadata: dict = {}

@app.post("/embed")
def embed_and_store(data: EmbedRequest):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=data.text)
    vector = response['data'][0]['embedding']

    index.upsert([(data.user_id + "_" + str(hash(data.text)), vector, data.metadata)])
    return {"message": "embedded", "vector_dim": len(vector)}

@app.post("/search")
def search_similar(data: EmbedRequest):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=data.text)
    vector = response['data'][0]['embedding']

    results = index.query(vector=vector, top_k=5, include_metadata=True)
    return {"matches": results['matches']}
