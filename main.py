from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI  # ✅ новый импорт клиента
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()
client = OpenAI()  # ✅ создание клиента

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_docs(req: QueryRequest):
    query_embedding = model.encode(req.query, convert_to_tensor=True)

    # Здесь ищем похожие документы (пример — заглушка)
    retrieved_context = "документ 1\nдокумент 2"


    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Ты — AI-юрист, помогающий анализировать документы."},
            {"role": "user", "content": f"Вот контекст:\n{retrieved_context}\n\nВопрос: {req.query}"}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return {"answer": response.choices[0].message.content}
