from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone

# === Инициализация ===
app = FastAPI()
load_dotenv()

# 🔑 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔗 Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# 🔧 Получение эмбеддинга через OpenAI
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 📥 Модель запроса
class QueryRequest(BaseModel):
    query: str
    user_id: str

# 🚀 Эндпоинт поиска
@app.post("/search")
async def search_docs(req: QueryRequest):
    try:
        query_embedding = get_embedding(req.query)

        # 🔍 Поиск в Pinecone с фильтрацией по user_id
        search_response = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            # filter={"user_id": {"$eq": req.user_id}}
        )

        matches = search_response.get('matches', [])
        if not matches:
            return {"matches": None, "answer": "Ничего не найдено по вашему запросу."}

        # 📚 Сбор контекста из метаданных
        retrieved_context = "\n".join([m['metadata'].get('text', '') for m in matches])

        # 🧠 GPT-4 запрос
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ты — AI-юрист, помогающий анализировать документы."},
                {"role": "user", "content": f"Вот контекст:\n{retrieved_context}\n\nВопрос: {req.query}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return {
            "matches": jsonable_encoder(matches),
            "answer": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
