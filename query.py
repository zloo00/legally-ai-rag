import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# === Шаг 1: Загрузка переменных среды ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "legally-index"

# === Шаг 2: Инициализация клиентов ===
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === Шаг 3: Функция получения эмбеддингов запроса ===
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = openai.embeddings.create(input=[text], model=model)
    return res.data[0].embedding

# === Шаг 4: Поиск по Pinecone ===
def query_pinecone(query_text, top_k=5):
    print(f"\n🔍 Поиск по запросу: {query_text}")
    query_vector = get_embedding(query_text)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print("\n📄 Результаты:")
    for i, match in enumerate(results['matches'], start=1):
        print(f"\n#{i}:")
        print(f"Score: {match['score']:.4f}")
        print(f"Filename: {match['metadata'].get('filename', 'N/A')}")

# === Шаг 5: Запуск запроса ===
if __name__ == "__main__":
    user_query = input("Введите юридический запрос: ")
    query_pinecone(user_query)
