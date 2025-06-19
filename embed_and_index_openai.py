import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# === Шаг 1: Загрузка ключей ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "legally-index"

# === Шаг 2: Настройка клиентов ===
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# === Шаг 3: Создание индекса, если не существует ===
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    print(f"📎 Индекс '{INDEX_NAME}' не найден. Создаём новый...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )
else:
    print(f"✅ Индекс '{INDEX_NAME}' уже существует.")

index = pc.Index(INDEX_NAME)

# === Шаг 4: Загрузка текстов чанков ===
chunk_dir = "data/chunks"
texts = []
metadatas = []

for filename in os.listdir(chunk_dir):
    path = os.path.join(chunk_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if len(text) > 10:
            texts.append(text)
            metadatas.append({"filename": filename})

# === Шаг 5: Функция получения эмбеддингов ===
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = openai.embeddings.create(input=[text], model=model)
    return res.data[0].embedding


# === Шаг 6: Векторизация и загрузка в Pinecone ===
batch_size = 100
for i in tqdm(range(0, len(texts), batch_size), desc="📦 Индексация в Pinecone"):
    batch_texts = texts[i:i + batch_size]
    batch_ids = [f"doc-{i + j}" for j in range(len(batch_texts))]
    batch_embeddings = [get_embedding(text) for text in batch_texts]

    index.upsert(vectors=[
        {
            "id": batch_ids[j],
            "values": batch_embeddings[j],
            "metadata": metadatas[i + j]
        } for j in range(len(batch_texts))
    ])

print("✅ Индексация завершена.")
