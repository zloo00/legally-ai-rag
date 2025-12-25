import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === Шаг 1: Загрузка ключей ===
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "legally-index"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME") or "BAAI/bge-m3"
EMBEDDING_PASSAGE_PROMPT = os.getenv("EMBEDDING_PASSAGE_PROMPT") or "Represent this passage for retrieval: "

print(f"🔧 Configuration:")
print(f"   INDEX_NAME: {INDEX_NAME}")
print(f"   PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"   EMBEDDING_MODEL: {EMBEDDING_MODEL_NAME}")

# === Шаг 2: Настройка клиентов ===
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize sentence transformer for multilingual legal embeddings (ru/kz friendly)
sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = sentence_model.get_sentence_embedding_dimension()

# === Шаг 3: Создание индекса, если не существует ===
try:
    existing_indexes = [index.name for index in pc.list_indexes()]
    print(f"📋 Existing indexes: {existing_indexes}")
    
    if INDEX_NAME not in existing_indexes:
        print(f"📎 Индекс '{INDEX_NAME}' не найден. Создаём новый...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,  # bge-m3 dense dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
        print(f"✅ Индекс '{INDEX_NAME}' создан.")
    else:
        print(f"✅ Индекс '{INDEX_NAME}' уже существует.")
        
    index = pc.Index(INDEX_NAME)
    print(f"✅ Подключение к индексу установлено.")
    
except Exception as e:
    print(f"❌ Error with Pinecone index: {e}")
    exit(1)

# === Шаг 4: Загрузка текстов чанков с метаданными ===
chunk_dir = "data/chunks"
texts = []
metadatas = []

print(f"\n📖 Загрузка чанков из {chunk_dir}...")

chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith(".txt") and not f.endswith("_meta.txt")]
print(f"📁 Found {len(chunk_files)} chunk files")

for filename in chunk_files:
    path = os.path.join(chunk_dir, filename)
    meta_path = os.path.join(chunk_dir, filename.replace(".txt", "_meta.txt"))
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if len(text) < 10:
            print(f"⚠️  Skipping {filename}: too short ({len(text)} chars)")
            continue
            
        # Load metadata if available
        metadata = {"filename": filename, "text": text[:200] + "..." if len(text) > 200 else text}
        
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                for line in mf:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        metadata[key] = value
        
        texts.append(text)
        metadatas.append(metadata)
        
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        continue

print(f"📊 Загружено {len(texts)} чанков")

if len(texts) == 0:
    print("❌ No texts loaded! Exiting.")
    exit(1)

# === Шаг 5: Функция получения эмбеддингов ===
def get_embedding(text: str):
    """Get embedding using bge-m3 (multilingual, strong for ru/kz legal)"""
    try:
        text = text.replace("\n", " ")
        embedding = sentence_model.encode(
            text,
            normalize_embeddings=True,
            prompt=EMBEDDING_PASSAGE_PROMPT
        )
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None

# === Шаг 6: Векторизация и загрузка в Pinecone ===
batch_size = 50  # Back to normal batch size

print(f"\n🚀 Начинаем индексацию с batch_size={batch_size}...")

successful_uploads = 0
failed_uploads = 0
detailed_errors = []

for i in tqdm(range(0, len(texts), batch_size), desc="📦 Индексация в Pinecone"):
    batch_texts = texts[i:i + batch_size]
    batch_metadatas = metadatas[i:i + batch_size]
    
    vectors_to_upsert = []
        
    for j, (text, metadata) in enumerate(zip(batch_texts, batch_metadatas)):
        try:
            # Get bge-m3 embedding
            embedding = get_embedding(text)
            
            if embedding is None:
                error_msg = f"Failed to get embedding for {metadata.get('filename', 'unknown')}"
                detailed_errors.append(error_msg)
                failed_uploads += 1
                continue
            
            # Enhanced metadata - only include Pinecone-compatible types
            enhanced_metadata = {
                "filename": metadata.get("filename", ""),
                "text_preview": metadata.get("text", "")[:500],  # Limit text preview
                "text_length": len(text),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "has_local_embedding": True
            }
            
            # Add other metadata fields if they exist and are compatible
            for key, value in metadata.items():
                if key not in enhanced_metadata:
                    # Only add if it's a string, number, or boolean
                    if isinstance(value, (str, int, float, bool)):
                        enhanced_metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        enhanced_metadata[key] = value
            
            vectors_to_upsert.append({
                "id": f"doc-{i + j}",
                "values": embedding,
                "metadata": enhanced_metadata
            })
            
        except Exception as e:
            error_msg = f"Error processing {metadata.get('filename', 'unknown')}: {e}"
            detailed_errors.append(error_msg)
            failed_uploads += 1
            continue
    
    # Upload batch
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            successful_uploads += len(vectors_to_upsert)
        except Exception as e:
            error_msg = f"Error uploading batch {i//batch_size + 1}: {e}"
            detailed_errors.append(error_msg)
            print(f"❌ {error_msg}")
            failed_uploads += len(vectors_to_upsert)

print(f"\n✅ Индексация завершена!")
print(f"📊 Успешно загружено: {successful_uploads}")
print(f"❌ Ошибок: {failed_uploads}")

# Save detailed error log
if detailed_errors:
    with open("embedding_errors.log", "w", encoding="utf-8") as f:
        f.write("Embedding Errors Log\n")
        f.write("=" * 50 + "\n")
        for error in detailed_errors:
            f.write(f"{error}\n")
    print(f"📝 Detailed errors saved to embedding_errors.log")

# Save index statistics
stats = {
    "total_chunks": len(texts),
    "successful_uploads": successful_uploads,
    "failed_uploads": failed_uploads,
    "index_name": INDEX_NAME,
    "errors": detailed_errors[:10]  # Save first 10 errors
}

with open("index_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("📈 Статистика сохранена в index_stats.json") 
