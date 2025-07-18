import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone

# Загружаем переменные окружения
load_dotenv()

print("🔧 Проверка API-ключей...")

# Проверка OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✅ OpenAI API ключ найден: {openai_key[:10]}...")
    
    # Тест OpenAI
    try:
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Привет"}],
            max_tokens=10
        )
        print("✅ OpenAI API работает")
    except Exception as e:
        print(f"❌ OpenAI API ошибка: {e}")
else:
    print("❌ OpenAI API ключ не найден в .env")

# Проверка Pinecone
pinecone_key = os.getenv("PINECONE_API_KEY")
if pinecone_key:
    print(f"✅ Pinecone API ключ найден: {pinecone_key[:10]}...")
    
    # Тест Pinecone
    try:
        pc = Pinecone(api_key=pinecone_key)
        indexes = pc.list_indexes()
        print(f"✅ Pinecone API работает. Доступные индексы: {[idx.name for idx in indexes]}")
    except Exception as e:
        print(f"❌ Pinecone API ошибка: {e}")
else:
    print("❌ Pinecone API ключ не найден в .env")

print("\n💡 Если ключи не найдены, создайте файл .env с:")
print("OPENAI_API_KEY=your_openai_api_key")
print("PINECONE_API_KEY=your_pinecone_api_key")
print("PINECONE_ENVIRONMENT=us-east-1")
print("PINECONE_INDEX_NAME=legally-index") 