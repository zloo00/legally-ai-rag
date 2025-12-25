import os
import time

import pandas as pd
from dotenv import load_dotenv

from legal_rag.app.legal_chat import LegalChatBot

# Загрузка переменных окружения
load_dotenv()

# 1. Загрузка исходных данных
INPUT_FILE = "jurhelp_questions.xlsx"
OUTPUT_FILE = "test_rag_only.xlsx"

df = pd.read_excel(INPUT_FILE)

# 2. Инициализация бота (только RAG)
chatbot = LegalChatBot(model="gpt-4")

# 3. Получение ответов (только RAG, первые 5)
rag_answers = []

# Ограничиваем до 5 вопросов
test_df = df.head(5)

for i, row in enumerate(test_df.itertuples(index=False), start=1):
    q = str(getattr(row, "Вопрос"))
    print(f"[{i}/5] {q[:60]}...")
    
    try:
        # Только ответ RAG
        rag_result = chatbot.get_legal_answer(q)
        rag_answers.append(rag_result["answer"])
        print(f"  RAG: {rag_result['answer'][:100]}...")
        
        # Показываем источники
        if rag_result.get("sources"):
            print(f"  📚 Источники: {rag_result['sources']}")
        
    except Exception as e:
        print(f"  Ошибка: {e}")
        rag_answers.append("Ошибка")
    
    print()

# 4. Сохранение результатов
test_df = test_df.copy()
test_df["Ответ_RAG"] = rag_answers

test_df.to_excel(OUTPUT_FILE, index=False)
print(f"Результат сохранен в {OUTPUT_FILE}")
print(f"Обработано {len(test_df)} вопросов")

# 5. Статистика
print("\n📊 Статистика:")
stats = chatbot.get_system_stats()
print(f"  Векторов в индексе: {stats.get('total_vectors', 0)}")
print(f"  Размерность индекса: {stats.get('index_dimension', 0)}") 
