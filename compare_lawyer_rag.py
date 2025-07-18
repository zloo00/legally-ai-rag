import pandas as pd
import time
from dotenv import load_dotenv
import os

from legal_chat import LegalChatBot

# Загрузка переменных окружения
load_dotenv()

# 1. Загрузка исходных данных
INPUT_FILE = "jurhelp_questions.xlsx"
OUTPUT_FILE = "lawyer_vs_rag.xlsx"

df = pd.read_excel(INPUT_FILE)

# 2. Инициализация бота (только RAG)
chatbot = LegalChatBot(model="gpt-4")

# 3. Получение ответов (первые 10 для теста)
rag_answers = []
sources_list = []

# Ограничиваем до 10 вопросов
test_df = df.head(10)

for i, row in enumerate(test_df.itertuples(index=False), start=1):
    q = str(getattr(row, "Вопрос"))
    lawyer_answer = str(getattr(row, "Ответ"))
    
    print(f"[{i}/10] {q[:60]}...")
    
    try:
        # Ответ RAG
        rag_result = chatbot.get_legal_answer(q)
        rag_answer = rag_result["answer"]
        rag_answers.append(rag_answer)
        
        # Источники
        sources = rag_result.get("sources", [])
        sources_list.append(", ".join(sources) if sources else "")
        
        print(f"  🤖 RAG: {rag_answer[:100]}...")
        print(f"  👨‍💼 Юрист: {lawyer_answer[:100]}...")
        if sources:
            print(f"  📚 Источники: {sources}")
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        rag_answers.append("Ошибка")
        sources_list.append("")
    
    print()

# 4. Сохранение результатов
test_df = test_df.copy()
test_df["Ответ_RAG"] = rag_answers
test_df["Источники_RAG"] = sources_list

# Переименуем колонки для ясности
test_df = test_df.rename(columns={
    "Ответ": "Ответ_Юриста",
    "Вопрос": "Вопрос",
    "ID": "ID",
    "Дата": "Дата",
    "Пользователь": "Пользователь",
    "Категория": "Категория",
    "Эксперт": "Эксперт"
})

test_df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Результат сохранен в {OUTPUT_FILE}")
print(f"📊 Обработано {len(test_df)} вопросов")

# 5. Статистика
print("\n📈 Статистика:")
stats = chatbot.get_system_stats()
print(f"  Векторов в индексе: {stats.get('total_vectors', 0)}")
print(f"  Размерность индекса: {stats.get('index_dimension', 0)}")

# 6. Анализ качества
print("\n🔍 Анализ качества:")
successful_rag = sum(1 for ans in rag_answers if ans != "Ошибка" and "ошибка" not in ans.lower())
print(f"  Успешных ответов RAG: {successful_rag}/{len(test_df)}")
print(f"  Процент успеха: {successful_rag/len(test_df)*100:.1f}%") 