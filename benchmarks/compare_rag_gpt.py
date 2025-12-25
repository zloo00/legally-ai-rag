import os
import time

import openai
import pandas as pd
from dotenv import load_dotenv

from legal_rag.app.legal_chat import LegalChatBot

# Загрузка переменных окружения (для OpenAI)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Загрузка исходных данных
INPUT_FILE = "jurhelp_questions.xlsx"
OUTPUT_FILE = "jurhelp_comparison.xlsx"

df = pd.read_excel(INPUT_FILE)

# 2. Инициализация бота
chatbot = LegalChatBot(model="gpt-4")  # или gpt-3.5-turbo

# 3. Получение ответов
rag_answers = []
gpt_answers = []

for i, row in enumerate(df.itertuples(index=False), start=1):
    q = str(getattr(row, "Вопрос"))
    print(f"[{i}/{len(df)}] {q[:60]}...")
    # Ответ RAG
    rag_result = chatbot.get_legal_answer(q)
    rag_answers.append(rag_result["answer"])
    # Ответ GPT (без RAG)
    gpt_answer = chatbot.get_general_answer(q)
    gpt_answers.append(gpt_answer)
    time.sleep(1)  # чтобы не получить бан от OpenAI

# 4. Сохраняем результаты
df["Ответ RAG"] = rag_answers
df["Ответ GPT"] = gpt_answers

df.to_excel(OUTPUT_FILE, index=False)
print(f"Готово! Сравнение сохранено в {OUTPUT_FILE}") 
