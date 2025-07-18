import pandas as pd
import openai
import time
from dotenv import load_dotenv
import os
import tiktoken

from legal_chat import LegalChatBot

# Загрузка переменных окружения (для OpenAI)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Загрузка исходных данных
INPUT_FILE = "jurhelp_questions.xlsx"
OUTPUT_FILE = "test_comparison.xlsx"

df = pd.read_excel(INPUT_FILE)

# 2. Инициализация бота
chatbot = LegalChatBot(model="gpt-3.5-turbo")  # или gpt-3.5-turbo

# 3. Получение ответов (только первые 100)
rag_answers = []
gpt_answers = []

# Для токенизации
def count_tokens(text, model="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(str(text)))
    except Exception:
        return None

# Ограничиваем до 100 вопросов
test_df = df.head(100)

# Для хранения токенов
question_tokens = []
lawyer_tokens = []
rag_tokens = []
gpt_tokens = []

for i, row in enumerate(test_df.itertuples(index=False), start=1):
    q = str(getattr(row, "Вопрос"))
    lawyer_answer = str(getattr(row, "Ответ")) if hasattr(row, "Ответ") else ""
    print(f"[{i}/100] {q[:60]}...")
    
    try:
        # Ответ RAG
        rag_result = chatbot.get_legal_answer(q)
        rag_answer = rag_result["answer"]
    except Exception as e:
        rag_answer = f"Error: {e}"
    rag_answers.append(rag_answer)
    
    try:
        # Ответ GPT (без RAG)
        gpt_answer = chatbot.get_general_answer(q)
    except Exception as e:
        gpt_answer = f"Error: {e}"
    gpt_answers.append(gpt_answer)
    
    # Токены
    question_tokens.append(count_tokens(q))
    lawyer_tokens.append(count_tokens(lawyer_answer))
    rag_tokens.append(count_tokens(rag_answer))
    gpt_tokens.append(count_tokens(gpt_answer))
    
    time.sleep(1)  # чтобы не получить бан от OpenAI

# Сохраняем результаты
result_df = test_df.copy()
result_df["Ответ_RAG"] = rag_answers
result_df["Ответ_GPT"] = gpt_answers
result_df["Токены_Вопрос"] = question_tokens
result_df["Токены_Юрист"] = lawyer_tokens
result_df["Токены_RAG"] = rag_tokens
result_df["Токены_GPT"] = gpt_tokens

result_df.to_excel(OUTPUT_FILE, index=False)
print(f"Готово! Сравнение и токены сохранены в {OUTPUT_FILE}") 