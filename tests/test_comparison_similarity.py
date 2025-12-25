import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 1. Загрузка таблицы
INPUT_FILE = "test_comparison.xlsx"
OUTPUT_FILE = "test_comparison_with_similarity.xlsx"

df = pd.read_excel(INPUT_FILE)

# 2. Модель для эмбеддингов
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Считаем сходство для каждой пары
similarities = []
for i, row in df.iterrows():
    answer1 = str(row["Ответ"])      # Ответ юриста
    answer2 = str(row["Ответ_RAG"])  # Ответ RAG
    emb1 = model.encode(answer1)
    emb2 = model.encode(answer2)
    score = float(util.cos_sim(emb1, emb2))
    similarities.append(score)

# 4. Добавляем колонку и сохраняем
df["Сходство_RAG_Юрист"] = similarities
df.to_excel(OUTPUT_FILE, index=False)
print(f"Готово! Сходство добавлено в файл {OUTPUT_FILE}") 