# Python 3.10 базалық образ
FROM python:3.10-slim

# Жұмыс директориясын орнату
WORKDIR /app

# Тәуелділіктерді көшіру және орнату
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Барлық кодты көшіру
COPY . .

# Портты ашу (мысалы, Flask үшін 5000)
EXPOSE 5000

# web_legal_chat.py-ды іске қосу (егер Flask болса)
CMD ["python", "web_legal_chat.py"] 