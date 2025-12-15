FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pinecone-client || true \
    && pip install --no-cache-dir --upgrade pinecone
COPY . .
EXPOSE 5000
CMD ["python", "web_legal_chat.py"]