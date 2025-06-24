# 🤖 Юридический AI-ассистент

Интеллектуальная система для анализа казахстанского законодательства с использованием RAG (Retrieval-Augmented Generation) технологии.

## 🚀 Быстрый старт

### 1. Установка
```bash
pip install -r requirements.txt
```

### 2. Настройка
Создайте файл `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=legally-index
```

### 3. Индексация документов
```bash
python preprocess_articles.py
python embed_and_index_fixed.py
```

### 4. Запуск

#### Консольный интерфейс:
```bash
python legal_chat.py
```

#### Веб-интерфейс:
```bash
python web_legal_chat.py
```
Откройте: http://localhost:5001

## 📚 Возможности

- ✅ Поиск в 744 статьях казахстанского законодательства
- ✅ Умные ответы на основе найденных документов
- ✅ Гибридный поиск (векторный + ключевой)
- ✅ Автоматическое определение типа вопроса
- ✅ Контекстные разговоры с историей
- ✅ Современный веб-интерфейс

## 📁 Структура

```
├── rag_system.py              # Основная RAG система
├── legal_chat.py              # Консольный чат
├── web_legal_chat.py          # Веб-сервер
├── test_rag.py                # Тестирование
├── preprocess_articles.py     # Обработка документов
├── embed_and_index_fixed.py   # Индексация
├── templates/legal_chat.html  # Веб-интерфейс
└── data/                      # Документы и чанки
```

## 🧪 Тестирование

```bash
python test_rag.py
```

## 📖 Подробная документация

См. [README_LEGAL_RAG.md](README_LEGAL_RAG.md) для полной документации.

---

**🎉 Готово к использованию! Задавайте вопросы по казахстанскому законодательству!** 