#!/usr/bin/env python3
"""
Тестирование RAG системы для юридических документов
"""

import os
from rag_system import EnhancedRAGSystem
from dotenv import load_dotenv

load_dotenv()

def test_rag_system():
    """Тестирование RAG системы"""
    print("🧪 Тестирование RAG системы для юридических документов")
    print("=" * 60)
    
    # Инициализация RAG системы
    try:
        rag = EnhancedRAGSystem()
        print("✅ RAG система инициализирована успешно")
        
        # Получение статистики
        stats = rag.get_system_stats()
        print(f"📊 Статистика системы:")
        print(f"   Векторов в индексе: {stats.get('total_vectors', 0)}")
        print(f"   Размерность: {stats.get('index_dimension', 0)}")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка инициализации RAG системы: {e}")
        return
    
    # Тестовые вопросы
    test_questions = [
        "Что такое гражданское право?",
        "Какие права имеет собственник имущества?",
        "Как заключается договор купли-продажи?",
        "Что такое трудовой договор?",
        "Какие основания для расторжения брака?",
        "Что такое наследование по закону?",
        "Какие виды ответственности предусмотрены в гражданском праве?",
        "Как защищаются права потребителей?",
        "Что такое административная ответственность?",
        "Какие права имеет работник при увольнении?"
    ]
    
    print("🔍 Тестирование поиска и ответов:")
    print("-" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        print("-" * 40)
        
        try:
            # Выполнение запроса
            result = rag.query(question)
            
            # Вывод ответа
            print(f"🤖 Ответ: {result['answer']}")
            
            # Вывод источников
            if result.get('sources'):
                print(f"📚 Источники: {', '.join(result['sources'])}")
            
            # Вывод статистики
            print(f"🔍 Найдено документов: {result.get('results_count', 0)}")
            print(f"📏 Длина контекста: {result.get('context_length', 0)} символов")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке вопроса: {e}")
        
        print()

def test_specific_legal_questions():
    """Тестирование конкретных юридических вопросов"""
    print("\n🎯 Тестирование конкретных юридических вопросов")
    print("=" * 60)
    
    try:
        rag = EnhancedRAGSystem()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Специфические вопросы по казахстанскому законодательству
    specific_questions = [
        {
            "question": "Что говорит статья 1 Гражданского кодекса РК?",
            "description": "Тест поиска конкретной статьи"
        },
        {
            "question": "Какие права имеет участник полного товарищества?",
            "description": "Тест поиска по хозяйственному праву"
        },
        {
            "question": "Как определяется дееспособность гражданина?",
            "description": "Тест поиска по гражданскому праву"
        },
        {
            "question": "Что такое обязательство и как оно возникает?",
            "description": "Тест поиска по обязательственному праву"
        },
        {
            "question": "Какие виды собственности предусмотрены в законодательстве?",
            "description": "Тест поиска по праву собственности"
        }
    ]
    
    for i, q_data in enumerate(specific_questions, 1):
        print(f"\n{i}. {q_data['description']}")
        print(f"Вопрос: {q_data['question']}")
        print("-" * 50)
        
        try:
            result = rag.query(q_data['question'])
            
            print(f"🤖 Ответ: {result['answer']}")
            
            if result.get('sources'):
                print(f"📚 Источники: {', '.join(result['sources'])}")
            
            # Показать найденные документы
            if result.get('search_results'):
                print(f"🔍 Найденные документы:")
                for j, doc in enumerate(result['search_results'][:3], 1):
                    print(f"   {j}. {doc['source']} (релевантность: {doc['score']:.3f})")
                    print(f"      {doc['text']}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print()

def test_conversation_context():
    """Тестирование контекста разговора"""
    print("\n💬 Тестирование контекста разговора")
    print("=" * 60)
    
    try:
        rag = EnhancedRAGSystem()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Серия связанных вопросов
    conversation = [
        "Что такое гражданское право?",
        "А какие основные принципы гражданского права?",
        "Как эти принципы применяются в договорных отношениях?",
        "Приведи примеры договоров в гражданском праве"
    ]
    
    print("💬 Серия связанных вопросов:")
    
    for i, question in enumerate(conversation, 1):
        print(f"\n{i}. Вопрос: {question}")
        print("-" * 40)
        
        try:
            result = rag.query(question)
            print(f"🤖 Ответ: {result['answer']}")
            
            if result.get('sources'):
                print(f"📚 Источники: {', '.join(result['sources'])}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    # Показать историю разговора
    print(f"\n📜 История разговора ({len(rag.conversation_history)} записей):")
    for i, turn in enumerate(rag.conversation_history, 1):
        print(f"   {i}. {turn.user_query[:50]}...")

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования RAG системы")
    print("=" * 60)
    
    # Проверка переменных окружения
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        print("Убедитесь, что файл .env настроен правильно")
        return
    
    print("✅ Переменные окружения настроены")
    
    # Запуск тестов
    try:
        test_rag_system()
        test_specific_legal_questions()
        test_conversation_context()
        
        print("\n🎉 Тестирование завершено!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")

if __name__ == "__main__":
    main() 