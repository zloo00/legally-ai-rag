#!/usr/bin/env python3
"""
Демонстрационный скрипт для benchmark'ов RAG системы
Показывает основные возможности и метрики
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

from benchmark_rag import RAGBenchmark
from benchmark_quality import QualityBenchmark
from benchmark_load_test import LoadTestBenchmark

load_dotenv()

def demo_quick_benchmark():
    """Демонстрация быстрого benchmark'а"""
    print("🎯 Демонстрация быстрого benchmark'а")
    print("=" * 60)
    
    # Создаем benchmark
    benchmark = RAGBenchmark()
    
    # Тестовые вопросы для демо
    demo_questions = [
        "Что такое гражданское право?",
        "Какие права имеет собственник имущества?",
        "Что такое трудовой договор?"
    ]
    
    print("📝 Тестовые вопросы:")
    for i, question in enumerate(demo_questions, 1):
        print(f"   {i}. {question}")
    
    print("\n🚀 Запуск тестирования...")
    
    try:
        # Инициализируем RAG систему
        from rag_system import EnhancedRAGSystem
        rag_system = EnhancedRAGSystem()
        
        # Тестируем каждый вопрос
        results = []
        for i, question in enumerate(demo_questions, 1):
            print(f"\n[{i}/{len(demo_questions)}] Тестирование: {question}")
            
            start_time = time.time()
            try:
                result = rag_system.query(question)
                end_time = time.time()
                
                response_time = end_time - start_time
                answer_length = len(result.get("answer", ""))
                sources_count = len(result.get("sources", []))
                
                print(f"   ✅ Успешно (время: {response_time:.2f}с)")
                print(f"   📏 Длина ответа: {answer_length} символов")
                print(f"   📚 Источников: {sources_count}")
                
                results.append({
                    "question": question,
                    "response_time": response_time,
                    "answer_length": answer_length,
                    "sources_count": sources_count,
                    "success": True
                })
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                results.append({
                    "question": question,
                    "response_time": 0,
                    "answer_length": 0,
                    "sources_count": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Вычисляем статистику
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
            avg_answer_length = sum(r["answer_length"] for r in successful_results) / len(successful_results)
            avg_sources_count = sum(r["sources_count"] for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / len(results)
            
            print(f"\n📊 Результаты демо:")
            print(f"   Успешность: {success_rate:.2%}")
            print(f"   Среднее время ответа: {avg_response_time:.2f}с")
            print(f"   Средняя длина ответа: {avg_answer_length:.0f} символов")
            print(f"   Среднее количество источников: {avg_sources_count:.1f}")
        else:
            print("\n❌ Все тесты завершились с ошибками")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")

def demo_quality_metrics():
    """Демонстрация метрик качества"""
    print("\n🎯 Демонстрация метрик качества")
    print("=" * 60)
    
    # Создаем benchmark качества
    quality_benchmark = QualityBenchmark()
    
    # Демонстрационные вопросы
    demo_questions = [
        {
            "question": "Что говорит статья 1 Гражданского кодекса РК?",
            "expected_keywords": ["гражданское право", "отношения", "имущественные"],
            "expected_sources": ["Гражданский кодекс", "статья 1"]
        },
        {
            "question": "Какие права имеет собственник имущества?",
            "expected_keywords": ["владение", "пользование", "распоряжение"],
            "expected_sources": ["Гражданский кодекс", "право собственности"]
        }
    ]
    
    print("📝 Демонстрационные вопросы с ожидаемыми ответами:")
    for i, q_data in enumerate(demo_questions, 1):
        print(f"   {i}. {q_data['question']}")
        print(f"      Ожидаемые ключевые слова: {', '.join(q_data['expected_keywords'])}")
        print(f"      Ожидаемые источники: {', '.join(q_data['expected_sources'])}")
    
    print("\n🚀 Запуск тестирования качества...")
    
    try:
        # Инициализируем RAG систему
        from rag_system import EnhancedRAGSystem
        rag_system = EnhancedRAGSystem()
        
        # Тестируем качество
        quality_results = []
        for i, question_data in enumerate(demo_questions, 1):
            print(f"\n[{i}/{len(demo_questions)}] Тестирование качества: {question_data['question']}")
            
            result = quality_benchmark.measure_quality_metrics(rag_system, question_data)
            quality_results.append(result)
            
            if "error" not in result:
                print(f"   ✅ Успешно")
                print(f"   📊 Оценка по ключевым словам: {result['keyword_score']:.2f}")
                print(f"   📚 Оценка по источникам: {result['source_score']:.2f}")
                print(f"   📏 Оценка по длине ответа: {result['answer_length_score']:.2f}")
                print(f"   🎯 Общая оценка: {result['overall_score']:.2f}")
            else:
                print(f"   ❌ Ошибка: {result['error']}")
        
        # Вычисляем общую статистику
        successful_results = [r for r in quality_results if "error" not in r]
        if successful_results:
            avg_keyword_score = sum(r["keyword_score"] for r in successful_results) / len(successful_results)
            avg_source_score = sum(r["source_score"] for r in successful_results) / len(successful_results)
            avg_overall_score = sum(r["overall_score"] for r in successful_results) / len(successful_results)
            
            print(f"\n📊 Результаты качества:")
            print(f"   Средняя оценка по ключевым словам: {avg_keyword_score:.2f}")
            print(f"   Средняя оценка по источникам: {avg_source_score:.2f}")
            print(f"   Средняя общая оценка: {avg_overall_score:.2f}")
        else:
            print("\n❌ Все тесты качества завершились с ошибками")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")

def demo_load_test():
    """Демонстрация нагрузочного тестирования"""
    print("\n🎯 Демонстрация нагрузочного тестирования")
    print("=" * 60)
    
    # Создаем benchmark нагрузки
    load_test = LoadTestBenchmark()
    
    print("📝 Конфигурация нагрузочного тестирования:")
    print("   Одновременных запросов: 2")
    print("   Длительность: 30 секунд")
    print("   Вопросов для тестирования: 5")
    
    print("\n🚀 Запуск нагрузочного тестирования...")
    
    try:
        # Инициализируем RAG систему
        from rag_system import EnhancedRAGSystem
        rag_system = EnhancedRAGSystem()
        
        # Конфигурация для демо
        demo_config = {
            "concurrent_queries": 2,
            "duration_seconds": 30,
            "name": "demo_load"
        }
        
        # Запускаем тест
        result = load_test.run_single_load_test(rag_system, demo_config)
        
        print(f"\n📊 Результаты нагрузочного тестирования:")
        print(f"   Всего запросов: {result['total_queries']}")
        print(f"   Успешных: {result['successful_queries']}")
        print(f"   Неудачных: {result['failed_queries']}")
        print(f"   Успешность: {result['success_rate']:.2%}")
        print(f"   Запросов в секунду: {result['queries_per_second']:.2f}")
        print(f"   Среднее время ответа: {result['avg_response_time']:.2f}с")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")

def main():
    """Основная функция демонстрации"""
    print("🎯 Демонстрация benchmark'ов RAG системы")
    print("=" * 80)
    
    # Проверяем переменные окружения
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        print("Убедитесь, что файл .env настроен правильно")
        return
    
    print("✅ Переменные окружения настроены")
    
    # Запускаем демонстрации
    try:
        demo_quick_benchmark()
        demo_quality_metrics()
        demo_load_test()
        
        print("\n🎉 Демонстрация завершена!")
        print("\n💡 Для полного тестирования используйте:")
        print("   make benchmark     # Полный benchmark")
        print("   make performance   # Тест производительности")
        print("   make quality       # Тест качества")
        print("   make load          # Нагрузочное тестирование")
        
    except KeyboardInterrupt:
        print("\n⏹️ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при демонстрации: {e}")

if __name__ == "__main__":
    main()

