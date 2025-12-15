#!/usr/bin/env python3
"""
Benchmark для RAG системы юридических документов
Измеряет производительность, качество и стабильность системы
"""

import os
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv

from rag_system import EnhancedRAGSystem
from rag_factory import EnhancedRAGSystem

rag = EnhancedRAGSystem()

load_dotenv()

class RAGBenchmark:
    """Класс для проведения benchmark тестов RAG системы"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Тестовые вопросы для benchmark
        self.test_questions = [
            "Что такое гражданское право?",
            "Какие права имеет собственник имущества?",
            "Как заключается договор купли-продажи?",
            "Что такое трудовой договор?",
            "Какие основания для расторжения брака?",
            "Что такое наследование по закону?",
            "Какие виды ответственности предусмотрены в гражданском праве?",
            "Как защищаются права потребителей?",
            "Что такое административная ответственность?",
            "Какие права имеет работник при увольнении?",
            "Что говорит статья 1 Гражданского кодекса РК?",
            "Какие права имеет участник полного товарищества?",
            "Как определяется дееспособность гражданина?",
            "Что такое обязательство и как оно возникает?",
            "Какие виды собственности предусмотрены в законодательстве?",
            "Как защищаются авторские права?",
            "Что такое лицензионный договор?",
            "Какие права имеет наследник?",
            "Как определяется размер алиментов?",
            "Что такое презумпция невиновности?"
        ]
        
        # Специальные вопросы для тестирования качества
        self.quality_questions = [
            {
                "question": "Что говорит статья 1 Гражданского кодекса РК?",
                "expected_keywords": ["гражданское право", "отношения", "имущественные", "личные"],
                "expected_sources": ["Гражданский кодекс", "статья 1"]
            },
            {
                "question": "Какие права имеет собственник имущества?",
                "expected_keywords": ["владение", "пользование", "распоряжение", "собственность"],
                "expected_sources": ["Гражданский кодекс", "право собственности"]
            },
            {
                "question": "Что такое трудовой договор?",
                "expected_keywords": ["трудовой договор", "работник", "работодатель", "трудовые отношения"],
                "expected_sources": ["Трудовой кодекс", "трудовой договор"]
            }
        ]

    def measure_query_performance(self, rag_system, question: str, iterations: int = 3) -> Dict[str, Any]:
        """Измеряет производительность одного запроса"""
        times = []
        results = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = rag_system.query(question)
                end_time = time.time()
                
                times.append(end_time - start_time)
                results.append(result)
                
            except Exception as e:
                print(f"Ошибка при выполнении запроса: {e}")
                times.append(float('inf'))
                results.append(None)
        
        return {
            "question": question,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "success_rate": len([r for r in results if r is not None]) / len(results),
            "results": results[0] if results[0] else None
        }

    def measure_system_stats(self, rag_system) -> Dict[str, Any]:
        """Измеряет статистику системы"""
        try:
            stats = rag_system.get_system_stats()
            return {
                "total_vectors": stats.get('total_vectors', 0),
                "index_dimension": stats.get('index_dimension', 0),
                "index_name": stats.get('index_name', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    def measure_quality_metrics(self, rag_system, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Измеряет качество ответа"""
        question = question_data["question"]
        expected_keywords = question_data.get("expected_keywords", [])
        expected_sources = question_data.get("expected_sources", [])
        
        try:
            result = rag_system.query(question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Проверяем наличие ожидаемых ключевых слов
            keyword_matches = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in answer.lower())
            keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
            
            # Проверяем наличие ожидаемых источников
            source_matches = sum(1 for source in expected_sources 
                               if any(source.lower() in s.lower() for s in sources))
            source_score = source_matches / len(expected_sources) if expected_sources else 0
            
            # Длина ответа
            answer_length = len(answer)
            
            # Количество источников
            sources_count = len(sources)
            
            return {
                "question": question,
                "keyword_score": keyword_score,
                "source_score": source_score,
                "answer_length": answer_length,
                "sources_count": sources_count,
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "keyword_score": 0,
                "source_score": 0,
                "answer_length": 0,
                "sources_count": 0
            }

    def run_performance_benchmark(self, rag_system, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает benchmark производительности"""
        print(f"🚀 Запуск benchmark производительности для {engine_name}")
        print("=" * 60)
        
        # Измеряем статистику системы
        system_stats = self.measure_system_stats(rag_system)
        print(f"📊 Статистика системы: {system_stats}")
        
        # Измеряем производительность каждого запроса
        performance_results = []
        for i, question in enumerate(self.test_questions, 1):
            print(f"[{i}/{len(self.test_questions)}] Тестирование: {question[:50]}...")
            result = self.measure_query_performance(rag_system, question)
            performance_results.append(result)
        
        # Вычисляем общую статистику
        avg_times = [r["avg_time"] for r in performance_results if r["avg_time"] != float('inf')]
        success_rates = [r["success_rate"] for r in performance_results]
        
        benchmark_result = {
            "engine": engine_name,
            "timestamp": datetime.now().isoformat(),
            "system_stats": system_stats,
            "total_questions": len(self.test_questions),
            "avg_query_time": statistics.mean(avg_times) if avg_times else 0,
            "min_query_time": min(avg_times) if avg_times else 0,
            "max_query_time": max(avg_times) if avg_times else 0,
            "avg_success_rate": statistics.mean(success_rates),
            "performance_results": performance_results
        }
        
        return benchmark_result

    def run_quality_benchmark(self, rag_system, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает benchmark качества"""
        print(f"🎯 Запуск benchmark качества для {engine_name}")
        print("=" * 60)
        
        quality_results = []
        for i, question_data in enumerate(self.quality_questions, 1):
            print(f"[{i}/{len(self.quality_questions)}] Тестирование качества: {question_data['question'][:50]}...")
            result = self.measure_quality_metrics(rag_system, question_data)
            quality_results.append(result)
        
        # Вычисляем общую статистику качества
        keyword_scores = [r["keyword_score"] for r in quality_results if "keyword_score" in r]
        source_scores = [r["source_score"] for r in quality_results if "source_score" in r]
        answer_lengths = [r["answer_length"] for r in quality_results if "answer_length" in r]
        sources_counts = [r["sources_count"] for r in quality_results if "sources_count" in r]
        
        quality_benchmark = {
            "engine": engine_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(self.quality_questions),
            "avg_keyword_score": statistics.mean(keyword_scores) if keyword_scores else 0,
            "avg_source_score": statistics.mean(source_scores) if source_scores else 0,
            "avg_answer_length": statistics.mean(answer_lengths) if answer_lengths else 0,
            "avg_sources_count": statistics.mean(sources_counts) if sources_counts else 0,
            "quality_results": quality_results
        }
        
        return quality_benchmark

    def run_load_test(self, rag_system, concurrent_queries: int = 5, duration_seconds: int = 60) -> Dict[str, Any]:
        """Запускает нагрузочное тестирование"""
        print(f"⚡ Запуск нагрузочного тестирования для {concurrent_queries} одновременных запросов")
        print("=" * 60)
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        start_time = time.time()
        
        def worker():
            while time.time() - start_time < duration_seconds:
                question = self.test_questions[hash(str(time.time())) % len(self.test_questions)]
                try:
                    result = rag_system.query(question)
                    results_queue.put({
                        "success": True,
                        "time": time.time() - start_time,
                        "question": question,
                        "result": result
                    })
                except Exception as e:
                    results_queue.put({
                        "success": False,
                        "time": time.time() - start_time,
                        "question": question,
                        "error": str(e)
                    })
                time.sleep(0.1)  # Небольшая задержка между запросами
        
        # Запускаем worker'ы
        threads = []
        for _ in range(concurrent_queries):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Ждем завершения
        time.sleep(duration_seconds)
        
        # Собираем результаты
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Вычисляем статистику
        successful_queries = [r for r in results if r["success"]]
        failed_queries = [r for r in results if not r["success"]]
        
        load_test_result = {
            "concurrent_queries": concurrent_queries,
            "duration_seconds": duration_seconds,
            "total_queries": len(results),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "success_rate": len(successful_queries) / len(results) if results else 0,
            "queries_per_second": len(results) / duration_seconds,
            "results": results
        }
        
        return load_test_result

    def run_full_benchmark(self, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает полный benchmark"""
        print(f"🎯 Запуск полного benchmark для {engine_name}")
        print("=" * 80)
        
        # Инициализируем RAG систему
        try:
            if engine_name == "baseline":
                rag_system = EnhancedRAGSystem()
            else:
                rag_system = RAGFactory.create_rag_system(engine_name)
        except Exception as e:
            print(f"❌ Ошибка инициализации {engine_name}: {e}")
            return {"error": str(e)}
        
        # Запускаем все тесты
        performance_result = self.run_performance_benchmark(rag_system, engine_name)
        quality_result = self.run_quality_benchmark(rag_system, engine_name)
        load_test_result = self.run_load_test(rag_system)
        
        # Объединяем результаты
        full_result = {
            "engine": engine_name,
            "timestamp": datetime.now().isoformat(),
            "performance": performance_result,
            "quality": quality_result,
            "load_test": load_test_result
        }
        
        # Сохраняем результаты
        self.save_results(full_result, engine_name)
        
        return full_result

    def save_results(self, results: Dict[str, Any], engine_name: str):
        """Сохраняет результаты benchmark в файлы"""
        # JSON файл
        json_file = f"{self.output_dir}/benchmark_{engine_name}_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV файл с результатами производительности
        if "performance" in results and "performance_results" in results["performance"]:
            perf_df = pd.DataFrame(results["performance"]["performance_results"])
            csv_file = f"{self.output_dir}/performance_{engine_name}_{self.timestamp}.csv"
            perf_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # CSV файл с результатами качества
        if "quality" in results and "quality_results" in results["quality"]:
            qual_df = pd.DataFrame(results["quality"]["quality_results"])
            csv_file = f"{self.output_dir}/quality_{engine_name}_{self.timestamp}.csv"
            qual_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"📁 Результаты сохранены в {self.output_dir}/")

    def compare_engines(self, engines: List[str] = ["baseline"]) -> Dict[str, Any]:
        """Сравнивает разные движки RAG"""
        print("🔄 Сравнение движков RAG")
        print("=" * 60)
        
        comparison_results = {}
        
        for engine in engines:
            print(f"\n🔧 Тестирование {engine}...")
            try:
                result = self.run_full_benchmark(engine)
                comparison_results[engine] = result
            except Exception as e:
                print(f"❌ Ошибка при тестировании {engine}: {e}")
                comparison_results[engine] = {"error": str(e)}
        
        # Создаем сравнительную таблицу
        comparison_summary = self.create_comparison_summary(comparison_results)
        
        # Сохраняем сравнение
        comparison_file = f"{self.output_dir}/comparison_{self.timestamp}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        return comparison_results

    def create_comparison_summary(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """Создает сводную таблицу сравнения движков"""
        summary_data = []
        
        for engine, results in comparison_results.items():
            if "error" in results:
                continue
                
            perf = results.get("performance", {})
            qual = results.get("quality", {})
            load = results.get("load_test", {})
            
            summary_data.append({
                "Engine": engine,
                "Avg Query Time (s)": perf.get("avg_query_time", 0),
                "Success Rate": perf.get("avg_success_rate", 0),
                "Keyword Score": qual.get("avg_keyword_score", 0),
                "Source Score": qual.get("avg_source_score", 0),
                "Queries/Second": load.get("queries_per_second", 0),
                "Load Success Rate": load.get("success_rate", 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Сохраняем сводную таблицу
        summary_file = f"{self.output_dir}/comparison_summary_{self.timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print("\n📊 Сводная таблица сравнения:")
        print(summary_df.to_string(index=False))
        
        return summary_df

def main():
    """Основная функция для запуска benchmark"""
    print("🎯 RAG System Benchmark")
    print("=" * 60)
    
    # Проверяем переменные окружения
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        return
    
    # Создаем benchmark
    benchmark = RAGBenchmark()
    
    # Запускаем benchmark для baseline
    print("🚀 Запуск benchmark для baseline движка...")
    baseline_result = benchmark.run_full_benchmark("baseline")
    
    # Если есть другие движки, сравниваем их
    available_engines = ["baseline"]  # Можно добавить "graphrag", "lightrag"
    
    if len(available_engines) > 1:
        print("\n🔄 Сравнение движков...")
        comparison_results = benchmark.compare_engines(available_engines)
    
    print("\n🎉 Benchmark завершен!")
    print(f"📁 Результаты сохранены в {benchmark.output_dir}/")

if __name__ == "__main__":
    main()
