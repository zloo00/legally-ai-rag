#!/usr/bin/env python3
"""
Нагрузочное тестирование RAG системы
Тестирует производительность под различными нагрузками
"""

import os
import time
import json
import statistics
import threading
import queue
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
import psutil

from rag_system import EnhancedRAGSystem
from rag_factory import RAGFactory

load_dotenv()

class LoadTestBenchmark:
    """Класс для нагрузочного тестирования RAG системы"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Тестовые вопросы для нагрузочного тестирования
        self.load_test_questions = [
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
            "Что такое презумпция невиновности?",
            "Какие права имеет арендатор?",
            "Что такое залог и как он оформляется?",
            "Какие виды договоров существуют в гражданском праве?",
            "Как защищаются права несовершеннолетних?",
            "Что такое эмансипация несовершеннолетних?"
        ]
        
        # Конфигурации нагрузочного тестирования
        self.load_configs = [
            {"concurrent_queries": 1, "duration_seconds": 30, "name": "light_load"},
            {"concurrent_queries": 3, "duration_seconds": 60, "name": "medium_load"},
            {"concurrent_queries": 5, "duration_seconds": 90, "name": "heavy_load"},
            {"concurrent_queries": 10, "duration_seconds": 120, "name": "stress_load"}
        ]

    def get_system_metrics(self) -> Dict[str, Any]:
        """Получает метрики системы"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            return {"error": str(e)}

    def run_single_load_test(self, rag_system, config: Dict[str, Any]) -> Dict[str, Any]:
        """Запускает один тест нагрузки"""
        concurrent_queries = config["concurrent_queries"]
        duration_seconds = config["duration_seconds"]
        test_name = config["name"]
        
        print(f"⚡ Запуск нагрузочного теста: {test_name}")
        print(f"   Одновременных запросов: {concurrent_queries}")
        print(f"   Длительность: {duration_seconds} секунд")
        print("-" * 60)
        
        # Получаем начальные метрики системы
        start_metrics = self.get_system_metrics()
        
        # Очередь для результатов
        results_queue = queue.Queue()
        start_time = time.time()
        
        # Счетчики
        total_queries = 0
        successful_queries = 0
        failed_queries = 0
        total_response_time = 0
        
        def worker(worker_id: int):
            """Worker функция для выполнения запросов"""
            nonlocal total_queries, successful_queries, failed_queries, total_response_time
            
            while time.time() - start_time < duration_seconds:
                # Выбираем случайный вопрос
                question = self.load_test_questions[hash(str(time.time()) + str(worker_id)) % len(self.load_test_questions)]
                
                query_start = time.time()
                try:
                    result = rag_system.query(question)
                    query_end = time.time()
                    
                    response_time = query_end - query_start
                    total_response_time += response_time
                    successful_queries += 1
                    
                    results_queue.put({
                        "success": True,
                        "worker_id": worker_id,
                        "question": question,
                        "response_time": response_time,
                        "timestamp": query_start,
                        "result": result
                    })
                    
                except Exception as e:
                    query_end = time.time()
                    response_time = query_end - query_start
                    total_response_time += response_time
                    failed_queries += 1
                    
                    results_queue.put({
                        "success": False,
                        "worker_id": worker_id,
                        "question": question,
                        "response_time": response_time,
                        "timestamp": query_start,
                        "error": str(e)
                    })
                
                total_queries += 1
                
                # Небольшая задержка между запросами
                time.sleep(0.1)
        
        # Запускаем worker'ы
        threads = []
        for i in range(concurrent_queries):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Ждем завершения
        time.sleep(duration_seconds)
        
        # Получаем финальные метрики системы
        end_metrics = self.get_system_metrics()
        
        # Собираем результаты
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Вычисляем статистику
        if total_queries > 0:
            success_rate = successful_queries / total_queries
            avg_response_time = total_response_time / total_queries
            queries_per_second = total_queries / duration_seconds
        else:
            success_rate = 0
            avg_response_time = 0
            queries_per_second = 0
        
        # Статистика по времени ответа
        response_times = [r["response_time"] for r in results if "response_time" in r]
        if response_times:
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            median_response_time = statistics.median(response_times)
            std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        else:
            min_response_time = 0
            max_response_time = 0
            median_response_time = 0
            std_response_time = 0
        
        load_test_result = {
            "test_name": test_name,
            "concurrent_queries": concurrent_queries,
            "duration_seconds": duration_seconds,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": success_rate,
            "queries_per_second": queries_per_second,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "median_response_time": median_response_time,
            "std_response_time": std_response_time,
            "start_metrics": start_metrics,
            "end_metrics": end_metrics,
            "results": results
        }
        
        print(f"   Завершено запросов: {total_queries}")
        print(f"   Успешных: {successful_queries}")
        print(f"   Неудачных: {failed_queries}")
        print(f"   Успешность: {success_rate:.2%}")
        print(f"   Запросов в секунду: {queries_per_second:.2f}")
        print(f"   Среднее время ответа: {avg_response_time:.2f}с")
        
        return load_test_result

    def run_progressive_load_test(self, rag_system, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает прогрессивное нагрузочное тестирование"""
        print(f"🚀 Запуск прогрессивного нагрузочного тестирования для {engine_name}")
        print("=" * 80)
        
        load_test_results = []
        
        for config in self.load_configs:
            print(f"\n🔧 Тестирование конфигурации: {config['name']}")
            result = self.run_single_load_test(rag_system, config)
            load_test_results.append(result)
            
            # Пауза между тестами
            print("⏸️ Пауза между тестами...")
            time.sleep(10)
        
        # Объединяем результаты
        progressive_result = {
            "engine": engine_name,
            "timestamp": datetime.now().isoformat(),
            "load_tests": load_test_results,
            "summary": self.create_load_test_summary(load_test_results)
        }
        
        return progressive_result

    def create_load_test_summary(self, load_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Создает сводку результатов нагрузочного тестирования"""
        summary = {
            "total_tests": len(load_test_results),
            "max_queries_per_second": 0,
            "max_success_rate": 0,
            "min_avg_response_time": float('inf'),
            "max_avg_response_time": 0,
            "performance_degradation": [],
            "recommendations": []
        }
        
        for result in load_test_results:
            # Максимальная производительность
            if result["queries_per_second"] > summary["max_queries_per_second"]:
                summary["max_queries_per_second"] = result["queries_per_second"]
            
            # Максимальная успешность
            if result["success_rate"] > summary["max_success_rate"]:
                summary["max_success_rate"] = result["success_rate"]
            
            # Время ответа
            if result["avg_response_time"] < summary["min_avg_response_time"]:
                summary["min_avg_response_time"] = result["avg_response_time"]
            
            if result["avg_response_time"] > summary["max_avg_response_time"]:
                summary["max_avg_response_time"] = result["avg_response_time"]
        
        # Анализ деградации производительности
        for i in range(1, len(load_test_results)):
            prev_result = load_test_results[i-1]
            curr_result = load_test_results[i]
            
            if prev_result["queries_per_second"] > 0:
                degradation = (prev_result["queries_per_second"] - curr_result["queries_per_second"]) / prev_result["queries_per_second"]
                summary["performance_degradation"].append({
                    "from": prev_result["test_name"],
                    "to": curr_result["test_name"],
                    "degradation_percent": degradation * 100
                })
        
        # Рекомендации
        if summary["max_success_rate"] < 0.95:
            summary["recommendations"].append("Низкая успешность запросов - рекомендуется оптимизировать систему")
        
        if summary["max_avg_response_time"] > 10:
            summary["recommendations"].append("Высокое время ответа - рекомендуется оптимизировать производительность")
        
        if summary["max_queries_per_second"] < 1:
            summary["recommendations"].append("Низкая пропускная способность - рекомендуется масштабирование")
        
        return summary

    def save_load_test_results(self, results: Dict[str, Any], engine_name: str):
        """Сохраняет результаты нагрузочного тестирования"""
        # JSON файл
        json_file = f"{self.output_dir}/load_test_{engine_name}_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV файл с результатами тестов
        if "load_tests" in results:
            load_df = pd.DataFrame(results["load_tests"])
            csv_file = f"{self.output_dir}/load_test_details_{engine_name}_{self.timestamp}.csv"
            load_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"📁 Результаты нагрузочного тестирования сохранены в {self.output_dir}/")

    def run_full_load_test(self, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает полное нагрузочное тестирование"""
        print(f"🎯 Запуск полного нагрузочного тестирования для {engine_name}")
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
        
        # Запускаем прогрессивное нагрузочное тестирование
        load_test_result = self.run_progressive_load_test(rag_system, engine_name)
        
        # Сохраняем результаты
        self.save_load_test_results(load_test_result, engine_name)
        
        return load_test_result

def main():
    """Основная функция для запуска нагрузочного тестирования"""
    print("⚡ RAG Load Test Benchmark")
    print("=" * 60)
    
    # Проверяем переменные окружения
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        return
    
    # Создаем нагрузочное тестирование
    load_test = LoadTestBenchmark()
    
    # Запускаем нагрузочное тестирование для baseline
    print("🚀 Запуск нагрузочного тестирования для baseline движка...")
    load_test_result = load_test.run_full_load_test("baseline")
    
    if "error" not in load_test_result:
        print("\n📊 Результаты нагрузочного тестирования:")
        summary = load_test_result["summary"]
        print(f"   Максимальная пропускная способность: {summary['max_queries_per_second']:.2f} запросов/сек")
        print(f"   Максимальная успешность: {summary['max_success_rate']:.2%}")
        print(f"   Минимальное время ответа: {summary['min_avg_response_time']:.2f}с")
        print(f"   Максимальное время ответа: {summary['max_avg_response_time']:.2f}с")
        
        if summary["recommendations"]:
            print("\n💡 Рекомендации:")
            for rec in summary["recommendations"]:
                print(f"   • {rec}")
    
    print("\n🎉 Нагрузочное тестирование завершено!")
    print(f"📁 Результаты сохранены в {load_test.output_dir}/")

if __name__ == "__main__":
    main()

