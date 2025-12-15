#!/usr/bin/env python3
"""
Сравнение разных RAG движков (baseline, GraphRAG, LightRAG)
"""

import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from benchmark_rag import RAGBenchmark

load_dotenv()

class EngineComparison:
    """Класс для сравнения разных RAG движков"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark = RAGBenchmark(output_dir)
        
        # Тестовые вопросы для сравнения
        self.comparison_questions = [
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

    def compare_engines_performance(self, engines: List[str]) -> Dict[str, Any]:
        """Сравнивает производительность разных движков"""
        print("🚀 Сравнение производительности движков")
        print("=" * 60)
        
        results = {}
        
        for engine in engines:
            print(f"\n🔧 Тестирование {engine}...")
            try:
                # Запускаем benchmark для каждого движка
                result = self.benchmark.run_full_benchmark(engine)
                results[engine] = result
            except Exception as e:
                print(f"❌ Ошибка при тестировании {engine}: {e}")
                results[engine] = {"error": str(e)}
        
        return results

    def compare_engines_quality(self, engines: List[str]) -> Dict[str, Any]:
        """Сравнивает качество ответов разных движков"""
        print("🎯 Сравнение качества движков")
        print("=" * 60)
        
        quality_results = {}
        
        for engine in engines:
            print(f"\n🔧 Тестирование качества {engine}...")
            try:
                # Инициализируем RAG систему
                if engine == "baseline":
                    from rag_system import EnhancedRAGSystem
                    rag_system = EnhancedRAGSystem()
                else:
                    from rag_factory import RAGFactory
                    rag_system = RAGFactory.create_rag_system(engine)
                
                # Тестируем качество
                quality_result = self.benchmark.run_quality_benchmark(rag_system, engine)
                quality_results[engine] = quality_result
                
            except Exception as e:
                print(f"❌ Ошибка при тестировании качества {engine}: {e}")
                quality_results[engine] = {"error": str(e)}
        
        return quality_results

    def compare_engines_load(self, engines: List[str]) -> Dict[str, Any]:
        """Сравнивает нагрузочные характеристики разных движков"""
        print("⚡ Сравнение нагрузочных характеристик движков")
        print("=" * 60)
        
        load_results = {}
        
        for engine in engines:
            print(f"\n🔧 Тестирование нагрузки {engine}...")
            try:
                # Инициализируем RAG систему
                if engine == "baseline":
                    from rag_system import EnhancedRAGSystem
                    rag_system = EnhancedRAGSystem()
                else:
                    from rag_factory import RAGFactory
                    rag_system = RAGFactory.create_rag_system(engine)
                
                # Тестируем нагрузку
                load_result = self.benchmark.run_load_test(rag_system)
                load_results[engine] = load_result
                
            except Exception as e:
                print(f"❌ Ошибка при тестировании нагрузки {engine}: {e}")
                load_results[engine] = {"error": str(e)}
        
        return load_results

    def create_comparison_report(self, performance_results: Dict[str, Any], 
                               quality_results: Dict[str, Any], 
                               load_results: Dict[str, Any]) -> pd.DataFrame:
        """Создает сводный отчет сравнения движков"""
        print("📊 Создание сводного отчета")
        print("=" * 60)
        
        comparison_data = []
        
        for engine in performance_results.keys():
            if engine in performance_results and "error" not in performance_results[engine]:
                perf = performance_results[engine].get("performance", {})
                qual = quality_results.get(engine, {}).get("quality", {})
                load = load_results.get(engine, {}).get("load_test", {})
                
                comparison_data.append({
                    "Engine": engine,
                    "Avg Query Time (s)": perf.get("avg_query_time", 0),
                    "Min Query Time (s)": perf.get("min_query_time", 0),
                    "Max Query Time (s)": perf.get("max_query_time", 0),
                    "Success Rate": perf.get("avg_success_rate", 0),
                    "Keyword Score": qual.get("avg_keyword_score", 0),
                    "Source Score": qual.get("avg_source_score", 0),
                    "Avg Answer Length": qual.get("avg_answer_length", 0),
                    "Avg Sources Count": qual.get("avg_sources_count", 0),
                    "Queries/Second": load.get("queries_per_second", 0),
                    "Load Success Rate": load.get("success_rate", 0),
                    "Total Vectors": perf.get("system_stats", {}).get("total_vectors", 0),
                    "Index Dimension": perf.get("system_stats", {}).get("index_dimension", 0)
                })
            else:
                comparison_data.append({
                    "Engine": engine,
                    "Avg Query Time (s)": "ERROR",
                    "Min Query Time (s)": "ERROR",
                    "Max Query Time (s)": "ERROR",
                    "Success Rate": "ERROR",
                    "Keyword Score": "ERROR",
                    "Source Score": "ERROR",
                    "Avg Answer Length": "ERROR",
                    "Avg Sources Count": "ERROR",
                    "Queries/Second": "ERROR",
                    "Load Success Rate": "ERROR",
                    "Total Vectors": "ERROR",
                    "Index Dimension": "ERROR"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Сохраняем отчет
        report_file = f"{self.output_dir}/engine_comparison_{self.timestamp}.csv"
        comparison_df.to_csv(report_file, index=False, encoding='utf-8')
        
        print("\n📊 Сводная таблица сравнения движков:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df

    def run_full_comparison(self, engines: List[str] = ["baseline"]) -> Dict[str, Any]:
        """Запускает полное сравнение движков"""
        print("🎯 Запуск полного сравнения движков")
        print("=" * 80)
        
        # Сравниваем производительность
        performance_results = self.compare_engines_performance(engines)
        
        # Сравниваем качество
        quality_results = self.compare_engines_quality(engines)
        
        # Сравниваем нагрузочные характеристики
        load_results = self.compare_engines_load(engines)
        
        # Создаем сводный отчет
        comparison_report = self.create_comparison_report(performance_results, quality_results, load_results)
        
        # Объединяем все результаты
        full_comparison = {
            "timestamp": datetime.now().isoformat(),
            "engines": engines,
            "performance": performance_results,
            "quality": quality_results,
            "load": load_results,
            "comparison_report": comparison_report.to_dict('records')
        }
        
        # Сохраняем полные результаты
        json_file = f"{self.output_dir}/full_comparison_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_comparison, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 Полные результаты сохранены в {json_file}")
        
        return full_comparison

    def analyze_results(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует результаты сравнения и выдает рекомендации"""
        print("🔍 Анализ результатов сравнения")
        print("=" * 60)
        
        analysis = {
            "best_performance": None,
            "best_quality": None,
            "best_load": None,
            "recommendations": []
        }
        
        # Анализируем производительность
        perf_scores = {}
        for engine, results in comparison_results.get("performance", {}).items():
            if "error" not in results:
                perf_score = results.get("performance", {}).get("avg_query_time", float('inf'))
                perf_scores[engine] = perf_score
        
        if perf_scores:
            best_perf_engine = min(perf_scores, key=perf_scores.get)
            analysis["best_performance"] = best_perf_engine
            analysis["recommendations"].append(f"Лучшая производительность: {best_perf_engine}")
        
        # Анализируем качество
        qual_scores = {}
        for engine, results in comparison_results.get("quality", {}).items():
            if "error" not in results:
                qual_score = results.get("quality", {}).get("avg_keyword_score", 0)
                qual_scores[engine] = qual_score
        
        if qual_scores:
            best_qual_engine = max(qual_scores, key=qual_scores.get)
            analysis["best_quality"] = best_qual_engine
            analysis["recommendations"].append(f"Лучшее качество: {best_qual_engine}")
        
        # Анализируем нагрузочные характеристики
        load_scores = {}
        for engine, results in comparison_results.get("load", {}).items():
            if "error" not in results:
                load_score = results.get("load_test", {}).get("queries_per_second", 0)
                load_scores[engine] = load_score
        
        if load_scores:
            best_load_engine = max(load_scores, key=load_scores.get)
            analysis["best_load"] = best_load_engine
            analysis["recommendations"].append(f"Лучшая нагрузочная производительность: {best_load_engine}")
        
        # Общие рекомендации
        if len(analysis["recommendations"]) > 0:
            analysis["recommendations"].append("Рекомендуется использовать движок с лучшим балансом производительности и качества")
        
        print("\n📋 Анализ и рекомендации:")
        for rec in analysis["recommendations"]:
            print(f"   • {rec}")
        
        return analysis

def main():
    """Основная функция для запуска сравнения движков"""
    print("🔄 RAG Engine Comparison")
    print("=" * 60)
    
    # Проверяем переменные окружения
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        return
    
    # Создаем сравнение
    comparison = EngineComparison()
    
    # Доступные движки
    available_engines = ["baseline"]  # Можно добавить "graphrag", "lightrag"
    
    # Запускаем сравнение
    print(f"🚀 Запуск сравнения движков: {', '.join(available_engines)}")
    comparison_results = comparison.run_full_comparison(available_engines)
    
    # Анализируем результаты
    analysis = comparison.analyze_results(comparison_results)
    
    print("\n🎉 Сравнение движков завершено!")
    print(f"📁 Результаты сохранены в {comparison.output_dir}/")

if __name__ == "__main__":
    main()

