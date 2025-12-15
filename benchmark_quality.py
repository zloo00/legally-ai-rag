#!/usr/bin/env python3
"""
Benchmark для измерения качества ответов RAG системы
Включает метрики релевантности, точности и полноты
"""

import os
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
import re

from rag_system import EnhancedRAGSystem
from rag_factory import EnhancedRAGSystem

rag = EnhancedRAGSystem()

load_dotenv()

class QualityBenchmark:
    """Класс для измерения качества ответов RAG системы"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Тестовые вопросы с ожидаемыми ответами
        self.quality_questions = [
            {
                "question": "Что говорит статья 1 Гражданского кодекса РК?",
                "expected_keywords": ["гражданское право", "отношения", "имущественные", "личные", "неимущественные"],
                "expected_sources": ["Гражданский кодекс", "статья 1"],
                "expected_article_number": "1",
                "expected_code": "ГК"
            },
            {
                "question": "Какие права имеет собственник имущества?",
                "expected_keywords": ["владение", "пользование", "распоряжение", "собственность", "права"],
                "expected_sources": ["Гражданский кодекс", "право собственности"],
                "expected_article_number": None,
                "expected_code": "ГК"
            },
            {
                "question": "Что такое трудовой договор?",
                "expected_keywords": ["трудовой договор", "работник", "работодатель", "трудовые отношения", "соглашение"],
                "expected_sources": ["Трудовой кодекс", "трудовой договор"],
                "expected_article_number": None,
                "expected_code": "ТК"
            },
            {
                "question": "Какие основания для расторжения брака?",
                "expected_keywords": ["расторжение", "брак", "основания", "развод", "суд"],
                "expected_sources": ["Семейный кодекс", "брак"],
                "expected_article_number": None,
                "expected_code": "СК"
            },
            {
                "question": "Что такое наследование по закону?",
                "expected_keywords": ["наследование", "закон", "наследники", "очередь", "имущество"],
                "expected_sources": ["Гражданский кодекс", "наследование"],
                "expected_article_number": None,
                "expected_code": "ГК"
            },
            {
                "question": "Какие виды ответственности предусмотрены в гражданском праве?",
                "expected_keywords": ["ответственность", "гражданское право", "виды", "ущерб", "возмещение"],
                "expected_sources": ["Гражданский кодекс", "ответственность"],
                "expected_article_number": None,
                "expected_code": "ГК"
            },
            {
                "question": "Как защищаются права потребителей?",
                "expected_keywords": ["потребители", "права", "защита", "товары", "услуги"],
                "expected_sources": ["Закон о защите прав потребителей", "потребители"],
                "expected_article_number": None,
                "expected_code": "ЗЗПП"
            },
            {
                "question": "Что такое административная ответственность?",
                "expected_keywords": ["административная", "ответственность", "правонарушение", "штраф", "наказание"],
                "expected_sources": ["Кодекс об административных правонарушениях", "административная ответственность"],
                "expected_article_number": None,
                "expected_code": "КоАП"
            },
            {
                "question": "Какие права имеет работник при увольнении?",
                "expected_keywords": ["работник", "увольнение", "права", "компенсация", "уведомление"],
                "expected_sources": ["Трудовой кодекс", "увольнение"],
                "expected_article_number": None,
                "expected_code": "ТК"
            },
            {
                "question": "Что такое презумпция невиновности?",
                "expected_keywords": ["презумпция", "невиновность", "уголовное право", "обвинение", "доказательства"],
                "expected_sources": ["Уголовный кодекс", "презумпция невиновности"],
                "expected_article_number": None,
                "expected_code": "УК"
            }
        ]

    def calculate_keyword_score(self, answer: str, expected_keywords: List[str]) -> float:
        """Вычисляет оценку по ключевым словам"""
        if not expected_keywords:
            return 0.0
        
        answer_lower = answer.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        return matches / len(expected_keywords)

    def calculate_source_score(self, sources: List[str], expected_sources: List[str]) -> float:
        """Вычисляет оценку по источникам"""
        if not expected_sources:
            return 0.0
        
        source_matches = 0
        for expected_source in expected_sources:
            for source in sources:
                if expected_source.lower() in source.lower():
                    source_matches += 1
                    break
        
        return source_matches / len(expected_sources)

    def calculate_article_number_score(self, answer: str, sources: List[str], expected_article_number: str) -> float:
        """Вычисляет оценку по номеру статьи"""
        if not expected_article_number:
            return 0.0
        
        # Ищем номер статьи в ответе и источниках
        article_pattern = r'статья\s*' + re.escape(expected_article_number) + r'\b'
        
        answer_match = bool(re.search(article_pattern, answer, re.IGNORECASE))
        source_match = any(re.search(article_pattern, source, re.IGNORECASE) for source in sources)
        
        return 1.0 if (answer_match or source_match) else 0.0

    def calculate_code_score(self, sources: List[str], expected_code: str) -> float:
        """Вычисляет оценку по коду закона"""
        if not expected_code:
            return 0.0
        
        code_patterns = [
            expected_code,
            expected_code + " РК",
            expected_code + " Республики Казахстан"
        ]
        
        for source in sources:
            for pattern in code_patterns:
                if pattern.lower() in source.lower():
                    return 1.0
        
        return 0.0

    def calculate_answer_length_score(self, answer: str) -> float:
        """Вычисляет оценку по длине ответа (нормализованную)"""
        # Идеальная длина ответа - 200-500 символов
        ideal_min = 200
        ideal_max = 500
        
        if len(answer) < ideal_min:
            return len(answer) / ideal_min
        elif len(answer) > ideal_max:
            return ideal_max / len(answer)
        else:
            return 1.0

    def calculate_sources_count_score(self, sources_count: int) -> float:
        """Вычисляет оценку по количеству источников"""
        # Идеальное количество источников - 2-5
        ideal_min = 2
        ideal_max = 5
        
        if sources_count < ideal_min:
            return sources_count / ideal_min
        elif sources_count > ideal_max:
            return ideal_max / sources_count
        else:
            return 1.0

    def calculate_relevance_score(self, answer: str, question: str) -> float:
        """Вычисляет оценку релевантности ответа вопросу"""
        # Простая эвристика: проверяем наличие ключевых слов из вопроса в ответе
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Исключаем стоп-слова
        stop_words = {'что', 'какие', 'как', 'где', 'когда', 'почему', 'зачем', 'для', 'чего', 'это', 'такое', 'имеет', 'имеет', 'имеет'}
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words
        
        if not question_words:
            return 0.0
        
        common_words = question_words.intersection(answer_words)
        return len(common_words) / len(question_words)

    def measure_quality_metrics(self, rag_system, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Измеряет все метрики качества для одного вопроса"""
        question = question_data["question"]
        expected_keywords = question_data.get("expected_keywords", [])
        expected_sources = question_data.get("expected_sources", [])
        expected_article_number = question_data.get("expected_article_number")
        expected_code = question_data.get("expected_code")
        
        try:
            result = rag_system.query(question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Вычисляем все метрики
            keyword_score = self.calculate_keyword_score(answer, expected_keywords)
            source_score = self.calculate_source_score(sources, expected_sources)
            article_number_score = self.calculate_article_number_score(answer, sources, expected_article_number)
            code_score = self.calculate_code_score(sources, expected_code)
            answer_length_score = self.calculate_answer_length_score(answer)
            sources_count_score = self.calculate_sources_count_score(len(sources))
            relevance_score = self.calculate_relevance_score(answer, question)
            
            # Общая оценка качества (взвешенная сумма)
            overall_score = (
                keyword_score * 0.25 +
                source_score * 0.20 +
                article_number_score * 0.15 +
                code_score * 0.15 +
                answer_length_score * 0.10 +
                sources_count_score * 0.10 +
                relevance_score * 0.05
            )
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "keyword_score": keyword_score,
                "source_score": source_score,
                "article_number_score": article_number_score,
                "code_score": code_score,
                "answer_length_score": answer_length_score,
                "sources_count_score": sources_count_score,
                "relevance_score": relevance_score,
                "overall_score": overall_score,
                "answer_length": len(answer),
                "sources_count": len(sources)
            }
            
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "keyword_score": 0,
                "source_score": 0,
                "article_number_score": 0,
                "code_score": 0,
                "answer_length_score": 0,
                "sources_count_score": 0,
                "relevance_score": 0,
                "overall_score": 0,
                "answer_length": 0,
                "sources_count": 0
            }

    def run_quality_benchmark(self, rag_system, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает benchmark качества"""
        print(f"🎯 Запуск benchmark качества для {engine_name}")
        print("=" * 60)
        
        quality_results = []
        for i, question_data in enumerate(self.quality_questions, 1):
            print(f"[{i}/{len(self.quality_questions)}] Тестирование: {question_data['question'][:50]}...")
            result = self.measure_quality_metrics(rag_system, question_data)
            quality_results.append(result)
        
        # Вычисляем общую статистику
        keyword_scores = [r["keyword_score"] for r in quality_results if "keyword_score" in r]
        source_scores = [r["source_score"] for r in quality_results if "source_score" in r]
        article_number_scores = [r["article_number_score"] for r in quality_results if "article_number_score" in r]
        code_scores = [r["code_score"] for r in quality_results if "code_score" in r]
        answer_length_scores = [r["answer_length_score"] for r in quality_results if "answer_length_score" in r]
        sources_count_scores = [r["sources_count_score"] for r in quality_results if "sources_count_score" in r]
        relevance_scores = [r["relevance_score"] for r in quality_results if "relevance_score" in r]
        overall_scores = [r["overall_score"] for r in quality_results if "overall_score" in r]
        
        quality_benchmark = {
            "engine": engine_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(self.quality_questions),
            "avg_keyword_score": statistics.mean(keyword_scores) if keyword_scores else 0,
            "avg_source_score": statistics.mean(source_scores) if source_scores else 0,
            "avg_article_number_score": statistics.mean(article_number_scores) if article_number_scores else 0,
            "avg_code_score": statistics.mean(code_scores) if code_scores else 0,
            "avg_answer_length_score": statistics.mean(answer_length_scores) if answer_length_scores else 0,
            "avg_sources_count_score": statistics.mean(sources_count_scores) if sources_count_scores else 0,
            "avg_relevance_score": statistics.mean(relevance_scores) if relevance_scores else 0,
            "avg_overall_score": statistics.mean(overall_scores) if overall_scores else 0,
            "quality_results": quality_results
        }
        
        return quality_benchmark

    def save_quality_results(self, results: Dict[str, Any], engine_name: str):
        """Сохраняет результаты качества в файлы"""
        # JSON файл
        json_file = f"{self.output_dir}/quality_{engine_name}_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV файл с детальными результатами
        if "quality_results" in results:
            qual_df = pd.DataFrame(results["quality_results"])
            csv_file = f"{self.output_dir}/quality_details_{engine_name}_{self.timestamp}.csv"
            qual_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"📁 Результаты качества сохранены в {self.output_dir}/")

    def run_full_quality_benchmark(self, engine_name: str = "baseline") -> Dict[str, Any]:
        """Запускает полный benchmark качества"""
        print(f"🎯 Запуск полного benchmark качества для {engine_name}")
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
        
        # Запускаем benchmark качества
        quality_result = self.run_quality_benchmark(rag_system, engine_name)
        
        # Сохраняем результаты
        self.save_quality_results(quality_result, engine_name)
        
        return quality_result

def main():
    """Основная функция для запуска benchmark качества"""
    print("🎯 RAG Quality Benchmark")
    print("=" * 60)
    
    # Проверяем переменные окружения
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        return
    
    # Создаем benchmark качества
    quality_benchmark = QualityBenchmark()
    
    # Запускаем benchmark для baseline
    print("🚀 Запуск benchmark качества для baseline движка...")
    quality_result = quality_benchmark.run_full_quality_benchmark("baseline")
    
    if "error" not in quality_result:
        print("\n📊 Результаты качества:")
        print(f"   Средняя оценка по ключевым словам: {quality_result['avg_keyword_score']:.3f}")
        print(f"   Средняя оценка по источникам: {quality_result['avg_source_score']:.3f}")
        print(f"   Средняя оценка по номерам статей: {quality_result['avg_article_number_score']:.3f}")
        print(f"   Средняя оценка по кодам законов: {quality_result['avg_code_score']:.3f}")
        print(f"   Средняя оценка по длине ответов: {quality_result['avg_answer_length_score']:.3f}")
        print(f"   Средняя оценка по количеству источников: {quality_result['avg_sources_count_score']:.3f}")
        print(f"   Средняя оценка релевантности: {quality_result['avg_relevance_score']:.3f}")
        print(f"   Общая средняя оценка: {quality_result['avg_overall_score']:.3f}")
    
    print("\n🎉 Benchmark качества завершен!")
    print(f"📁 Результаты сохранены в {quality_benchmark.output_dir}/")

if __name__ == "__main__":
    main()

