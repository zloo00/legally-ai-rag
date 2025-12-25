# Makefile для запуска benchmark'ов RAG системы

.PHONY: help benchmark performance quality load compare clean install demo

# Переменные
PYTHON = python3
BENCHMARK_DIR = benchmark_results

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  benchmark     - Запустить полный benchmark"
	@echo "  performance   - Запустить benchmark производительности"
	@echo "  quality       - Запустить benchmark качества"
	@echo "  load          - Запустить нагрузочное тестирование"
	@echo "  compare       - Сравнить разные движки RAG"
	@echo "  demo          - Демонстрация benchmark'ов"
	@echo "  clean         - Очистить результаты benchmark'ов"
	@echo "  install       - Установить зависимости"

# Полный benchmark
benchmark:
	@echo "🚀 Запуск полного benchmark..."
	$(PYTHON) benchmarks/benchmark_rag.py

# Benchmark производительности
performance:
	@echo "⚡ Запуск benchmark производительности..."
	$(PYTHON) benchmarks/benchmark_rag.py

# Benchmark качества
quality:
	@echo "🎯 Запуск benchmark качества..."
	$(PYTHON) benchmarks/benchmark_quality.py

# Нагрузочное тестирование
load:
	@echo "⚡ Запуск нагрузочного тестирования..."
	$(PYTHON) benchmarks/benchmark_load_test.py

# Сравнение движков
compare:
	@echo "🔄 Сравнение движков RAG..."
	$(PYTHON) benchmarks/benchmark_compare_engines.py

# Очистка результатов
clean:
	@echo "🧹 Очистка результатов benchmark'ов..."
	rm -rf $(BENCHMARK_DIR)
	@echo "✅ Очистка завершена"

# Установка зависимостей
install:
	@echo "📦 Установка зависимостей..."
	pip install -r requirements.txt
	@echo "✅ Зависимости установлены"

# Создание директории для результатов
$(BENCHMARK_DIR):
	mkdir -p $(BENCHMARK_DIR)

# Запуск всех тестов
all: clean install benchmark
	@echo "🎉 Все тесты завершены!"

# Быстрый тест (только производительность)
quick: performance
	@echo "⚡ Быстрый тест завершен!"

# Тест качества
test-quality: quality
	@echo "🎯 Тест качества завершен!"

# Тест нагрузки
test-load: load
	@echo "⚡ Тест нагрузки завершен!"

# Демонстрация
demo:
	@echo "🎯 Демонстрация benchmark'ов..."
	$(PYTHON) benchmarks/demo_benchmark.py
