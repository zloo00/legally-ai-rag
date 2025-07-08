#!/usr/bin/env python3
"""
Тестовый скрипт для проверки веб-интерфейса юридического чата
"""

import requests
import json
import time
from bs4 import BeautifulSoup
import pandas as pd

def test_web_chat():
    """Тестирование веб-интерфейса"""
    base_url = "http://localhost:5001"
    
    # Тестовые вопросы
    test_questions = [
        "Что такое гражданская дееспособность?",
        "Какие права имеет собственник имущества?",
        "Как заключается трудовой договор?",
        "Что такое административная ответственность?"
    ]
    
    print("🧪 Тестирование веб-интерфейса юридического чата")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Тест {i}: {question}")
        print("-" * 40)
        
        try:
            # Отправляем запрос
            response = requests.post(
                f"{base_url}/chat",
                json={"message": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Ответ получен")
                print(f"🤖 Режим: {data.get('mode', 'N/A')}")
                print(f"📊 Найдено статей: {data.get('results_count', 0)}")
                print(f"📝 Размер контекста: {data.get('context_length', 0)} символов")
                
                # Показываем ответ
                answer = data.get('answer', '')
                print(f"\n💬 Ответ:\n{answer[:300]}{'...' if len(answer) > 300 else ''}")
                
                # Показываем найденные статьи
                search_results = data.get('search_results', [])
                if search_results:
                    print(f"\n📚 Найденные статьи ({len(search_results)}):")
                    for j, result in enumerate(search_results[:3], 1):  # Показываем первые 3
                        print(f"  {j}. {result.get('source', 'N/A')} (релевантность: {(result.get('score', 0) * 100):.1f}%)")
                        content = result.get('text', '')
                        print(f"     {content[:100]}{'...' if len(content) > 100 else ''}")
                
                print(f"\n🔗 Источники: {', '.join(data.get('sources', []))}")
                
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"📄 Ответ: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Не удалось подключиться к серверу. Убедитесь, что сервер запущен на http://localhost:5001")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        time.sleep(1)  # Небольшая пауза между запросами
    
    print("\n" + "=" * 60)
    print("✅ Тестирование завершено!")
    print(f"🌐 Веб-интерфейс доступен по адресу: {base_url}")

def get_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text

def parse_question_block(block):
    # Примерная структура, нужно уточнить по реальному HTML
    question = block.find("div", class_="question-text").get_text(strip=True)
    answer = block.find("div", class_="answer-text").get_text(strip=True)
    # Если есть "Читать дальше" — парсим ссылку
    more_link = block.find("a", string="Читать дальше")
    if more_link:
        answer = get_full_answer(BASE_URL + more_link["href"])
    return question, answer

def get_full_answer(url):
    html = get_page(url)
    soup = BeautifulSoup(html, "html.parser")
    answer = soup.find("div", class_="full-answer").get_text(strip=True)
    return answer

def parse_page(page_num):
    url = f"{BASE_URL}?page={page_num}"
    html = get_page(url)
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("div", class_="question-block")  # уточнить класс
    data = []
    for block in blocks:
        question, answer = parse_question_block(block)
        data.append({"question": question, "answer": answer})
    return data

# Пример: собрать первые 10 страниц
all_data = []
for page in range(1, 11):
    print(f"Парсим страницу {page}")
    all_data.extend(parse_page(page))
    time.sleep(1)  # чтобы не забанили

df = pd.DataFrame(all_data)
df.to_excel("jurhelp_questions.xlsx", index=False)

if __name__ == "__main__":
    test_web_chat() 