import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import time
from tqdm import tqdm

BASE_URL = "https://jurhelp.prg.kz/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_question_links(page_num):
    url = BASE_URL if page_num == 1 else f"{BASE_URL}?page={page_num}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a['href']
        if isinstance(href, list):
            href = href[0] if href else ""
        if not isinstance(href, str):
            continue
        if href.startswith("/question/") and len(href) > 20:
            links.add("https://jurhelp.prg.kz" + href)
    return list(links)


def parse_question_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    card = soup.find("div", class_="question-card")
    if not card or not isinstance(card, Tag):
        return None
    try:
        # ID и дата
        badges = [b for b in card.find_all("div", class_="prg-kit-badge") if isinstance(b, Tag)]
        date = badges[0].get_text(strip=True) if len(badges) > 0 else ""
        qid = badges[1].get_text(strip=True).replace("ID ", "") if len(badges) > 1 else ""
        # Пользователь
        user_tag = card.find("b", class_="user")
        user = user_tag.get_text(strip=True) if isinstance(user_tag, Tag) else ""
        # Вопрос
        question_tag = card.find("h1", class_="question-text")
        question = question_tag.get_text(strip=True) if isinstance(question_tag, Tag) else ""
        # Категория
        category_tag = card.find("div", class_="categories")
        category = category_tag.get_text(strip=True) if isinstance(category_tag, Tag) else ""
        # Эксперт
        expert_tag = card.find("div", class_="expert-name")
        expert = ""
        if isinstance(expert_tag, Tag):
            b_tag = expert_tag.find("b")
            expert = b_tag.get_text(strip=True) if isinstance(b_tag, Tag) else ""
        # Ответ
        answer_tag = card.find("div", class_="answer-text")
        answer = answer_tag.get_text(separator="\n", strip=True) if isinstance(answer_tag, Tag) else ""
        return {
            "ID": qid,
            "Дата": date,
            "Пользователь": user,
            "Вопрос": question,
            "Категория": category,
            "Эксперт": expert,
            "Ответ": answer
        }
    except Exception as e:
        print(f"Ошибка парсинга {url}: {e}")
        return None


def main():
    all_data = []
    page = 1
    max_questions = 1000
    print("Сбор ссылок на вопросы...")
    links = set()
    while len(links) < max_questions:
        new_links = get_question_links(page)
        if not new_links:
            break
        links.update(new_links)
        print(f"Страница {page}: найдено {len(new_links)} ссылок, всего {len(links)}")
        page += 1
        time.sleep(1)
    print(f"Всего собрано ссылок: {len(links)}")
    print("Парсинг страниц вопросов...")
    for i, url in enumerate(tqdm(list(links)[:max_questions], desc="Вопросы")):
        data = parse_question_page(url)
        if data:
            all_data.append(data)
        time.sleep(0.5)
    df = pd.DataFrame(all_data)
    df.to_excel("jurhelp_questions.xlsx", index=False)
    print("Готово! Данные сохранены в jurhelp_questions.xlsx")

if __name__ == "__main__":
    main() 