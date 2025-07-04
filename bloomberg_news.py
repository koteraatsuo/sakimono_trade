# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import cloudscraper
# from bs4 import BeautifulSoup
# from typing import Dict, List

# BASE_URL = "https://www.bloomberg.co.jp/"
# TARGET_SECTIONS = ["日本市況", "米国市況"]

# scraper = cloudscraper.create_scraper(
#     browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
# )

# def fetch_page(url: str) -> BeautifulSoup:
#     resp = scraper.get(url, timeout=10)
#     resp.raise_for_status()
#     return BeautifulSoup(resp.text, "html.parser")

# def find_section_links(soup: BeautifulSoup, targets: List[str]) -> Dict[str, str]:
#     """
#     <a> タグのテキストに targets の文字列を含むものを抽出
#     return: {対象キーワード: URL}
#     """
#     found: Dict[str,str] = {}
#     for a in soup.find_all("a"):
#         text = a.get_text(strip=True)
#         for target in targets:
#             if target in text:
#                 href = a.get("href")
#                 if not href:
#                     continue
#                 # 相対パスなら補完
#                 if href.startswith("/"):
#                     href = BASE_URL.rstrip("/") + href
#                 # 最初に見つかったリンクのみ記録
#                 if target not in found:
#                     found[target] = href
#     return found

# def extract_article_body(soup: BeautifulSoup) -> str:
#     # 通常は <section class="body-copy"> に本文がある想定
#     body = soup.select_one("section.body-copy") or soup.select_one("article")
#     if not body:
#         return ""
#     paragraphs = [p.get_text(strip=True) for p in body.find_all("p")]
#     return "\n\n".join(paragraphs)

# def main():
#     top = fetch_page(BASE_URL)
#     links = find_section_links(top, TARGET_SECTIONS)

#     if not links:
#         print("リンクが見つかりませんでした。")
#         return

#     for section, url in links.items():
#         print(f"\n=== {section} ===\nURL: {url}\n")
#         article = fetch_page(url)
#         body = extract_article_body(article)
#         if body:
#             print(body)
#         else:
#             print("（本文を抽出できませんでした）")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install playwright
# 初回にターミナルで: playwright install

# from playwright.sync_api import sync_playwright

# BASE_URL     = "https://www.bloomberg.co.jp"
# TARGETS      = ["日本市況", "米国市況"]
# MAX_ARTICLES = 10
# SCROLL_STEPS = 15    # スクロール回数
# SCROLL_DELAY = 200   # ミリ秒

# def scroll_page(page, steps: int, delay: int):
#     for _ in range(steps):
#         page.evaluate("window.scrollBy(0, window.innerHeight)")
#         page.wait_for_timeout(delay)

# def get_section_url(page, target_text: str) -> str:
#     # ページ上のすべての <a> を取ってきて、テキストに target_text を含むものを探す
#     for a in page.query_selector_all("a"):
#         text = (a.inner_text() or "").strip()
#         if target_text in text:
#             href = a.get_attribute("href")
#             if href:
#                 return href if href.startswith("http") else BASE_URL + href
#     return ""

# def scrape_section(playwright, target_text: str):
#     browser = playwright.chromium.launch(headless=True)
#     page    = browser.new_page()
#     page.goto(BASE_URL, timeout=30_000)
#     page.wait_for_load_state("networkidle")

#     url = get_section_url(page, target_text)
#     if not url:
#         print(f"⚠️ リンクが見つかりません: {target_text}")
#         browser.close()
#         return []

#     page.goto(url, timeout=30_000)
#     page.wait_for_load_state("networkidle")
#     scroll_page(page, SCROLL_STEPS, SCROLL_DELAY)

#     articles = []
#     seen = set()
#     # ニュース系リンクを幅広く取得
#     for a in page.query_selector_all("a[href*='/news/'], a[href*='/article']"):
#         title = (a.inner_text() or "").strip()
#         href  = a.get_attribute("href") or ""
#         if not title or not href:
#             continue
#         if href.startswith("/"):
#             href = BASE_URL + href
#         if href in seen:
#             continue
#         seen.add(href)
#         articles.append({"title": title, "url": href})
#         if len(articles) >= MAX_ARTICLES:
#             break

#     browser.close()
#     return articles

# def main():
#     with sync_playwright() as p:
#         for target in TARGETS:
#             print(f"\n=== {target} ===")
#             items = scrape_section(p, target)
#             if not items:
#                 print("（記事が取得できませんでした）")
#             for i, art in enumerate(items, start=1):
#                 print(f"{i:2d}. {art['title']}\n    {art['url']}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List

from playwright.sync_api import sync_playwright

# ─── SMTP／送信先設定 ─────────────────────────────────────────
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587
GMAIL_USER     = "k.atsuofxtrade@gmail.com"
GMAIL_PASSWORD = "yyyegokbvfcyufnm"

RECIPIENT_LIST = [
    "kotera2hjp@gmail.com",
    "k.atsuofxtrade@gmail.com",
    "satosato.k543@gmail.com",
    "clubtrdr@gmail.com"
]

# ─── News 抽出設定 ────────────────────────────────────────────
BASE_URL     = "https://www.bloomberg.co.jp"
TARGETS      = ["日本市況", "米国市況"]
MAX_ARTICLES = 10
SCROLL_STEPS = 15
SCROLL_DELAY = 200  # ms

def scroll_page(page):
    for _ in range(SCROLL_STEPS):
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        page.wait_for_timeout(SCROLL_DELAY)

def get_section_url(page, keyword: str) -> str:
    for a in page.query_selector_all("a"):
        text = (a.inner_text() or "").strip()
        if keyword in text:
            href = a.get_attribute("href")
            if href:
                return href if href.startswith("http") else BASE_URL + href
    return ""

def scrape_section(playwright, keyword: str) -> List[Dict[str,str]]:
    browser = playwright.chromium.launch(headless=True)
    page    = browser.new_page()
    page.goto(BASE_URL, timeout=30000)
    page.wait_for_load_state("networkidle")

    url = get_section_url(page, keyword)
    if not url:
        browser.close()
        return []

    page.goto(url, timeout=30000)
    page.wait_for_load_state("networkidle")
    scroll_page(page)

    seen = set()
    results = []
    for a in page.query_selector_all("a[href*='/news/'], a[href*='/article']"):
        title = (a.inner_text() or "").strip()
        href  = a.get_attribute("href") or ""
        if not title or not href:
            continue
        if href.startswith("/"):
            href = BASE_URL + href
        if href in seen:
            continue
        seen.add(href)
        results.append({"title": title, "url": href})
        if len(results) >= MAX_ARTICLES:
            break

    browser.close()
    return results

def build_email_content(all_news: Dict[str,List[Dict[str,str]]]) -> str:
    """
    ニュースをセクションごとに見やすく整形してHTML本文を返す
    """
    lines = ["<h1>Bloomberg 市況ニュース</h1>"]
    for section, items in all_news.items():
        lines.append(f"<h2>{section}</h2><ol>")
        if not items:
            lines.append("<li>取得できませんでした</li>")
        else:
            for it in items:
                lines.append(
                    f'<li><a href="{it["url"]}">{it["title"]}</a></li>'
                )
        lines.append("</ol>")
    return "\n".join(lines)

def send_email(html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Bloomberg 市況ニュース {time.strftime('%Y-%m-%d')}"
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(RECIPIENT_LIST)

    # HTML本文を添付
    part = MIMEText(html_body, "html", "utf-8")
    msg.attach(part)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(GMAIL_USER, GMAIL_PASSWORD)
        smtp.sendmail(GMAIL_USER, RECIPIENT_LIST, msg.as_string())

def main():
    # 1) ニュース取得
    all_news = {}
    with sync_playwright() as p:
        for keyword in TARGETS:
            articles = scrape_section(p, keyword)
            all_news[keyword] = articles

    # 2) メール本文作成
    html_body = build_email_content(all_news)

    # 3) メール送信
    send_email(html_body)
    print("ニュースをメールで送信しました。")

if __name__ == "__main__":
    main()
