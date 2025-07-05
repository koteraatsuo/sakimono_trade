#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
マーケット情報レポート送信スクリプト

– X（旧Twitter）アカウント一覧をテーブル形式でHTMLリンク化
– Bloomberg日本市況／米国市況の最新記事をPlaywrightでスクレイピング
– Bloombergトップ5記事（全体）を取得
– YouTubeチャンネルの最新動画をYouTube Data APIで取得
– まとめてGmail経由でHTMLメール送信
"""

import time
import datetime
import smtplib
from typing import Dict, List
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from playwright.sync_api import sync_playwright
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ─── SMTP／送信先設定 ─────────────────────────────────────────
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587
GMAIL_USER     = "k.atsuofxtrade@gmail.com"
GMAIL_PASSWORD = "yyyegokbvfcyufnm"

RECIPIENT_LIST = [
    "kotera2hjp@gmail.com",
    "k.atsuofxtrade@gmail.com",
    "satosato.k543@gmail.com",
    "clubtrdr@gmail.com",
]

# ─── X アカウント表示名→URL設定 ─────────────────────────────────
X_ACCOUNTS: Dict[str,str] = {
    "NOBU塾":            "https://x.com/JACK_ORAN21",
    "池田伸太郎":        "https://x.com/sikeda23",
    "Yuto Haga":          "https://x.com/Yuto_Headline", 
    "Street_Insights":          "https://x.com/Street_Insights", 
    "後藤達也":          "https://x.com/goto_tatsuya",
    "石原順（西山孝四郎）":  "https://x.com/ishiharajun",      # placeholder URL
    "TEAM ハロンズ":          "https://x.com/TeamHallons",
    "Silver hand":          "https://x.com/Silver47samurai",      # placeholder URL
    "亀太郎":          "https://x.com/kame_taro_kabu1",  
    "ゆるまづ":          "https://x.com/yurumazu",
    "にこそくfx":        "https://x.com/nicosokufx",
    "関原 大輔":     "https://x.com/sekihara_d",
    "Bloomberg Japan":   "https://x.com/BloombergJapan",
    "Reuters Japan":     "https://x.com/ReutersJapan",








    # 追加: 日本人マーケット系インフルエンサー
    #  # placeholder URL
    # "佐々木軒":          "https://x.com/sasaki_ken",      # placeholder URL
    # "吉田真弓":          "https://x.com/yoshida_mayumi",  # placeholder URL
    # # 追加: 外資系金融出身マーケット系インフルエンサー
    # "Afzal Hussein":     "https://x.com/Afzal_Hussein_",
    # "Patrick Boyle":     "https://x.com/PatrickEBoyle",
    # "Anton Kreil":       "https://x.com/AntonKreil",
    # "Raoul Pal":         "https://x.com/RaoulGMI",
    # "Gary Black":        "https://x.com/garyblack00",
    # "Economics Ethan":   "https://x.com/EconEthan",
    # "Dan Takahashi":     "https://x.com/Dan_Takahashi_",
    # "Kei Tanaka":        "https://x.com/KeiTanaka_Radio",

}


# ─── Bloomberg市況ニュース設定 ────────────────────────────────────
BASE_URL     = "https://www.bloomberg.co.jp"
TARGETS      = ["日本市況", "米国市況"]
MAX_ARTICLES = 10
TOP_N        = 5
SCROLL_STEPS = 15
SCROLL_DELAY = 200  # ミリ秒

# ─── YouTube設定 ─────────────────────────────────────────────────
YOUTUBE_CHANNELS: Dict[str,str] = {
    "ノブ塾":               "UCOX7X_ddhi1oXurSbJsk45Q",
    "yenzo market":        "UCk-Jlsfh0cIfTTbB6U5xMaA",
    "ばっちゃまの米国株":   "UCzoYzblsE4SEfrQmdjTQZDw",
    # # 追加: 外資系金融出身マーケット系インフルエンサー
    # "Afzal Hussein":       "UCUhoTvkXvwXQ8n935H8l1lg",  # channel ID placeholder
    # "Patrick Boyle":       "UCMYK3QVGKhGCiWJaOsaG9Rg",  # channel ID placeholder
    # "Anton Kreil":         "UCgPf35i6H0GnGkR0CGlvdJQ",  # InstituteOfTrading channel
    # "Real Vision Finance": "UCP5tMZxH6sJA5QsYA1L3FQA",  # Real Vision channel
    # "Economics Ethan":     "UCeOz5YHePhqjzFmiODoy2cA",  # placeholder
    # "Dan Takahashi":       "UCX6b17PVsYBQ0ip5gyeme-Q",  # c/DanTakahashi
}
YOUTUBE_API_KEY = "AIzaSyCQfXzF4Nn3UOux-DE9m4ldDtdNmo6C5jE"

# ─── スクレイピング・ユーティリティ ─────────────────────────────────
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

def scrape_top_articles(playwright) -> List[Dict[str,str]]:
    browser = playwright.chromium.launch(headless=True)
    page    = browser.new_page()
    page.goto(BASE_URL, timeout=30000)
    page.wait_for_load_state("networkidle")
    scroll_page(page)
    seen = set()
    top5 = []
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
        top5.append({"title": title, "url": href})
        if len(top5) >= TOP_N:
            break
    browser.close()
    return top5

# ─── YouTube最新動画取得 ────────────────────────────────────────────
def fetch_latest_video(channel_id: str) -> Dict[str,str]:
    if YOUTUBE_API_KEY.startswith("YOUR_"):
        return {"title": "APIキー未設定のため取得できません。", "url": ""}
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        res = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            order="date",
            maxResults=1,
            type="video"
        ).execute()
        items = res.get("items", [])
        if not items:
            return {"title": "動画が見つかりませんでした。", "url": ""}
        it = items[0]
        title = it["snippet"]["title"]
        pub   = it["snippet"]["publishedAt"]
        vid   = it["id"]["videoId"]
        url   = f"https://youtu.be/{vid}"
        dt    = datetime.datetime.fromisoformat(pub.replace("Z", "+00:00"))
        dt_str= dt.astimezone().strftime("%Y-%m-%d %H:%M")
        return {"title": f"{dt_str} | {title}", "url": url}
    except HttpError as e:
        return {"title": f"YouTube APIエラー: {e}", "url": ""}

# ─── HTMLメール本文作成 ─────────────────────────────────────────
def build_email_content(
    all_news: Dict[str, List[Dict[str,str]]],
    top_articles: List[Dict[str,str]]
) -> str:
    lines: List[str] = []
    lines.append("<html><body>")
    lines.append(f"<h1>マーケット情報レポート {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>")
    # X一覧をテーブルで表示
    lines.append("<h2>X アカウント一覧</h2>")
    lines.append("<table border='1' cellspacing='0' cellpadding='5'><tr><th>名称</th><th>URL</th></tr>")
    for name, url in X_ACCOUNTS.items():
        lines.append(f"<tr><td>{name}</td><td><a href='{url}'>{url}</a></td></tr>")
    lines.append("</table>")
    # Bloombergニュース
    lines.append("<h2>Bloomberg 市況ニュース</h2>")

    lines.append("<h3>トップ5記事（全体）</h3>")
    lines.append("<table border='1' cellspacing='0' cellpadding='5'><tr><th>記事タイトル</th><th>URL</th></tr>")
    for ta in top_articles:
        lines.append(f"<tr><td>{ta['title']}</td><td><a href='{ta['url']}'>{ta['url']}</a></td></tr>")
    lines.append("</table>")
    for section, items in all_news.items():
        lines.append(f"<h3>{section}</h3>")
        lines.append("<table border='1' cellspacing='0' cellpadding='5'><tr><th>記事タイトル</th><th>URL</th></tr>")
        if items:
            for it in reversed(items):
                lines.append(f"<tr><td>{it['title']}</td><td><a href='{it['url']}'>{it['url']}</a></td></tr>")
        else:
            lines.append("<tr><td colspan='2'>取得できませんでした</td></tr>")
        lines.append("</table>")
        # トップ5記事（全体）

    # YouTube最新動画
    lines.append("<h2>YouTube 最新動画</h2>")
    lines.append("<table border='1' cellspacing='0' cellpadding='5'><tr><th>チャンネル</th><th>動画</th></tr>")
    for name, cid in YOUTUBE_CHANNELS.items():
        vid = fetch_latest_video(cid)
        if vid['url']:
            lines.append(f"<tr><td>{name}</td><td><a href='{vid['url']}'>{vid['title']}</a></td></tr>")
        else:
            lines.append(f"<tr><td>{name}</td><td>{vid['title']}</td></tr>")
    lines.append("</table>")
    lines.append("</body></html>")
    return "\n".join(lines)

# ─── メール送信 ────────────────────────────────────────────────
def send_email(html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"マーケット情報レポート {time.strftime('%Y-%m-%d')}"
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(RECIPIENT_LIST)
    part = MIMEText(html_body, "html", "utf-8")
    msg.attach(part)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(GMAIL_USER, GMAIL_PASSWORD)
        smtp.sendmail(GMAIL_USER, RECIPIENT_LIST, msg.as_string())

# ─── メイン処理 ────────────────────────────────────────────────
def main():
    all_news: Dict[str, List[Dict[str,str]]] = {}
    with sync_playwright() as p:
        for kw in TARGETS:
            all_news[kw] = scrape_section(p, kw)
        top_articles = scrape_top_articles(p)
    html_body = build_email_content(all_news, top_articles)
    send_email(html_body)
    print("メールを送信しました。")

if __name__ == "__main__":
    main()














#         lines.append("</ol>")
#     lines.append("<h4>トップ5記事（全体）</h4><ol>")
#     for ta in top_articles:
#         lines.append(f'<li><a href="{ta["url"]}">{ta["title"]}</a></li>')
#     lines.append("</ol>")

#     # YouTube 最新動画
#     lines.append("<h2>YouTube 最新動画</h2><ul>")
#     for name, cid in YOUTUBE_CHANNELS.items():
#         vid = fetch_latest_video(cid)
#         if vid["url"]:
#             lines.append(f'<li><strong>{name}</strong>：<a href="{vid["url"]}">{vid["title"]}</a></li>')
#         else:
#             lines.append(f'<li><strong>{name}</strong>：{vid["title"]}</li>')
#     lines.append("</ul>")

#     lines.append("</body></html>")
#     return "\n".join(lines)

# # ─── メール送信 ────────────────────────────────────────────────
# def send_email(html_body: str):
#     msg = MIMEMultipart("alternative")
#     msg["Subject"] = f"マーケット情報レポート {time.strftime('%Y-%m-%d')}"
#     msg["From"]    = GMAIL_USER
#     msg["To"]      = ", ".join(RECIPIENT_LIST)

#     part = MIMEText(html_body, "html", "utf-8")
#     msg.attach(part)

#     with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
#         smtp.ehlo()
#         smtp.starttls()
#         smtp.login(GMAIL_USER, GMAIL_PASSWORD)
#         smtp.sendmail(GMAIL_USER, RECIPIENT_LIST, msg.as_string())

# # ─── メイン処理 ────────────────────────────────────────────────
# def main():
#     all_news: Dict[str, List[Dict[str,str]]] = {}
#     with sync_playwright() as p:
#         for kw in TARGETS:
#             all_news[kw] = scrape_section(p, kw)
#         top_articles = scrape_top_articles(p)

#     html_body = build_email_content(all_news, top_articles)
#     send_email(html_body)
#     print("メールを送信しました。")

# if __name__ == "__main__":
#     main()
