#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import csv
import yt_dlp

def my_hook(d):
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded = d.get('downloaded_bytes', 0)
        percent = downloaded / total * 100 if total else 0
        print(f"\rDownloading... {percent:.1f}% ({downloaded//1024//1024} MiB/{(total//1024//1024) if total else '?'} MiB)", end='')
    elif d['status'] == 'finished':
        print("\nMerge complete.")

def download_4k(url: str):
    print(f"\n=== Downloading: {url} ===")
    ydl_opts = {
        'format': 'bestvideo[height<=2160]+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': '%(title)s.%(ext)s',
        'progress_hooks': [my_hook],
        'nocheckcertificate': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def load_urls_from_csv(path: str):
    urls = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                urls.append(row[0].strip())
    return urls

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("使い方:")
        print("  単一URL → python movie_dl.py \"https://...\"")
        print("  CSV一括 → python movie_dl.py movie_dl.csv")
        sys.exit(1)

    arg = sys.argv[1]

    # デバッグ出力
    print(f"DEBUG: cwd = {os.getcwd()}")
    print(f"DEBUG: arg  = {arg!r}")
    print(f"DEBUG: isfile(arg) = {os.path.isfile(arg)}")

    # CSVファイルなら読み込み、そうでなければURLとして扱う
    if os.path.isfile(arg) and arg.lower().endswith('.csv'):
        print("DEBUG: CSVルートに入りました")
        urls = load_urls_from_csv(arg)
        if not urls:
            print(f"CSV ({arg}) 内に URL が見つかりません。")
            sys.exit(1)
    else:
        print("DEBUG: URLルートに入りました")
        urls = [arg]

    for url in urls:
        try:
            download_4k(url)
        except Exception as e:
            print(f"⚠️ Failed to download {url}: {e}")
