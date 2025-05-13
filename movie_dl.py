
import yt_dlp

import yt_dlp

def my_hook(d):
    # d['status'] は 'downloading' / 'finished' など
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded = d.get('downloaded_bytes', 0)
        percent = downloaded / total * 100 if total else 0
        print(f"\rDownloading... {percent:.1f}% ({downloaded//1024//1024} MiB of {total//1024//1024} MiB)", end='')
    elif d['status'] == 'finished':
        print("\nDownload complete, now merging...")

def download_4k(url: str, output_template: str = '%(title)s.%(ext)s'):
    ydl_opts = {
        'format': 'bestvideo[height<=2160]+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        # ここを [...] で囲んで、ちゃんと関数を渡す
        'progress_hooks': [my_hook],
        'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('使い方: python movie_dl.py "https://www.youtube.com/watch?v=pqUNX16D2Xs&t=14883s"')
        sys.exit(1)
    video_url = sys.argv[1]
    download_4k(video_url)