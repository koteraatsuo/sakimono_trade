import os
import subprocess

def git_pull_in_folders(base_dir):
    for root, dirs, files in os.walk(base_dir):
        # .git フォルダがある場合、そのディレクトリで git pull を実行
        if '.git' in dirs:
            try:
                print(f"Running 'git pull' in: {root}")
                result = subprocess.run(['git', '-C', root, 'pull'], capture_output=True, text=True, check=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error in {root}: {e.stderr}")
            except Exception as e:
                print(f"Unexpected error in {root}: {e}")

# ベースディレクトリを指定（画像のフォルダを例として使用）
base_directory = "C:/workspace"

# Git Pull を実行
git_pull_in_folders(base_directory)