import subprocess
import os
import schedule
import time
from datetime import datetime

def exe_japanese_stocks():
    # 日本株スクリプトのみ実行
    conda_env = "py310"
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.04_short.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.04.py")
    ]

    activate_command = f"conda activate {conda_env}"
    for folder, script in scripts_list:
        try:
            os.chdir(folder)
            print(f"Running {script} in {folder}...")
            subprocess.run(f"{activate_command} && python {script}", shell=True, check=True)
            print(f"Finished running {script}.")
        except Exception as e:
            print(f"Error running {script} in {folder}: {e}")

def exe_other_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310"
    scripts_list = [
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.10_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.10_2_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.10.py"),
        ("C:/workspace/cfd_trade", "cfd_america_3top100単位_機械_S_emsemble_today_ver1.07_早く損切_結論_1.5_short.py"),
        ("C:/workspace/cfd_trade", "cfd_america_3top100単位_機械_S_emsemble_today_ver1.07_早く損切.py")
    ]

    activate_command = f"conda activate {conda_env}"
    for folder, script in scripts_list:
        try:
            os.chdir(folder)
            print(f"Running {script} in {folder}...")
            subprocess.run(f"{activate_command} && python {script}", shell=True, check=True)
            print(f"Finished running {script}.")
        except Exception as e:
            print(f"Error running {script} in {folder}: {e}")

# スケジュール設定
schedule.every().day.at("07:30").do(lambda: schedule_job("other"))
schedule.every().day.at("16:15").do(lambda: schedule_job("japanese"))

def schedule_job(script_type):
    today = datetime.today().weekday()  # 0=月曜, 6=日曜
    if today < 5:  # 平日のみ実行
        if script_type == "japanese":
            print("Starting Japanese stock scripts at 16:15...")
            exe_japanese_stocks()
        elif script_type == "other":
            print("Starting other scripts at 07:30...")
            exe_other_scripts()

print("スケジュール開始: 平日 07:30 と 16:15 に実行")

# 無限ループでスケジュールを実行
while True:
    schedule.run_pending()
    time.sleep(30)  # 30秒ごとにチェック