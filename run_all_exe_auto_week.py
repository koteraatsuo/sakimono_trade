import subprocess
import os
import schedule
import time
from datetime import datetime

def exe_japanese_stocks():
    # 日本株スクリプトのみ実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06_short.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06_top30.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06.py")
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

def exe_sakimono_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.11_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.11_2_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.11.py"),
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



def exe_metal_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.12_silver_open_short.11_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.12_gold_open_short.py"),
        # ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.11.py"),
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


def exe_cocoa_coffee_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.12_cocoa_open_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.12_coffee_open_short.py"),
        # ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.11.py"),
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

def exe_cfd_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    # scripts_list = [
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_v4.py"),
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_top8.py"),
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open.py"),
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_open_v4.py")
    # ]


    scripts_list = [
        ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_v4_lev_2.py"),
        ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_v4.py"),
        ("C:/workspace/cfd_trade", "cfd_america_ver1.10_open_v4_lev_2.py"),
        ("C:/workspace/cfd_trade", "cfd_america_ver1.10_open_v4.py")
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

def exe_fx_scripts():
    # fxスクリプトを実行（今回は土曜日に実行）
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_GBPJPY_損切_ajust_ver1.19_送信.py"),
        ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_EURJPY_損切_ajust_ver1.19_送信.py"),
        ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_USDJPY_損切_ajust_ver1.19_送信.py"),
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


def exe_update_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/sakimono_trade", "auto_git_pull.py"),
        # ("C:/workspace/sakimono_trade", "Sakimono_ver1.12_coffee_open_short.py"),
        # ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.11.py"),
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


def schedule_job(script_type):
    today = datetime.today().weekday()  # 月=0, 火=1, …, 土=5, 日=6
    if script_type == "japanese":
        if today <= 5:  # 月～土
            print("Starting Japanese stock scripts at 16:15...")
            exe_japanese_stocks()
    elif script_type == "cfd":
        if today <= 5:  # 月～土
            print("Starting other scripts at 07:30...")
            exe_cfd_scripts()
    elif script_type == "sakimono":
        if today <= 5:  # 月～土
            print("Starting other scripts at 07:10...")
            exe_sakimono_scripts()
    elif script_type == "metal":
        if today <= 5:  # 月～土
            print("Starting other scripts at 08:02...")
            exe_metal_scripts()
    elif script_type == "cocoa_coffee":
        if today <= 5:  # 月～土
            print("Starting other scripts at 08:02...")
            exe_cocoa_coffee_scripts()
    elif script_type == "fx":
        if today == 5:  # 土曜日
            print("Starting fx scripts at 07:00 on Saturday...")
            exe_fx_scripts()

# スケジュール設定
# 平日（月～金）は07:30に「other」スクリプト、16:15に「japanese」スクリプトを実行
schedule.every().day.at("07:00").do(lambda: schedule_job("fx"))
schedule.every().day.at("09:02").do(lambda: schedule_job("japanese"))
schedule.every().day.at("08:03").do(lambda: schedule_job("metal"))
schedule.every().day.at("19:00").do(lambda: schedule_job("cocoa_coffee"))
schedule.every().day.at("23:31").do(lambda: schedule_job("cfd"))
schedule.every().day.at("13:15").do(lambda: exe_update_scripts())
schedule.every().day.at("01:15").do(lambda: exe_update_scripts())
# 土曜日は07:30にfxスクリプトを実行
schedule.every().saturday.at("07:30").do(lambda: schedule_job("fx"))

print("スケジュール開始: 平日 07:30 と 16:15、土曜日 07:30 に実行")

# 無限ループでスケジュールを実行
while True:
    schedule.run_pending()
    time.sleep(30)  # 30秒ごとにチェック