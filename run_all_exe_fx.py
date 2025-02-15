import subprocess
import os

def exe(): 
# Anaconda 仮想環境名
    conda_env = "py310_fx"

    # 実行するフォルダとスクリプトのリスト
    scripts_list = [
        ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_GBPJPY_損切_ajust_ver1.19_送信.py"),
        ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_EURJPY_損切_ajust_ver1.19_送信.py"),
        ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_USDJPY_損切_ajust_ver1.19_送信.py"),
    ]

    # Anaconda の activate コマンド
    activate_command = f"conda activate {conda_env}"

    # 各スクリプトを順番に実行
    for folder, script in scripts_list:
        try:
            # フォルダへ移動
            os.chdir(folder)
            print(f"Running {script} in {folder}...")

            # 仮想環境を有効化してスクリプトを実行
            subprocess.run(f"{activate_command} && python {script}", shell=True, check=True)
            print(f"Finished running {script}.")
        except Exception as e:
            print(f"Error running {script} in {folder}: {e}")

    print("All scripts executed.")

exe()

