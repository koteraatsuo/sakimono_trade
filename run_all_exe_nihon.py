import subprocess
import os

def exe(): 
# Anaconda 仮想環境名
    conda_env = "py310_fx"

    # 実行するフォルダとスクリプトのリスト
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06_top30.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06_short.py"),
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

