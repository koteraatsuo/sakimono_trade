import subprocess
import os

def exe(): 
# Anaconda 仮想環境名
    conda_env = "py310"

    # 実行するフォルダとスクリプトのリスト
    scripts_list = [
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.10_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_3top100単位_機械_S_emsemble_today_ver1.10.py"),
        ("C:/workspace/cfd_trade", "cfd_america_3top100単位_機械_S_emsemble_today_ver1.07_早く損切_結論_1.5_short.py"),
        ("C:/workspace/cfd_trade", "cfd_america_3top100単位_機械_S_emsemble_today_ver1.07_早く損切.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.04_short.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.04.py")
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

