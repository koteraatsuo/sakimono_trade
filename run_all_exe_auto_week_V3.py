import subprocess
import os
import schedule
import time
from datetime import datetime

def exe_japanese_stocks():
    # 日本株スクリプトのみ実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_load_sim_ansemble5_bigcompany_top40_1day.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_load_sim_ansemble5_bigcompany_top40_2day.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_load_sim_ansemble5_bigcompany_top40.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_load_sim_ansemble5_bigcompany.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_load_sim_ansemble_bigcompany.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_load_sim_ansemble_bigcompany_top70.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.14_open_ETF_load.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim_bigcompany.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim_nomalcompany.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim_bigcompany_top60.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim_nomalcompany_top60.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim_bigcompany_top70.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim_nomalcompany_top70.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.08_top30_load_sim_Fernandes.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.07_top30_load_sim_Fernandes.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_load_sim.py"),
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
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.14_open_commodity_load.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.14_open_kasoutuka_load.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.14_open_commodity_before.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.14_open_kasoutuka_before.py"),     
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



def exe_japanese_before_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_before_sim_ansemble5_bigcompany.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_before_sim_ansemble5_bigcompany_top40_1day.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v5_before_sim_ansemble5_bigcompany_top40_2day.py"),
        ("C:/workspace/nihon_kabu_trade", "nihon_refresh_list.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.14_open_ETF_before.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_3top100単位_機械_S_emsemble_today_ver1.06_short.py"),
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
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.13_open_Indices.py"),
        ("C:/workspace/sakimono_trade", "CFD_indice_ver1.13_Fernandes_lev2.py"),
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



def exe_before_cfd_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    # scripts_list = [
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_v4.py"),
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_top8.py"),
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open.py"),
    #     ("C:/workspace/cfd_trade", "cfd_america_ver1.10_open_v4.py")
    # ]


    scripts_list = [
        ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v5_before_sim_ansemble5_bigcompany.py"),
        ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v5_before_sim_ansemble5_bigcompany_top40_1day.py"),
        ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v5_before_sim_ansemble5_bigcompany_top40_2day.py"),
        # ("C:/workspace/cfd_trade", "cfd_america_ver1.10_open_v4_lev_2.py"),
        # ("C:/workspace/cfd_trade", "cfd_america_ver1.10_open_v4.py")
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
        # ("C:/workspace/cfd_trade", "cfd_america_ver1.10_short_open_v4_lev_2.py"),
        ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v5_load_sim_ansemble5_bigcompany_top40.py"),
        ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v5_load_sim_ansemble5_bigcompany.py.py"),
        # ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v4_load_sim_nomalcompany.py"),
        # ("C:/workspace/cfd_trade", "america_ver1.12_open_v5_load_sim.py"),
        # ("C:/workspace/cfd_trade", "cfd_america_ver1.11_Fernandes_lev5.py"),
        # # ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v4_load_sim_bigcompany.py"),
        # # ("C:/workspace/cfd_trade", "nihon_ver1.12_open_v4_load_sim_nomalcompany.py"),
        # ("C:/workspace/cfd_trade", "america_ver1.12_open_v4_load_sim.py")
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


def exe_sendmail_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_data_send_topix_2_convert_chart_2.py"),
        ("C:/workspace/cfd_trade", "america_data_send.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_before_train_4000.py"),
        # ("C:/workspace/cfd_trade", "america_ver1.12_open_v4_before_train_4000.py"),
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

def exe_every_6hour_sendmail_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        # ("C:/workspace/sakimono_trade", "bloomberg_news.py"),
        ("C:/workspace/sakimono_trade", "sns_news.py"),
        # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_before_train_4000.py"),
        # ("C:/workspace/cfd_trade", "america_ver1.12_open_v4_before_train_4000.py"),
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



def exe_fx_scripts():
    # fxスクリプトを実行（今回は土曜日に実行）
    conda_env = "py310_fx"
    scripts_list = [
        # ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_GBPJPY_損切_ajust_ver1.19_送信.py"),
        # ("C:/workspace/fx_trade_3", "simulation_W1_M5_long_EURJPY_損切_ajust_ver1.19_送信.py"),
        ("C:/workspace/fx_trade_3", "FX_straight_yen_ver1.12_Fernandes_lev2.py"),
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
        # ("C:/workspace/sakimono_trade", "Sakimono_ver1.12_silver_open_short.11_short.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.13_open_Metals.py"),
        ("C:/workspace/sakimono_trade", "Sakimono_ver1.13_open_gold.py"),
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


def exe_refresh_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_fx"
    scripts_list = [
        ("C:/workspace/nihon_kabu_trade", "nihon_refresh_list.py"),
        ("C:/workspace/cfd_trade", "nihon_refresh_listpy"),
        ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_before_train_4000.py"),
        ("C:/workspace/cfd_trade", "america_ver1.12_open_v4_before_train_4000.py"),
        ("C:/workspace/cfd_trade", "sp500_csv.py"),    
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


def exe_refresh_fx_scripts():
    # 日本株以外のスクリプトを実行
    conda_env = "py310_deeplearning"
    scripts_list = [
        ("C:/workspace/fx_deepleaning", "alpha_zero_fx_USDJPY_before_train_V70_term8_FP16_long_v2_auto_refresh.py"),
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


# def exe_sendmail_scripts():
#     # 日本株以外のスクリプトを実行
#     conda_env = "py310_fx"
#     scripts_list = [
#         ("C:/workspace/nihon_kabu_trade", "nihon_data_send_topix_2_convert_chart.py"),
#         ("C:/workspace/cfd_trade", "america_data_send.py"),
#         # ("C:/workspace/nihon_kabu_trade", "nihon_ver1.12_open_v4_before_train_4000.py"),
#         # ("C:/workspace/cfd_trade", "america_ver1.12_open_v4_before_train_4000.py"),
#     ]

#     activate_command = f"conda activate {conda_env}"
#     for folder, script in scripts_list:
#         try:
#             os.chdir(folder)
#             print(f"Running {script} in {folder}...")
#             subprocess.run(f"{activate_command} && python {script}", shell=True, check=True)
#             print(f"Finished running {script}.")
#         except Exception as e:
#             print(f"Error running {script} in {folder}: {e}")


def schedule_job(script_type):
    # 秒を 5 秒に合わせる（必ず 0～59 の範囲に収まる）
    now = datetime.now()
    delay = (5 - now.second) % 60
    if delay:
        time.sleep(delay)

    today = datetime.today().weekday()  # 月=0, 火=1, …, 土=5, 日=6

    if script_type == "japanese":
        if today <= 5:
            print("Starting Japanese stock scripts at 16:15...")
            exe_japanese_stocks()

    elif script_type == "cfd":
        if today <= 5:
            print("Starting other scripts at 07:30...")
            exe_cfd_scripts()

    elif script_type == "before_cfd":
        if today <= 5:
            print("Starting other scripts at 07:30...")
            exe_before_cfd_scripts()

    elif script_type == "sakimono":
        if today <= 5:
            print("Starting other scripts at 07:10...")
            exe_sakimono_scripts()

    elif script_type == "japanese_before":
        if today <= 5:
            print("Starting other scripts at 08:02...")
            exe_japanese_before_scripts()

    elif script_type == "cocoa_coffee":
        if today <= 5:
            print("Starting other scripts at 08:02...")
            exe_cocoa_coffee_scripts()

    elif script_type == "fx":
        if today < 5:
            print("Starting fx scripts at 07:00 on Saturday...")
            exe_fx_scripts()
    elif script_type == "send_mail":
        if today == 5:
            print("Starting fx scripts at 07:00 on Saturday...")
            exe_sendmail_scripts()
    elif script_type == "6hour_send_mail":
        exe_every_6hour_sendmail_scripts()

    elif script_type == "refresh":
        if today == 5 or today == 6:
            print("Starting refresh scripts at 07:00 on Saturday...")
            exe_refresh_scripts()

    elif script_type == "refresh_fx":
        if today == 5 or today == 6:
            print("Starting refresh scripts at 07:00 on Saturday...")
            exe_refresh_fx_scripts()

    elif script_type == "metal":
        if today <= 5:
            print("Starting other scripts at 08:02...")
            exe_metal_scripts()



# スケジュール設定
# 平日（月～金）は07:30に「other」スクリプト、16:15に「japanese」スクリプトを実行
# schedule.every().day.at("07:00").do(lambda: schedule_job("fx"))

schedule.every().day.at("05:00").do(lambda: schedule_job("japanese_before"))
schedule.every().day.at("07:00").do(lambda: schedule_job("fx"))
schedule.every().day.at("09:00").do(lambda: schedule_job("japanese"))
schedule.every().day.at("08:00").do(lambda: schedule_job("metal"))
schedule.every().day.at("08:01").do(lambda: schedule_job("sakimono"))
schedule.every().day.at("22:31").do(lambda: schedule_job("cocoa_coffee"))
schedule.every().day.at("18:30").do(lambda: schedule_job("before_cfd"))
schedule.every().day.at("22:30").do(lambda: schedule_job("cfd"))
schedule.every().day.at("06:30").do(lambda: schedule_job("send_mail"))
schedule.every().day.at("06:30").do(lambda: schedule_job("6hour_send_mail"))
schedule.every().day.at("12:30").do(lambda: schedule_job("6hour_send_mail"))
schedule.every().day.at("18:30").do(lambda: schedule_job("6hour_send_mail"))
schedule.every().day.at("01:30").do(lambda: schedule_job("6hour_send_mail"))
schedule.every().day.at("07:00").do(lambda: schedule_job("refresh_fx"))
schedule.every().day.at("10:00").do(lambda: schedule_job("refresh"))
schedule.every().day.at("13:15").do(lambda: exe_update_scripts())
schedule.every().day.at("01:15").do(lambda: exe_update_scripts())
# 土曜日は07:30にfxスクリプトを実行
schedule.every().saturday.at("07:30").do(lambda: schedule_job("fx"))

print("スケジュール開始: 平日 07:00 と  08:00 と 09:00 と 19:00 と 22:30、土曜日 07:30 に実行")

# 無限ループでスケジュールを実行
while True:
    schedule.run_pending()
    time.sleep(30)  # 30秒ごとにチェック