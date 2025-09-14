import customtkinter as ctk
from tkinter import messagebox
import threading
import re
import os
import time

# --- ライブラリのインポート ---
try:
    import pygame
    pygame.mixer.init()
except ImportError:
    pygame = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# --- CustomTkinterの基本設定 ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TextToSpeechApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Modern TTS App (改行無視・安定版)")
        self.geometry("700x600")
        self.minsize(600, 550)

        # --- メンバー変数 ---
        self.is_reading = False
        self.stop_thread = False
        self.reading_thread = None
        self.default_pyttsx3_rate = self.get_default_pyttsx3_rate()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.create_widgets()
        self.on_engine_change()

        if not self.get_default_engine():
            self.show_startup_error()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # (GUI作成や初期化部分は変更ありません)
    def show_startup_error(self):
        messagebox.showerror(
            "依存関係エラー", 
            "読み上げに必要なライブラリが見つかりません。\n"
            "pip install customtkinter pyttsx3 gTTS pygame\n"
            "上記のコマンドでインストールしてください。"
        )
        self.after(100, self.destroy)

    def get_default_pyttsx3_rate(self):
        if pyttsx3:
            try:
                temp_engine = pyttsx3.init()
                rate = temp_engine.getProperty('rate')
                temp_engine.stop()
                return rate
            except Exception: return 200
        return 200

    def create_widgets(self):
        settings_frame = ctk.CTkFrame(self)
        settings_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)
        engine_label = ctk.CTkLabel(settings_frame, text="Engine:", font=ctk.CTkFont(weight="bold"))
        engine_label.grid(row=0, column=0, padx=(10, 5), pady=10)
        self.engine_choice = ctk.StringVar(value=self.get_default_engine())
        pyttsx3_radio = ctk.CTkRadioButton(settings_frame, text="Standard (Offline)", variable=self.engine_choice, value="pyttsx3", command=self.on_engine_change)
        pyttsx3_radio.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        gtts_radio = ctk.CTkRadioButton(settings_frame, text="High-Quality (Online)", variable=self.engine_choice, value="gtts", command=self.on_engine_change)
        gtts_radio.grid(row=0, column=2, padx=5, pady=10, sticky="w")
        if not pyttsx3: pyttsx3_radio.configure(state="disabled")
        if not gTTS or not pygame: gtts_radio.configure(state="disabled")
        speed_label_desc = ctk.CTkLabel(settings_frame, text="Speed (Standard Only):", font=ctk.CTkFont(weight="bold"))
        speed_label_desc.grid(row=1, column=0, padx=(10, 5), pady=10)
        self.speed_slider = ctk.CTkSlider(settings_frame, from_=self.default_pyttsx3_rate-100, to=self.default_pyttsx3_rate+150, command=self.update_speed_label)
        self.speed_slider.set(self.default_pyttsx3_rate)
        self.speed_slider.grid(row=1, column=1, columnspan=2, padx=5, pady=10, sticky="ew")
        self.speed_value_label = ctk.CTkLabel(settings_frame, text=f"{int(self.default_pyttsx3_rate)}", width=35)
        self.speed_value_label.grid(row=1, column=3, padx=(5, 10), pady=10)
        self.text_area = ctk.CTkTextbox(self, font=("Arial", 16), wrap="word", border_width=2)
        self.text_area.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.text_area.insert("1.0", "ここにテキストをお願います。")
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        button_frame.grid_columnconfigure((0, 1), weight=1)
        self.start_button = ctk.CTkButton(button_frame, text="読み上げ開始", command=self.start_reading_thread, font=ctk.CTkFont(size=16, weight="bold"), height=40)
        self.start_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.stop_button = ctk.CTkButton(button_frame, text="停止", command=self.stop_reading, state="disabled", fg_color="#D32F2F", hover_color="#B71C1C", height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.stop_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        self.status_bar = ctk.CTkLabel(self, text="準備完了", anchor="w", font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="ew")

    def on_closing(self):
        self.stop_reading()
        if pygame: pygame.mixer.quit()
        self.destroy()

    def on_engine_change(self):
        if self.engine_choice.get() == "pyttsx3":
            self.speed_slider.configure(state="normal")
            self.speed_value_label.configure(state="normal")
        else:
            self.speed_slider.configure(state="disabled")
            self.speed_value_label.configure(state="disabled")

    def update_speed_label(self, value):
        self.speed_value_label.configure(text=f"{int(value)}")
        
    def get_default_engine(self):
        if pyttsx3: return "gtts"
        if gTTS and pygame: return "gtts"
        return ""
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ ここがご要望に応じた修正の核心部分です ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    def split_text(self, text):
        """【シンプル版】改行を全てスペースに置換し、句読点で分割する"""
        # 1. 連続する改行や空白を一つの半角スペースに統一する
        flat_text = re.sub(r'\s+', ' ', text).strip()
        flat_text = re.sub(r'\\+', ' ', flat_text).strip()
        
        # 2. 句読点で区切ることで、停止ボタンの応答性を確保する
        #    .+? は最短マッチ、[。！？]は区切り文字、|.+は文末に句読点がない場合も拾うため
        sentences = re.findall(r'.+?[。！？]|.+', flat_text)
        
        # 3. 各文の前後の空白を削除し、空の文は除外する
        return [s.strip() for s in sentences if s.strip()]

    # --- 以下、スレッドや再生のロジック (変更なし) ---
    def start_reading_thread(self):
        text = self.text_area.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("テキストがありません", "読み上げるテキストを入力してください。")
            return
        self.is_reading = True
        self.stop_thread = False
        self.toggle_controls(reading=True)
        self.status_bar.configure(text="読み上げ準備中...")
        engine_type = self.engine_choice.get()
        target_func = self.read_with_pyttsx3 if engine_type == "pyttsx3" else self.read_with_gtts
        self.reading_thread = threading.Thread(target=target_func, args=(text,), daemon=True)
        self.reading_thread.start()
        self.after(100, self.check_thread_status)

    def stop_reading(self):
        if self.is_reading:
            self.stop_thread = True
            self.status_bar.configure(text="停止処理中...")
            if pygame and pygame.mixer.get_busy():
                pygame.mixer.stop()

    def check_thread_status(self):
        if self.reading_thread and self.reading_thread.is_alive():
            self.after(100, self.check_thread_status)
        else:
            self.is_reading = False
            self.toggle_controls(reading=False)
            status_text = "停止しました" if self.stop_thread else "読み上げ完了"
            self.status_bar.configure(text=status_text)
            if os.path.exists("temp_voice.mp3"):
                try: os.remove("temp_voice.mp3")
                except OSError as e: print(f"Error removing file: {e}")

    def toggle_controls(self, reading: bool):
        state = "disabled" if reading else "normal"
        self.start_button.configure(state=state)
        self.text_area.configure(state=state)
        self.stop_button.configure(state="normal" if reading else "disabled")
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(child, (ctk.CTkRadioButton, ctk.CTkSlider)):
                        child.configure(state=state)

    def read_with_pyttsx3(self, text):
        engine = None
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'ja' in voice.languages or 'Japanese' in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
            rate = int(self.speed_slider.get())
            engine.setProperty('rate', rate)
            sentences = self.split_text(text)
            for i, sentence in enumerate(sentences):
                if self.stop_thread: break
                self.status_bar.configure(text=f"読み上げ中... ({i+1}/{len(sentences)})")
                engine.say(sentence)
            engine.runAndWait()
        except Exception as e:
            messagebox.showerror("pyttsx3 エラー", f"読み上げ中にエラーが発生しました:\n{e}")
        finally:
            if engine:
                if self.stop_thread: engine.stop()
                del engine

    def read_with_gtts(self, text):
        temp_filename = "temp_voice.mp3"
        try:
            sentences = self.split_text(text)
            for i, sentence in enumerate(sentences):
                if self.stop_thread: break
                try:
                    self.status_bar.configure(text=f"音声生成中... ({i+1}/{len(sentences)})")
                    tts = gTTS(text=sentence, lang='ja')
                    tts.save(temp_filename)
                    if self.stop_thread: break
                    self.status_bar.configure(text=f"再生中... ({i+1}/{len(sentences)})")
                    sound = pygame.mixer.Sound(temp_filename)
                    sound.play()
                    while pygame.mixer.get_busy():
                        if self.stop_thread:
                            pygame.mixer.stop()
                            break
                        time.sleep(0.1)
                except Exception as e:
                    print(f"gTTSでエラーが発生しました: {e}")
                    if "429" in str(e):
                        messagebox.showwarning("API制限", "リクエストが多すぎます。")
                        break
        finally:
            if os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except OSError as e: print(f"一時ファイルの削除に失敗しました: {e}")

if __name__ == "__main__":
    app = TextToSpeechApp()
    app.mainloop()