# kabocha_ui.py
import sys
import os
import time
import subprocess
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMainWindow
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, qInstallMessageHandler, QtMsgType
from core.intelligents import KaboAI
from core.speak import play_audio

print("Aktives Python:", sys.executable)

class KaboUI(QWidget):
    def __init__(self):
        super().__init__()
        self.kabo = KaboAI()
        self.init_ui()

    def init_ui(self):
        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Message...")
        self.input_line.returnPressed.connect(self.handle_text_input)

        self.speak_btn = QPushButton("Say something...")
        self.speak_btn.clicked.connect(self.handle_speech_input)

        self.tts_btn = QPushButton("Play Audio")
        self.tts_btn.clicked.connect(self.play_tts)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.speak_btn)
        btn_layout.addWidget(self.tts_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.chat_box)
        layout.addWidget(self.input_line)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def handle_text_input(self):
        user_input = self.input_line.text().strip()
        if user_input:
            self.chat_box.append(f"<p style='color:#c8d2bb'><b>Pascal:</b> {user_input}</p><div style='margin-bottom:5px'></div>")
            self.input_line.clear()

            response = self.get_llm_response(user_input)
            self.chat_box.append(f"<p style='color:#a1446c'><b>Kabo-chan:</b> {response}</p><div style='margin-bottom:10px'></div>")

    def handle_speech_input(self):
        user_input = "Platzhalter (STT Ergebnis)"
        self.chat_box.append(f"<b>Pascal:</b> {user_input}</p><br")

        response = self.get_llm_response(user_input)
        self.chat_box.append(f"<b>Kabo-chan:</b> {response}</p><br")

    def get_llm_response(self, text):
        return self.kabo.get_response(text)

    def play_tts(self):
        print("TTS wird erneut abgespielt...")
        play_audio(os.path.join("core", "output.wav"))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kabo-chan AI")
        self.setGeometry(200, 200, 900, 2160)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Unity-WebView oben
        self.web_view = QWebEngineView()
        self.web_view.setFixedSize(900, 900)
        url = f"http://localhost:8000?cache_buster={random.randint(0, 999999)}"
        self.web_view.load(QUrl(url))
        self.web_view.page().settings().setAttribute(
        self.web_view.page().settings().ShowScrollBars, False
        )
        webview_layout = QHBoxLayout()
        webview_layout.addStretch()  # linker Abstand
        webview_layout.addWidget(self.web_view)
        webview_layout.addStretch()  # rechter Abstand

        layout.addLayout(webview_layout)

        # Kabo-Chat darunter
        self.kabo_ui = KaboUI()
        layout.addWidget(self.kabo_ui)

        self.setLayout(layout)

def suppress_qt_warnings(msg_type, msg_log_context, msg_string):
    if msg_type == QtMsgType.QtDebugMsg and "js:" in msg_string:
        return  # Unterdrückt JavaScript-Logs
    # Für alles andere, z. B. Fehler, kann man ggf. print(msg_string) erlauben
    return

qInstallMessageHandler(suppress_qt_warnings)

if __name__ == "__main__":
    # WebGL-Server starten
    unity_webgl_path = os.path.expanduser("~/Kabocha_AI/models/UnityWebGLBuild")
    server_process = subprocess.Popen(
        ["python3", "-m", "http.server", "8000"],
        cwd=unity_webgl_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(1)
    app = QApplication(sys.argv)

    dark_stylesheet = """
    QWidget {
        background-color: #121212;
        color: #EEEEEE;
        font-family: 'Segoe UI';
        font-size: 16pt;
    }

    QPushButton {
        background-color: #1F1F1F;
        color: #FFFFFF;
        border: 1px solid #444444;
        padding: 10px;
        border-radius: 6px;
        font-size: 16pt;
    }

    QPushButton:hover {
        background-color: #2A2A2A;
    }

    QLineEdit, QTextEdit {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #444444;
        padding: 10px;
        border-radius: 6px;
        font-size: 16pt;
    }
"""
    app.setStyleSheet(dark_stylesheet)

    window = MainWindow()
    window.show()
    exit_code = app.exec_()

    server_process.terminate()
    sys.exit(exit_code)
