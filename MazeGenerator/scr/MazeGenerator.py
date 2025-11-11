# maze_gui.py
# PyQt6 GUI + Pillow export. DFS generator + auto-update z GitHub (raw).
# Zależności: PyQt6, Pillow
import sys
import os
import shutil
import json
import random
import threading
import subprocess
import urllib.request
import urllib.error
import math
from collections import deque
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QHBoxLayout,
    QVBoxLayout, QFormLayout, QComboBox, QCheckBox, QMessageBox, QSpinBox, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt6.QtCore import QTimer, Qt, QSize, QEvent

# ---------------- CONFIG ----------------
LOCAL_VERSION = "1.0"
VERSION_JSON_URL = "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/version.json"
PY_RAW_TEMPLATE = "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/v{ver}.py"
BACKUP_SUFFIX = ".backup_before_update"

# ---------------- helper: wersje ----------------
def parse_version(v):
    parts = [p for p in str(v).strip().split(".") if p != ""]
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except:
            nums.append(0)
    return tuple(nums)

def compare_versions(a, b):
    pa = parse_version(a)
    pb = parse_version(b)
    L = max(len(pa), len(pb))
    for i in range(L):
        ai = pa[i] if i < len(pa) else 0
        bi = pb[i] if i < len(pb) else 0
        if ai < bi:
            return -1
        if ai > bi:
            return 1
    return 0

# ---------------- network download ----------------
def fetch_url_text(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "MazeUpdater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")

def fetch_url_bytes(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "MazeUpdater/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

# ---------------- auto-update logic ----------------
def check_for_update_async(parent_widget):
    def job():
        try:
            txt = fetch_url_text(VERSION_JSON_URL, timeout=7)
            j = json.loads(txt)
            remote_ver = str(j.get("version", "")).strip()
            if not remote_ver:
                return
        except Exception:
            return

        if compare_versions(LOCAL_VERSION, remote_ver) >= 0:
            return

        def ask_user():
            mb = QMessageBox(parent_widget)
            mb.setWindowTitle("Aktualizacja dostępna")
            mb.setText(f"Dostępna nowa wersja: {remote_ver}\nAktualna wersja: {LOCAL_VERSION}\n\nPobrać i zainstalować teraz?")
            mb.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            res = mb.exec()
            if res == QMessageBox.StandardButton.Yes:
                threading.Thread(target=download_and_replace, args=(parent_widget, remote_ver), daemon=True).start()
        QApplication.instance().postEvent(parent_widget, _CallEvent(ask_user))

    threading.Thread(target=job, daemon=True).start()

def download_and_replace(parent_widget, remote_ver):
    py_url = PY_RAW_TEMPLATE.format(ver=remote_ver)
    try:
        data = fetch_url_bytes(py_url, timeout=15)
    except Exception as e:
        QApplication.instance().postEvent(parent_widget, _CallEvent(lambda: QMessageBox.warning(parent_widget, "Aktualizacja", f"Błąd pobierania: {e}")) )
        return

    if not data or len(data) < 100:
        QApplication.instance().postEvent(parent_widget, _CallEvent(lambda: QMessageBox.warning(parent_widget, "Aktualizacja", "Pobrano podejrzanie mały plik. Przerwano.") ) )
        return

    try:
        current_path = os.path.abspath(sys.argv[0])
        dirpath = os.path.dirname(current_path)
        temp_path = os.path.join(dirpath, f".tmp_update_v{remote_ver}.py")
        backup_path = current_path + BACKUP_SUFFIX

        with open(temp_path, "wb") as f:
            f.write(data)

        shutil.copy2(current_path, backup_path)
        os.replace(temp_path, current_path)

        def notify_and_restart():
            QMessageBox.information(parent_widget, "Aktualizacja", f"Pobrano i zainstalowano wersję {remote_ver}.\nAplikacja uruchomi się ponownie.")
            python = sys.executable
            args = [python] + sys.argv
            try:
                subprocess.Popen(args)
            except Exception:
                pass
            QApplication.instance().quit()
        QApplication.instance().postEvent(parent_widget, _CallEvent(notify_and_restart))
    except Exception as e:
        QApplication.instance().postEvent(parent_widget, _CallEvent(lambda: QMessageBox.warning(parent_widget, "Aktualizacja", f"Błąd podczas instalacji aktualizacji: {e}")) )
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, current_path)
        except:
            pass

class _CallEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    def __init__(self, callable_):
        super().__init__(self.EVENT_TYPE)
        self.callable = callable_

# ---------------- Maze generator ----------------
def make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1

def dfs_generator(W, H, seed=None, sides=4):
    """
    DFS generator dla labiryntu o n-kątnych krokach.
    sides = liczba boków wielokąta (3–12)
    """
    if seed is not None:
        random.seed(seed)

    maze = [['#' for _ in range(W)] for _ in range(H)]
    visited = [[False]*W for _ in range(H)]
    start = (1, 1)
    stack = [start]
    visited[start[1]][start[0]] = True
    maze[start[1]][start[0]] = ' '
    yield ('carve', start[0], start[1], maze)

    # tworzymy wektory ruchu dla n-kątów
    directions = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        dx = round(math.cos(angle) * 2)
        dy = round(math.sin(angle) * 2)
        if dx == 0 and dy == 0:
            continue
        directions.append((dx, dy))

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            wx, wy = dx // 2, dy // 2
            if 0 < nx < W-1 and 0 < ny < H-1 and not visited[ny][nx]:
                neighbors.append((nx, ny, wx, wy))
        if neighbors:
            nx, ny, wx, wy = random.choice(neighbors)
            maze[y+wy][x+wx] = ' '
            maze[ny][nx] = ' '
            visited[ny][nx] = True
            stack.append((nx, ny))
            yield ('carve', x+wx, y+wy, maze)
            yield ('carve', nx, ny, maze)
        else:
            stack.pop()
    yield ('done', maze)

def bfs_solve(maze, start, end):
    H = len(maze)
    W = len(maze[0])
    q = deque([start])
    prev = {start: None}
    while q:
        x,y = q.popleft()
        if (x,y) == end:
            break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and maze[ny][nx] == ' ' and (nx,ny) not in prev:
                prev[(nx,ny)] = (x,y)
                q.append((nx,ny))
    if end not in prev:
        return []
    path = []
    cur = end
    while cur:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

# ---------------- GUI ----------------
class MazeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Maze Generator - PyQt6")
        self.setMinimumSize(900, 600)

        self.default_W = 41
        self.default_H = 41

        form = QFormLayout()

        self.w_input = QSpinBox()
        self.w_input.setRange(5, 301)
        self.w_input.setValue(self.default_W)
        self.w_input.setSingleStep(2)

        self.h_input = QSpinBox()
        self.h_input.setRange(5, 301)
        self.h_input.setValue(self.default_H)
        self.h_input.setSingleStep(2)

        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("opcjonalny (liczba)")

        self.variant_combo = QComboBox()
        self.variant_combo.addItems([
            "góra–dół", "rogi", "przeciwne rogi", "lewo–prawo", "dół→środek"
        ])
        self.variant_combo.setCurrentIndex(0)

        self.sides_input = QSpinBox()
        self.sides_input.setRange(3, 12)
        self.sides_input.setValue(4)
        form.addRow("Ilość ścian figury bazowej:", self.sides_input)

        self.scale_default_checkbox = QCheckBox("Domyślne")
        self.scale_default_checkbox.setChecked(True)
        self.scale_input = QSpinBox()
        self.scale_input.setRange(1, 200)
        self.scale_input.setValue(self.compute_default_scale(self.default_W))
        self.scale_input.setEnabled(False)

        self.scale_default_checkbox.stateChanged.connect(self.on_scale_mode_changed)

        self.instant_checkbox = QCheckBox("Generuj natychmiast")

        self.generate_btn = QPushButton("Generuj")
        self.save_btn = QPushButton("Zapisz PNG")
        self.save_solution_btn = QPushButton("Zapisz z rozwiązaniem")
        self.save_btn.setEnabled(False)
        self.save_solution_btn.setEnabled(False)

        self.generate_btn.clicked.connect(self.on_generate)
        self.save_btn.clicked.connect(self.on_save)
        self.save_solution_btn.clicked.connect(self.on_save_solution_dialog)

        form.addRow("Szerokość (W):", self.w_input)
        form.addRow("Wysokość (H):", self.h_input)
        form.addRow("Seed:", self.seed_input)
        form.addRow("Wejście/wyjście:", self.variant_combo)

        scale_row = QHBoxLayout()
        scale_row.addWidget(self.scale_default_checkbox)
        scale_row.addWidget(self.scale_input)
        form.addRow("Skala (px na komórkę):", scale_row)
        form.addRow(self.instant_checkbox)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.generate_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.save_solution_btn)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #f7f7f7; border: 1px solid #bbb;")
        self.preview_pixmap = None
        self.preview_dirty = False

        left = QVBoxLayout()
        left.addLayout(form)
        left.addLayout(btn_row)
        left.addStretch()

        main = QHBoxLayout()
        main.addLayout(left, 0)
        main.addWidget(self.preview_label, 1)

        self.setLayout(main)
        self.resize(1200, 700)

        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.animation_step)

        self.gen_iter = None
        self.current_maze = None
        self.W = None
        self.H = None
        self.scale = None
        self.start = None
        self.end = None

    def event(self, ev):
        if isinstance(ev, _CallEvent):
            try:
                ev.callable()
            except Exception:
                pass
            return True
        return super().event(ev)

    def compute_default_scale(self, W):
        scale = (2480 - 300) // max(1, int(W))
        return max(1, scale)

    def on_scale_mode_changed(self, state):
        custom = not self.scale_default_checkbox.isChecked()
        self.scale_input.setEnabled(custom)
        if not custom:
            W = self.w_input.value()
            self.scale_input.setValue(self.compute_default_scale(W))

    def pick_entrance_and_exit(self, W, H, variant):
        if variant == "góra–dół":
            sx = random.choice([i for i in range(1, W, 2)])
            ex = random.choice([i for i in range(1, W, 2)])
            return (sx, 0), (ex, H-1)
        if variant == "rogi":
            return (1,0), (W-2,H-1)
        if variant == "przeciwne rogi":
            return (W-2,0), (1,H-1)
        if variant == "lewo–prawo":
            sy = random.choice([i for i in range(1, H, 2)])
            ey = random.choice([i for i in range(1, H, 2)])
            return (0, sy), (W-1, ey)
        if variant == "dół→środek":
            sx = random.choice([i for i in range(1, W, 2)])
            return (sx, H-1), (W//2, H//2)
        return (1,0), (W-2,H-1)

    def prepare_canvas(self, W, H, scale):
        if W*scale <= 0 or H*scale <= 0:
            return
        pixmap = QPixmap(W*scale, H*scale)
        pixmap.fill(QColor("white"))
        self.preview_pixmap = pixmap
        self.preview_dirty = True
        self.update_preview_display()

    def draw_maze_to_pixmap(self, maze, scale, highlight=None):
        W = len(maze[0])
        H = len(maze)
        required_size = QSize(W*scale, H*scale)
        if self.preview_pixmap is None or self.preview_pixmap.size() != required_size:
            self.preview_pixmap = QPixmap(required_size)
        img = self.preview_pixmap
        painter = QPainter(img)
        painter.fillRect(0,0, img.width(), img.height(), QColor("white"))
        for y in range(H):
            for x in range(W):
                if maze[y][x] == '#':
                    painter.fillRect(x*scale, y*scale, scale, scale, QColor("black"))
        if highlight:
            painter.setPen(Qt.PenStyle.NoPen)
            for (x,y) in highlight:
                painter.fillRect(x*scale, y*scale, scale, scale, QColor("red"))
        painter.end()
        self.preview_dirty = True
        self.update_preview_display()

    def update_preview_display(self):
        if self.preview_pixmap is None:
            return
        target_size = self.preview_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self.preview_pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.preview_dirty = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview_display()

    def on_generate(self):
        W = make_odd(self.w_input.value())
        H = make_odd(self.h_input.value())
        seed_text = self.seed_input.text().strip()
        seed = int(seed_text) if seed_text.isdigit() else None
        variant = self.variant_combo.currentText()
        scale = self.compute_default_scale(W) if self.scale_default_checkbox.isChecked() else int(self.scale_input.value())
        sides = self.sides_input.value()
        self.start, self.end = self.pick_entrance_and_exit(W,H,variant)

        self.W, self.H, self.scale = W, H, scale
        self.prepare_canvas(W,H,scale)
        self.current_maze = [['#' for _ in range(W)] for _ in range(H)]

        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.save_solution_btn.setEnabled(False)
        self.w_input.setEnabled(False)
        self.h_input.setEnabled(False)
        self.seed_input.setEnabled(False)
        self.variant_combo.setEnabled(False)
        self.scale_default_checkbox.setEnabled(False)
        self.scale_input.setEnabled(False)

        if self.instant_checkbox.isChecked():
            final = None
            for step in dfs_generator(W,H,seed,sides):
                if step[0] == 'done':
                    final = step[1]
            if final is None:
                QMessageBox.warning(self, "Błąd", "Generacja nie powiodła się.")
                self.unlock_ui()
                return
            self.current_maze = [row[:] for row in final]
            self.ensure_entrances()
            self.draw_maze_to_pixmap(self.current_maze, self.scale)
            self.unlock_ui()
            return

        self.gen_iter = dfs_generator(W,H,seed,sides)
        base_area = 32 * 32
        area = max(1, W * H)
        speed_factor = area / base_area
        base_ms = 12
        interval_ms = max(1, int(base_ms / speed_factor))
        self.timer.setInterval(interval_ms)
        self.timer.start()

    def ensure_entrances(self):
        sx,sy = self.start
        ex,ey = self.end
        if
