# maze_gui.py
# PyQt6 GUI + Pillow export. DFS generator + robust auto-update (mode C).
# Dependencies: PyQt6, Pillow
# Auto-update behavior (C): download new file -> write installer script -> spawn installer -> quit.
# Installer replaces file after current process exits, launches new process, removes itself.

import sys
import os
import shutil
import json
import random
import threading
import subprocess
import urllib.request
import urllib.error
import time
import tempfile
from collections import deque
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QHBoxLayout,
    QVBoxLayout, QFormLayout, QComboBox, QCheckBox, QMessageBox, QSpinBox, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt6.QtCore import QTimer, Qt, QSize, QEvent

# ---------------- CONFIG ----------------
LOCAL_VERSION = "1.0.1"
VERSION_JSON_URL = "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/version.json"
# Primary candidate(s) for the file to download (raw)
CANDIDATE_UPDATE_URLS = [
    "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/scr/MazeGenerator.py",
    "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/src/maze_gui.py",
    "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/MazeGenerator.py",
    "https://raw.githubusercontent.com/ISeeYou6510/Autorization/main/MazeGenerator/v{ver}.py"
]
BACKUP_SUFFIX = ".backup_before_update"
UPDATE_LOG = "update_error.log"

MIN_VALID_BYTES = 120  # minimalny rozmiar pobranego pliku by uznać go za prawdopodobny kod
VALID_MARKERS = [b"class MazeWindow", b"def main", b"PyQt6", b"maze_gui"]

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

# ---------------- network download (robust) ----------------
def fetch_url_text(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "MazeUpdater/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except Exception as e:
        raise

def fetch_url_bytes(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "MazeUpdater/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        raise

# ---------------- auto-update logic (mode C) ----------------
def log_update(msg):
    try:
        with open(UPDATE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.asctime()}: {msg}\n")
    except Exception:
        pass

def check_for_update_async(parent_widget):
    """Asynchroniczny check: pobiera version.json; jeśli nowsza, pyta użytkownika i uruchamia download."""
    def job():
        try:
            txt = fetch_url_text(VERSION_JSON_URL, timeout=7)
            j = json.loads(txt)
            remote_ver = str(j.get("version", "")).strip()
            if not remote_ver:
                log_update("version.json nie zawiera 'version' lub jest pusta.")
                return
        except Exception as e:
            log_update(f"check_for_update failed: {repr(e)}")
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
                threading.Thread(target=download_and_replace_mode_c, args=(parent_widget, remote_ver), daemon=True).start()
        # postEvent to GUI thread
        QApplication.instance().postEvent(parent_widget, _CallEvent(ask_user))

    threading.Thread(target=job, daemon=True).start()

def write_installer_script(installer_path):
    script = r'''# updater_installer (auto-generated)
import sys, os, time, shutil, subprocess

def main():
    try:
        target = sys.argv[1]   # full path to original file to replace
        newfile = sys.argv[2]  # full path to downloaded temp file
        python = sys.argv[3]   # python executable
        orig_args = sys.argv[4:]
    except Exception as e:
        print("installer: bad args", e)
        return

    timeout = 30.0
    deadline = time.time() + timeout
    while True:
        try:
            try:
                os.replace(newfile, target)
                break
            except Exception:
                shutil.copy2(newfile, target + ".tmp_update")
                os.replace(target + ".tmp_update", target)
                try:
                    os.remove(newfile)
                except:
                    pass
                break
        except Exception as e:
            if time.time() > deadline:
                print("installer: timeout replacing file:", e)
                return
            time.sleep(0.5)

    try:
        subprocess.Popen([python] + orig_args, close_fds=True)
    except Exception as e:
        print("installer: failed to launch new process:", e)

    try:
        os.remove(sys.argv[0])
    except Exception:
        pass

if __name__ == "__main__":
    main()
'''
    with open(installer_path, "w", encoding="utf-8") as f:
        f.write(script)

def validate_downloaded_data(data):
    if not data or len(data) < MIN_VALID_BYTES:
        return False, "Pobrany plik zbyt mały"
    # check for simple markers in bytes
    for m in VALID_MARKERS:
        if m in data:
            return True, ""
    return False, "Pobrany plik nie zawiera oczekiwanych markerów (class MazeWindow / def main / PyQt6)."

def download_and_replace_mode_c(parent_widget, remote_ver):
    """
    Mode C updater:
    - try multiple candidate URLs (some may include {ver})
    - validate content
    - write temp file + installer -> spawn -> quit
    """
    last_exc = None
    data = None
    tried = []

    # build candidate list (format v{ver} if needed)
    candidates = []
    for u in CANDIDATE_UPDATE_URLS:
        if "{ver}" in u:
            candidates.append(u.format(ver=remote_ver))
        else:
            candidates.append(u)

    for url in candidates:
        tried.append(url)
        try:
            d = fetch_url_bytes(url, timeout=15)
            ok, reason = validate_downloaded_data(d)
            if not ok:
                log_update(f"candidate {url} failed validation: {reason} (len={len(d) if d else 0})")
                last_exc = Exception(f"validation failed: {reason}")
                continue
            data = d
            break
        except Exception as e:
            last_exc = e
            log_update(f"download attempt failed for {url}: {repr(e)}")
            continue

    if data is None:
        msg = f"Nie udało się pobrać aktualizacji z żadnego z URLi.\nPróbowano:\n" + "\n".join(tried) + f"\nOstatni błąd:\n{repr(last_exc)}"
        log_update(msg)
        def show_err():
            QMessageBox.warning(parent_widget, "Aktualizacja nieudana", msg)
        QApplication.instance().postEvent(parent_widget, _CallEvent(show_err))
        return

    try:
        current_path = os.path.abspath(sys.argv[0])
        dirpath = os.path.dirname(current_path)

        fd, temp_path = tempfile.mkstemp(prefix=".maze_update_", suffix=".py", dir=dirpath)
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(data)

        installer_path = os.path.join(dirpath, ".maze_updater_installer.py")
        write_installer_script(installer_path)
        try:
            os.chmod(installer_path, 0o700)
        except Exception:
            pass

        python_exe = sys.executable or "python"
        orig_args = [current_path] + sys.argv[1:]
        popen_args = [python_exe, installer_path, current_path, temp_path, python_exe] + orig_args[1:]
        if os.name == "nt":
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(popen_args, close_fds=True, creationflags=CREATE_NEW_PROCESS_GROUP)
        else:
            subprocess.Popen(popen_args, close_fds=True)

        def info_and_quit():
            QMessageBox.information(parent_widget, "Aktualizacja",
                                    "Pobrano aktualizację. Aplikacja teraz się zamknie. Instalator podmieni plik i uruchomi nową wersję.")
            QApplication.instance().quit()
        QApplication.instance().postEvent(parent_widget, _CallEvent(info_and_quit))
    except Exception as e:
        log_update(f"installer setup failed: {repr(e)}")
        def show_err2():
            QMessageBox.warning(parent_widget, "Aktualizacja", f"Błąd podczas przygotowania instalatora: {repr(e)}")
        QApplication.instance().postEvent(parent_widget, _CallEvent(show_err2))
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

# mały mechanizm do wykonywania callables w wątku GUI przez postEvent
class _CallEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    def __init__(self, callable_):
        super().__init__(self.EVENT_TYPE)
        self.callable = callable_

# ---------------- Maze generator ----------------
def make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1

def dfs_generator(W, H, seed=None):
    """iterative DFS carving on rectangular grid (odd cell spacing)."""
    if seed is not None:
        random.seed(seed)
    maze = [['#' for _ in range(W)] for _ in range(H)]
    visited = [[False]*W for _ in range(H)]
    start = (1, 1)
    stack = [start]
    visited[start[1]][start[0]] = True
    maze[start[1]][start[0]] = ' '
    yield ('carve', start[0], start[1], maze)
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in ((2,0),(-2,0),(0,2),(0,-2)):
            nx, ny = x + dx, y + dy
            if 0 < nx < W-1 and 0 < ny < H-1 and not visited[ny][nx]:
                neighbors.append((nx, ny, dx//2, dy//2))
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
        self.start, self.end = self.pick_entrance_and_exit(W,H,variant)

        # ustawiamy self.W i self.H od razu
        self.W, self.H, self.scale = W, H, scale
        self.prepare_canvas(W,H,scale)
        self.current_maze = [['#' for _ in range(W)] for _ in range(H)]

        # zablokuj UI
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
            for step in dfs_generator(W,H,seed):
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

        # animacja
        self.gen_iter = dfs_generator(W,H,seed)
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
        # wejście
        if sy == 0:
            self.current_maze[0][sx] = ' '
            if 1 < self.H:
                self.current_maze[1][sx] = ' '
        if sy == self.H-1:
            self.current_maze[self.H-1][sx] = ' '
            if self.H-2 >= 0:
                self.current_maze[self.H-2][sx] = ' '
        if sx == 0:
            self.current_maze[sy][0] = ' '
            if 1 < self.W:
                self.current_maze[sy][1] = ' '
        if sx == self.W-1:
            self.current_maze[sy][self.W-1] = ' '
            if self.W-2 >= 0:
                self.current_maze[sy][self.W-2] = ' '
        # wyjście
        if ey == 0:
            self.current_maze[0][ex] = ' '
            if 1 < self.H:
                self.current_maze[1][ex] = ' '
        if ey == self.H-1:
            self.current_maze[self.H-1][ex] = ' '
            if self.H-2 >= 0:
                self.current_maze[self.H-2][ex] = ' '
        if ex == 0:
            self.current_maze[ey][0] = ' '
            if 1 < self.W:
                self.current_maze[ey][1] = ' '
        if ex == self.W-1:
            self.current_maze[ey][self.W-1] = ' '
            if self.W-2 >= 0:
                self.current_maze[ey][self.W-2] = ' '

    def animation_step(self):
        if not self.gen_iter:
            self.timer.stop()
            self.unlock_ui()
            return
        try:
            step = next(self.gen_iter)
            if step[0] == 'carve':
                _, x, y, maze_snapshot = step
                # snapshot może być referencją - bezpiecznie przypisz kopię
                self.current_maze = [row[:] for row in maze_snapshot]
                self.draw_maze_to_pixmap(self.current_maze, self.scale)
            elif step[0] == 'done':
                _, maze_snapshot = step
                self.current_maze = [row[:] for row in maze_snapshot]
                self.ensure_entrances()
                self.draw_maze_to_pixmap(self.current_maze, self.scale)
                self.timer.stop()
                self.unlock_ui()
        except StopIteration:
            self.timer.stop()
            self.unlock_ui()
        except Exception as e:
            self.timer.stop()
            QMessageBox.warning(self, "Błąd animacji", str(e))
            self.unlock_ui()

    def unlock_ui(self):
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.save_solution_btn.setEnabled(True)
        self.w_input.setEnabled(True)
        self.h_input.setEnabled(True)
        self.seed_input.setEnabled(True)
        self.variant_combo.setEnabled(True)
        self.scale_default_checkbox.setEnabled(True)
        self.on_scale_mode_changed(0)

    def on_save(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Zapisz PNG", "", "PNG Files (*.png)")
        if not filename:
            return
        self.save_maze_png(filename, self.current_maze, self.scale)

    def on_save_solution_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Zapisz z rozwiązaniem", "", "PNG Files (*.png)")
        if not filename:
            return
        path = filename
        path = path.replace("\\", "/")
        path = path if path.endswith(".png") else path + ".png"
        path = os.path.abspath(path)
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        self.save_maze_solution_png(path)

    def save_maze_png(self, filename, maze, scale):
        H = len(maze)
        W = len(maze[0])
        img = Image.new("RGB", (W*scale, H*scale), "white")
        draw = ImageDraw.Draw(img)
        for y in range(H):
            for x in range(W):
                if maze[y][x] == '#':
                    draw.rectangle([x*scale, y*scale, x*scale+scale-1, y*scale+scale-1], fill="black")
        img.save(filename)

    def save_maze_solution_png(self, filename):
        path = filename
        maze = [row[:] for row in self.current_maze]
        # korekta punktów BFS: jeśli wejście/wyjście na zewnątrz konwertujemy
        sx, sy = self.start
        ex, ey = self.end
        start_bfs = (sx, 1) if sy == 0 else (sx, self.H-2) if sy == self.H-1 else (1, sy) if sx == 0 else (self.W-2, sy) if sx == self.W-1 else (sx, sy)
        end_bfs = (ex, 1) if ey == 0 else (ex, self.H-2) if ey == self.H-1 else (1, ey) if ex == 0 else (self.W-2, ey) if ex == self.W-1 else (ex, ey)
        path_cells = bfs_solve(maze, start_bfs, end_bfs)
        scale = self.scale
        H = len(maze)
        W = len(maze[0])
        img = Image.new("RGB", (W*scale, H*scale), "white")
        draw = ImageDraw.Draw(img)
        for y in range(H):
            for x in range(W):
                if maze[y][x] == '#':
                    draw.rectangle([x*scale, y*scale, x*scale+scale-1, y*scale+scale-1], fill="black")
        for x,y in path_cells:
            draw.rectangle([x*scale, y*scale, x*scale+scale-1, y*scale+scale-1], fill="red")
        img.save(path)

# ---------------- main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MazeWindow()
    win.show()
    # uruchom check_for_update po krótkim delay, bezpieczniej niż natychmiast
    QTimer.singleShot(1000, lambda: check_for_update_async(win))
    sys.exit(app.exec())
