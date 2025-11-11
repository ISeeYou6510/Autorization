# maze_gui.py
# PyQt6 GUI + Pillow export. DFS generator
# Zależności: PyQt6, Pillow

import sys
import random
from collections import deque
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QHBoxLayout,
    QVBoxLayout, QFormLayout, QComboBox, QCheckBox, QMessageBox, QSpinBox
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt6.QtCore import QTimer, Qt, QSize

# ---------- Maze engine ----------

def make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1

def dfs_generator(W, H, seed=None):
    """Zwraca generator kroków rzeźbienia."""
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

# ---------- GUI ----------

class MazeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Maze Generator - PyQt6")
        self.setMinimumSize(800, 600)

        self.default_W = 41
        self.default_H = 41
        self.default_speed_base = 32  # dla 32x32 prędkość bazowa

        # UI: formularz
        form = QFormLayout()

        self.w_input = QSpinBox()
        self.w_input.setRange(5, 201)
        self.w_input.setValue(self.default_W)
        self.w_input.setSingleStep(2)
        self.h_input = QSpinBox()
        self.h_input.setRange(5, 201)
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
        self.save_btn.setEnabled(False)

        self.generate_btn.clicked.connect(self.on_generate)
        self.save_btn.clicked.connect(self.on_save)

        form.addRow("Szerokość (W)", self.w_input)
        form.addRow("Wysokość (H)", self.h_input)
        form.addRow("Seed", self.seed_input)
        form.addRow("Wejście/wyjście", self.variant_combo)
        scale_row = QHBoxLayout()
        scale_row.addWidget(self.scale_default_checkbox)
        scale_row.addWidget(self.scale_input)
        form.addRow("Skala (px na komórkę)", scale_row)
        form.addRow(self.instant_checkbox)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.generate_btn)
        btn_row.addWidget(self.save_btn)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #eee; border: 1px solid #aaa;")
        self.preview_pixmap = None

        left = QVBoxLayout()
        left.addLayout(form)
        left.addLayout(btn_row)
        left.addStretch()

        main = QHBoxLayout()
        main.addLayout(left, 0)
        main.addWidget(self.preview_label, 1)
        self.setLayout(main)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_step)

        # stan
        self.gen_iter = None
        self.current_maze = None
        self.W = None
        self.H = None
        self.scale = None
        self.start = None
        self.end = None

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
        pixmap = QPixmap(W*scale, H*scale)
        pixmap.fill(QColor("white"))
        self.preview_pixmap = pixmap
        self.preview_label.setPixmap(self.preview_pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def draw_maze_to_pixmap(self, maze, scale, highlight=None):
        W = len(maze[0])
        H = len(maze)
        if self.preview_pixmap is None or self.preview_pixmap.size() != QSize(W*scale, H*scale):
            self.preview_pixmap = QPixmap(W*scale, H*scale)
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
        self.preview_label.setPixmap(self.preview_pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def on_generate(self):
        W = make_odd(self.w_input.value())
        H = make_odd(self.h_input.value())
        seed_text = self.seed_input.text().strip()
        seed = int(seed_text) if seed_text.isdigit() else None
        variant = self.variant_combo.currentText()
        scale = self.compute_default_scale(W) if self.scale_default_checkbox.isChecked() else int(self.scale_input.value())
        self.start, self.end = self.pick_entrance_and_exit(W,H,variant)

        self.W, self.H, self.scale = W, H, scale
        self.prepare_canvas(W,H,scale)
        self.current_maze = [['#' for _ in range(W)] for _ in range(H)]

        if self.instant_checkbox.isChecked():
            # generuj natychmiast
            maze = None
            for step in dfs_generator(W,H,seed):
                if step[0]=='done':
                    maze = step[1]
            self.current_maze = [row[:] for row in maze]
            self.ensure_entrances()
            self.draw_maze_to_pixmap(self.current_maze, scale)
            self.save_btn.setEnabled(True)
        else:
            # animacja
            self.gen_iter = dfs_generator(W,H,seed)
            speed = max(1, 50 * self.default_speed_base // max(W,H))  # przy większych labiryntach szybciej
            self.timer.setInterval(speed)
            self.generate_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.timer.start()

    def ensure_entrances(self):
        sx,sy = self.start
        ex,ey = self.end
        # wejście
        if sy==0: self.current_maze[0][sx]=self.current_maze[1][sx]=' '
        if sy==self.H-1: self.current_maze[self.H-1][sx]=self.current_maze[self.H-2][sx]=' '
        if sx==0: self.current_maze[sy][0]=self.current_maze[sy][1]=' '
        if sx==self.W-1: self.current_maze[sy][self.W-1]=self.current_maze[sy][self.W-2]=' '
        # wyjście
        if ey==0: self.current_maze[0][ex]=self.current_maze[1][ex]=' '
        if ey==self.H-1: self.current_maze[self.H-1][ex]=self.current_maze[self.H-2][ex]=' '
        if ex==0: self.current_maze[ey][0]=self.current_maze[ey][1]=' '
        if ex==self.W-1: self.current_maze[ey][self.W-1]=self.current_maze[ey][self.W-2]=' '

    def animation_step(self):
        try:
            step = next(self.gen_iter)
        except StopIteration:
            step = ('done', self.current_maze)
        if step[0]=='carve':
            _,x,y,maze_snapshot = step
            self.current_maze = [row[:] for row in maze_snapshot]
            self.draw_maze_to_pixmap(self.current_maze, self.scale)
        elif step[0]=='done':
            self.current_maze = [row[:] for row in step[1]]
            self.ensure_entrances()
            self.draw_maze_to_pixmap(self.current_maze, self.scale)
            self.timer.stop()
            self.generate_btn.setEnabled(True)
            self.save_btn.setEnabled(True)

    def on_save(self):
        if not self.current_maze:
            QMessageBox.warning(self,"Brak labiryntu","Najpierw wygeneruj labirynt.")
            return
        maze = self.current_maze
        W,H,scale = len(maze[0]),len(maze),self.scale
        maze_fn,sol_fn="maze.png","maze_solution.png"
        img = Image.new("RGB",(W*scale,H*scale),"white")
        draw = ImageDraw.Draw(img)
        for y in range(H):
            for x in range(W):
                if maze[y][x]=='#': draw.rectangle([x*scale,y*scale,(x+1)*scale-1,(y+1)*scale-1],fill=(0,0,0))
        img.save(maze_fn)
        # rozwiązanie
        sx,sy=self.start
        ex,ey=self.end
        start_bfs=(sx,1) if sy==0 else (sx,H-2) if sy==H-1 else (1,sy) if sx==0 else (W-2,sy) if sx==W-1 else (sx,sy)
        end_bfs=(ex,1) if ey==0 else (ex,H-2) if ey==H-1 else (1,ey) if ex==0 else (W-2,ey) if ex==W-1 else (ex,ey)
        path=bfs_solve(maze,start_bfs,end_bfs)
        img2=img.copy()
        draw2=ImageDraw.Draw(img2)
        for x,y in path:
            draw2.rectangle([x*scale,y*scale,(x+1)*scale-1,(y+1)*scale-1],fill=(255,0,0))
        img2.save(sol_fn)
        QMessageBox.information(self,"Zapisano",f"Zapisano pliki:\n{maze_fn}\n{sol_fn}")

# ---------- Uruchomienie ----------

def main():
    app = QApplication(sys.argv)
    w = MazeWindow()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
