# MazeGenerator v1.0.2
# Główny plik programu — pobierany przez Updater z GitHuba

import tkinter as tk
import random

APP_VERSION = "1.0.2"

class MazeGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Maze Generator {APP_VERSION}")
        self.root.geometry("800x800")
        self.root.resizable(False, False)

        self.cell_size = 20
        self.rows = 35
        self.cols = 35

        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size, height=self.rows * self.cell_size, bg="black")
        self.canvas.pack()

        self.generate_button = tk.Button(root, text="Generuj labirynt", command=self.generate_maze)
        self.generate_button.pack(pady=10)

    def generate_maze(self):
        self.canvas.delete("all")

        grid = [[1 for _ in range(self.cols)] for _ in range(self.rows)]

        def carve(x, y):
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and grid[ny][nx] == 1:
                    grid[ny][nx] = 0
                    grid[y + dy // 2][x + dx // 2] = 0
                    carve(nx, ny)

        grid[1][1] = 0
        carve(1, 1)

        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y][x] == 1:
                    self.canvas.create_rectangle(
                        x * self.cell_size, y * self.cell_size,
                        (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                        fill="gray20", outline=""
                    )

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGeneratorApp(root)
    root.mainloop()
