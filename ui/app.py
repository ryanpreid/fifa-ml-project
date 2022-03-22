import tkinter as tk


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure the root window
        self.title("Fifa Predictor")

        window_width = 600
        window_height = 400

        # get the screen dimension
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.resizable(False, False)
