import tkinter as tk
from tkinter import font as tkfont
from ui.fifaoverallvalueframe import FifaOverallValueFrame
from ui.fifapositionclassifierframe import FifaPositionClassifierFrame

# https://www.geeksforgeeks.org/tkinter-application-to-switch-between-different-page-frames/
# https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        container = tk.Frame(self)

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

        container.pack()

        self.frames = {}
        for F in (FifaOverallValueFrame, FifaPositionClassifierFrame):
            page_name = F.__name__
            frame = F(container=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("FifaOverallValueFrame")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        # https://www.tutorialspoint.com/what-is-the-difference-between-focus-and-focus-set-methods-in-tkinter
        frame = self.frames[page_name]
        frame.focus_set()
        frame.tkraise()
