import tkinter as tk
from tkinter import ttk
# This is essentially a helper class to create a slider object with extra UI elements.
# Used to create the list of attributes


class FifaSliders:

    def __init__(self, root, name, column, row):
        self.root = root
        self.column = column
        self.row = row
        self.name = name

        self.current_value = tk.IntVar()
        self.current_value.set(50)

        # label for the slider
        self.slider_label = ttk.Label(
            self.root,
            text=self.name
        )

        self.slider_label.grid(
            column=self.column + 0,
            row=self.row,
            sticky='n'
        )

        #  slider
        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=100,
            orient='horizontal',  # vertical
            command=self.slider_changed,
            variable=self.current_value
        )

        self.slider.grid(
            column=self.column + 1,
            row=self.row,
            sticky='n'
        )

        # value label
        self.value_label = ttk.Label(
            self.root,
            text=self.get_current_value()
        )
        self.value_label.grid(
            column=self.column + 2,
            row=self.row,
            columnspan=2,
            sticky='n'
        )

    def get_current_value(self):
        return float('{:.0f}'.format(self.current_value.get()))

    def slider_changed(self, event):
        self.value_label.configure(text=self.get_current_value())