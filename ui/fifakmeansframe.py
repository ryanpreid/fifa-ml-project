import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models.kmeans import Kmeans
from tkinter import ttk


class FifaKmeansFrame(ttk.Frame):
    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)

        self.controller = controller
        self.container = container

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.kmeans = Kmeans()

        # Slider label
        self.label_name = "num of clusters"
        self.slider_label = ttk.Label(self, text=self.label_name)
        self.slider_label.grid(column=1, row=1, sticky='w')

        # Slider
        self.current_value = tk.DoubleVar()
        self.current_value.set(4)
        self.slider = ttk.Scale(self, from_=1, to=11, orient="horizontal", command=self.slider_changed, variable=self.current_value)
        self.slider.grid(column=1, row=1, sticky='n')

        # Slider value
        self.value_label = ttk.Label(self,text=self.get_current_value())
        self.value_label.grid(column=1, row=1, sticky='e')

        # Button used to get slider info.
        prediction_button = ttk.Button(self, text='Make a Prediction', command=self.kmeans_prediction)
        prediction_button.grid(column=1,row=2,sticky='n')

        # Plot canvas
        self.prediction_figure = plt.Figure(figsize=(8, 4), dpi=100)
        self.axis_3d = self.prediction_figure.add_subplot(121, projection='3d', computed_zorder=False)
        self.axis_2d = self.prediction_figure.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.prediction_figure, self)
        self.canvas.get_tk_widget().grid(column=1,row=4)

        # Accuracy label
        self.accuracy_value = tk.StringVar()
        self.accuracy_value.set("Accuracy score: ")
        self.accuracy_label = ttk.Label(self, textvariable=self.accuracy_value)
        self.accuracy_label.grid(column=1, row=5, sticky='n')

        # https://www.tutorialspoint.com/tkinter-keypress-keyrelease-events
        self.bind('<Return>', self.switch_frame)

    def get_current_value(self):
        return int('{:.0f}'.format(self.current_value.get()))

    def slider_changed(self, event):
        self.value_label.configure(text=self.get_current_value())

    def switch_frame(self, event):
        next_frame = "FifaOverallValueFrame"
        self.controller.show_frame(next_frame)

    def kmeans_prediction(self):

        slider_value = self.get_current_value()

        self.kmeans.kmeans_training(slider_value)
        self.kmeans.kmeans_prediction()
        self.kmeans.get_3d_plot(self.axis_3d)
        self.kmeans.get_2d_plot(self.axis_2d)
        self.canvas.draw()
        self.accuracy_value.set(self.kmeans.get_accuracy())
        self.focus_set()