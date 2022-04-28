import tkinter as tk
import torch
from modelfactory.model import ModelFactory
from ui.fifasliders import FifaSliders
from tkinter import ttk


class FifaOverallValueFrame(ttk.Frame):
    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)

        self.controller = controller
        self.container = container

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.fifa_attributes = ["pace",	"shooting",	"passing", "dribbling",	"defending", "physic"]
        self.attribute_sliders = self.build_list_of_sliders()

        # Button used to get slider info.
        prediction_button = ttk.Button(self, text='Overall Fifa Rating Prediction', command=self.get_a_prediction)
        prediction_button.grid(column=1,row=7,sticky='n')

        # Info label
        info_label = ttk.Label(self,text='Your predicted rating is: ')
        info_label.grid(column=1,row=8,sticky='n')

        # overall label
        self.overall_value = tk.DoubleVar()
        overall_label = ttk.Label(self, textvariable=self.overall_value)
        overall_label.grid(column=1,row=9,sticky='n')

        # Model for Frame
        self.frame_model = ModelFactory().get_model("ReducedMLP")

        # https://www.tutorialspoint.com/tkinter-keypress-keyrelease-events
        self.bind('<Return>', self.switch_frame)

    def build_list_of_sliders(self):
        return [FifaSliders(self, self.fifa_attributes[i], 0, i) for i in range(len(self.fifa_attributes))]

    def get_a_prediction(self):

        slider_values = [att.get_current_value() for att in self.attribute_sliders]

        prediction = self.frame_model.prediction(torch.FloatTensor(slider_values))

        # Update the overall label via the overall_value variable
        self.overall_value.set(prediction)
        self.focus_set()

    def switch_frame(self, event):
        next_frame = "FifaPositionClassifierFrame"
        self.controller.show_frame(next_frame)
