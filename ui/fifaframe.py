import tkinter as tk
import torch
from modelfactory.model import ModelFactory
from ui.fifasliders import FifaSliders
from tkinter import ttk


class FifaFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        self.container = container

        self.container.columnconfigure(0, weight=1)
        self.container.columnconfigure(1, weight=1)
        self.container.columnconfigure(2, weight=1)

        self.fifa_attributes = ["pace",	"shooting",	"passing", "dribbling",	"defending", "physic"]
        self.attribute_sliders = self.build_list_of_sliders()

        # Button used to get slider info.
        prediction_button = ttk.Button(self.container, text='Overall Fifa Rating Prediction', command=self.get_a_prediction)
        prediction_button.grid(column=1,row=7,sticky='n',)

        # Info label
        info_label = ttk.Label(self.container,text='Your predicted rating is: ')
        info_label.grid(column=1,row=8,sticky='n',)

        # overall label
        self.overall_value = tk.DoubleVar()
        overall_label = ttk.Label(self.container,textvariable=self.overall_value)
        overall_label.grid(column=1,row=9,sticky='n',)

        # Model for Frame
        self.frame_model = ModelFactory().get_model("ReducedMLP")

    def build_list_of_sliders(self):
        return [FifaSliders(self.container, self.fifa_attributes[i], 0, i) for i in range(len(self.fifa_attributes))]

    def get_a_prediction(self):

        slider_values = [att.get_current_value() for att in self.attribute_sliders]

        prediction = self.frame_model.prediction(torch.FloatTensor(slider_values))

        # Update the overall label via the overall_value variable
        self.overall_value.set(prediction)

