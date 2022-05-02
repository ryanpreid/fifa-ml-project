import tkinter as tk

import numpy as np
import torch
from modelfactory.model import ModelFactory
from tkinter import ttk
from ui.fifasliders import FifaSliders


class FifaPositionClassifierFrame(ttk.Frame):
    def __init__(self, container, controller):
        tk.Frame.__init__(self, container)

        self.controller = controller
        self.container = container

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.fifa_attributes = [("height", 155, 206),
                          ("attacking", 1, 99),
                          ("defending", 1, 99),
                          ("pace", 1, 99),
                          ("shooting", 1, 99),
                          ("passing", 1, 99),
                          ("dribbling", 1, 99),
                          ("physic", 1, 99),
                          ("skill", 1, 99),
                          ("movement", 1, 99),
                          ("power", 1, 99),
                          ("mentality", 1, 99),
                          ("goalkeeping", 1, 99)]

        self.attribute_sliders = self.build_list_of_sliders()

        # Button used to get slider info and make prediction with deep learning
        dl_prediction_button = ttk.Button(self, text='Deep Learning Prediction', command=self.get_a_dl_prediction)
        dl_prediction_button.grid(column=0,row=14,sticky='n',)

        # Button used to get slider info and make prediction with naive bayes
        nb_prediction_button = ttk.Button(self, text='Naive Bayes Prediction', command=self.get_a_nb_prediction)
        nb_prediction_button.grid(column=2,row=14,sticky='n',)

        # Info label
        info_label = ttk.Label(self,text='Your predicted position is: ')
        info_label.grid(column=1,row=15,sticky='n',)

        # overall label
        self.overall_value = tk.DoubleVar()
        overall_label = ttk.Label(self,textvariable=self.overall_value)
        overall_label.grid(column=1,row=16,sticky='n',)

        # Model for Frame
        self.frame_model_DL = ModelFactory().get_model("PositionClassifierModel")

        # Model for Frame
        self.frame_model_NB = ModelFactory().get_model("NB")

        self.bind('<Return>', self.switch_frame)

    def build_list_of_sliders(self):
        return [FifaSliders(self, self.fifa_attributes[i][0], 0, i, self.fifa_attributes[i][1], self.fifa_attributes[i][2]) for i in range(len(self.fifa_attributes))]

    def get_a_dl_prediction(self):

        slider_values = [att.get_current_value() for att in self.attribute_sliders]

        print(slider_values)
        prediction = self.frame_model_DL.prediction(torch.FloatTensor(slider_values))

        print(prediction)
        # Update the overall label via the overall_value variable
        self.overall_value.set(prediction)
        self.focus_set()

    def get_a_nb_prediction(self):

        slider_values = [att.get_current_value() for att in self.attribute_sliders]

        print(slider_values)
        prediction = self.frame_model_NB.prediction(slider_values)

        print(prediction)
        # Update the overall label via the overall_value variable
        self.overall_value.set(prediction)
        self.focus_set()

    def switch_frame(self, event):
        next_frame = "FifaKmeansFrame"
        self.controller.show_frame(next_frame)

