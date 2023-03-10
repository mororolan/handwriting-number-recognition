# -*- coding: utf-8 -*-
# @File    : hand_write_gui.py
# @Author  : Robin Lan
# @Time    : 8/3/23 22:36
# @Software: PyCharm
# @Description: This file is used to create a GUI for the user to draw a number and predict the number.

from tkinter import *
import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import time
from keras.models import load_model

model = load_model('models/model.h5')


class Gui(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Creating title
        self.title("Handwritten Digit Recognition")

        # Creating elements
        self.canvas = tk.Canvas(self, width=250, height=250, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw your number in the canvas", font=("Arial", 30, 'bold'))
        self.classify_btn = tk.Button(self, text="Predict", command=self.predict_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear)

        # Grid structure
        self.label.grid(row=0, column=0, pady=20, padx=20)
        self.canvas.grid(row=1, pady=1, padx=1, columnspan=2)
        self.classify_btn.grid(row=2, column=0, pady=2, padx=2)
        self.button_clear.grid(row=3, column=0, pady=2)

        # Binding mouse events
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def predict_handwriting(self):
        # save postscript image without using win32gui
        self.canvas.postscript(file='temp.eps', colormode='color')
        # use PIL to convert to PNG
        img = Image.open('temp.eps')
        raw_img = img.resize((28, 28))

        timestamp = time.time()
        formatted_time = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp))

        # convert rgb to grayscale
        img = raw_img.convert('L')
        img = ImageOps.invert(img)
        img = np.array(img)
        # reshaping to support our model input and normalizing
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        res = model.predict([img])[0]
        number = np.argmax(res)
        acc = max(res)
        raw_img.save('draws/{a}-{b}-{acc}.png'.format(a=formatted_time, b=number, acc="%.2f" % acc))
        self.label.configure(text='The number is: {}, accuracy: {}%'.format(number, int(acc * 100)))

    def clear(self):
        self.canvas.delete("all")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x, self.y, self.x + r, self.y + r, width=3, fill='white')



