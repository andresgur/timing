from tkinter import filedialog
import tkinter as tk
from os.path import sys
sys.path.insert(0, '/home/agurpide/scripts/pythonscripts')
from timing_analysis.lightcurves.load import load_ligth_curve
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from stingray import Lightcurve
from stingray import Powerspectrum,AveragedPowerspectrum
from plot_utils import plot_functions as pf
from timing_analysis.Model import Model
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg


def loadlc(model,canvas,rebinning_entry):
    fileName = filedialog.askopenfilename(filetypes=(("Light curve", "*.lc"),("Fits files","*.fits")))

    time,cts,std,time_res =  load_ligth_curve(fileName)

#store the lightcurve in memory in the model
    model.lc = Lightcurve(time,cts)
    model.std = std
    
    #plot it
    fig,ax = plot_lc(time,cts,std,time_res,'blue',fileName)

    plt.show()

    rebinning_entry.config(state=NORMAL)

    #fig_x, fig_y = 100, 100

    #fig_photo = draw_figure(canvas,fig,loc=(fig_x,fig_y))

    #fig_w, fig_h = fig_photo.width(), fig_photo.height()

    # Add more elements to the canvas, potentially on top of the figure
    #canvas.create_line(200, 50, fig_x + fig_w / 2, fig_y + fig_h / 2)
    #canvas.create_text(200, 50, text="Zero-crossing", anchor="s")


def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo


def plot_lc(time,cts,std,time_res,color,lightcurve):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_axes([0, 0, 1, 1])

    # Max, min, etc
    xmin = time[0]
    xmax = time[-1]
    ymax = np.max(cts + std)
    ###
    # Plotting
    ###
    # Source
    plt.errorbar(time, cts, yerr=std, fmt='o', color=color,ls="-",linewidth=0.5,elinewidth=0.5,markersize=1,errorevery=10,label="%s" %lightcurve)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("counts s$^{-1}$", fontsize=16)

    plt.xlim(xmin, xmax)
    #plt.ylim(ymin, ymax)

    pf.set_plot_format(14)
    plt.ticklabel_format(style='sci',axis='x',scilimits=(1,1000))
    return fig,ax

def rebinlc(model,seconds):
    model.rebinnedlc = model.lc.rebin(float(seconds))


w, h = 600, 400
main_window = tk.Tk()
model = Model()

rebinning_label = Label(main_window, text="dt").grid(row=0, column=0)
rebinning_entry = Entry(main_window, bd =5,state="readonly").grid(row=0, column=1)
units_label = Label(main_window, text="s").grid(row=0, column=2)
rebinning_entry.instert(0,"")
canvas = tk.Canvas(main_window, width=w, height=h).grid(row=1, column=0)
canvas.pack()

load_button = tk.Button(main_window, text="Load Lightcurve", command=lambda:loadlc(model,canvas,rebinning_entry)).grid(row=2, column=0)

rebin_button = tk.Button(main_window, text="Rebin Lightcurve", command=lambda:rebinlc(model,rebinning_entry.get())).grid(row=2, column=1)


main_window.mainloop()
