#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time
import datetime
from dateutil import tz

plt.rcParams['figure.constrained_layout.use'] = False

DATAFILE = "/media/usb0/t-temp_c-humidity/data.dat"
MAXTIMEFRAME = 24  # h

MAXTIMEFRAME = MAXTIMEFRAME * 60 * 60

def plot_data(data, outfile):
    fig, ax = plt.subplots()
    ax.plot_date(mdate.epoch2num(data[:,0]), data[:,1], fmt='-')
    # Choose your xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'
    # Use a DateFormatter to set the data to the correct format.
    date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
    ax.xaxis.set_major_formatter(date_formatter)
    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    plt.xlabel("date")
    plt.ylabel("T (°C)")
    last_sample_date = datetime.datetime.fromtimestamp(data[-1,0]).strftime('%d-%m-%y %H:%M:%S')
    last_sample_T = data[-1,1]
    plt.title(f"{last_sample_date}:{last_sample_T}°C")
    plt.grid()
    plt.savefig(outfile)

data = np.genfromtxt(DATAFILE)

sel = data[:, 0] >= time.time() - MAXTIMEFRAME
data = data[sel]

plot_data(data, outfile="/var/www/html/figures/T.png")
