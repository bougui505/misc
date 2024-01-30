#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time
import datetime
from dateutil import tz

plt.rcParams['figure.constrained_layout.use'] = False

DATAFILE = "/media/usb0/t-temp_c-humidity/data.dat"

def plot_data(data, outfile, ndays=1):
    print(f"plotting for {ndays} day(s)")
    ndays = ndays * 24 * 60 * 60  # s
    sel = data[:, 0] >= time.time() - ndays
    data = data[sel]
    fig, ax = plt.subplots()
    color = 'tab:blue'
    ax.plot_date(mdate.epoch2num(data[:,0]), data[:,1], fmt='-', color=color, lw=3)
    T_max = data[:,1].max()
    T_min = data[:,1].min()
    ax.axhline(y=T_max,linestyle="--",linewidth=1.0, color=color)
    ax.axhline(y=T_min,linestyle="--",linewidth=1.0, color=color)
    # Choose your xtick format string
    ax.set_ylabel("T (°C)")
    plt.grid()

    ax2 = ax.twinx()
    color = 'cyan'
    ax2.set_ylabel('humidity (%)', color=color)
    ax2.plot_date(mdate.epoch2num(data[:,0]), data[:,2], fmt='-', color=color, lw=1)
    ax2.tick_params(axis='y', labelcolor=color)

    # Use a DateFormatter to set the data to the correct format.
    # date_fmt = '%d-%m-%y %H:%M:%S'
    date_fmt = '%d-%m %H:%M'
    date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
    ax2.xaxis.set_major_formatter(date_formatter)
    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    ax2.set_xlabel("date")

    last_sample_date = datetime.datetime.fromtimestamp(data[-1,0]).strftime('%d-%m-%y %H:%M:%S')
    last_sample_T = data[-1,1]
    last_sample_H = data[-1,2]
    plt.title(f"{last_sample_date} T={last_sample_T}°C;H={last_sample_H}%")
    # plt.grid()
    plt.savefig(outfile)

data = np.genfromtxt(DATAFILE)


plot_data(data, outfile="/var/www/html/figures/T_year.png", ndays=365)
plot_data(data, outfile="/var/www/html/figures/T_month.png", ndays=31)
plot_data(data, outfile="/var/www/html/figures/T_week.png", ndays=7)
plot_data(data, outfile="/var/www/html/figures/T.png", ndays=1)
