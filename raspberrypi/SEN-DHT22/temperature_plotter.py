#!/usr/bin/env python3

import datetime
import os
import time

import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
from dateutil import tz

plt.rcParams['figure.constrained_layout.use'] = False

DATAFILE = "/media/usb0/t-temp_c-humidity/data.dat"

def get_period(timesteps, period=24, unit='hour'):
    """
    """
    if unit == 'hour':
        period = period * 60 * 60
    periods = [timesteps[-1]]
    while periods[-1] >= timesteps[0]:
        periods.append(periods[-1]-period)
    periods.pop()
    periods = list(reversed(periods))
    periods.pop()
    periods = np.asarray(periods).astype(np.datetime64)
    return periods

def get_gradient(data):
    grad = data[1:, :2] - data[:-1, :2]
    # convert from °C/s to °C/h
    grad = (grad[:, 1] / grad[:, 0]) * 60 * 60
    grad = np.insert(grad, 0, grad[0])
    return grad
        

def plot_data(data, outfile, ndays=1, compute_gradient=False):
    print(f"plotting for {ndays} day(s)")
    ndays = ndays * 24 * 60 * 60  # s
    sel = data[:, 0] >= time.time() - ndays
    data = data[sel]
    periods = get_period(data[:, 0], period=24, unit='hour')
    if compute_gradient:
        grad = get_gradient(data)
        print(grad.shape, data.shape)
        fig, ax = plt.subplots(figsize=[8,4.5])
        for t in periods:
            ax.axvline(x=t, color='k', linewidth=1.5)
        ax.plot_date(data[:,0].astype(np.datetime64), grad, fmt='-')
        bn, ext = os.path.splitext(outfile)
        outgradfile = f"{bn}_grad{ext}"
        # Use a DateFormatter to set the data to the correct format.
        # date_fmt = '%d-%m-%y %H:%M:%S'
        date_fmt = '%d-%m %H:%M'
        date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_ylabel("gradient (°C/h)")
        # Sets the tick labels diagonal so they fit easier.
        fig.autofmt_xdate()
        plt.grid()
        plt.savefig(outgradfile)
    fig, ax = plt.subplots()
    for t in periods:
        ax.axvline(x=t, color='k', linewidth=1.5)
    color = 'tab:blue'
    ax.plot_date(data[:,0].astype(np.datetime64), data[:,1], fmt='-', color=color, lw=3, label='in')
    ax.plot_date(data[:,0].astype(np.datetime64), data[:,3], fmt='.--', color=color, lw=2, label='out')
    T_max = np.nanmax(data[:,1].max())
    T_min = np.nanmin(data[:,1].min())
    T_max_out = np.nanmax(data[:,3])
    T_min_out = np.nanmin(data[:,3])
    ax.axhline(y=T_max,linestyle="-",linewidth=1.0, color=color)
    ax.axhline(y=T_min,linestyle="-",linewidth=1.0, color=color)
    ax.axhline(y=T_max_out,linestyle="--",linewidth=1.0, color=color)
    ax.axhline(y=T_min_out,linestyle="--",linewidth=1.0, color=color)
    xcenter = data[:,0].min() + (data[:,0].max()-data[:,0].min())//2
    ax.text(xcenter.astype(np.datetime64), T_max, f"{T_max}")
    ax.text(xcenter.astype(np.datetime64), T_min, f"{T_min}")
    ax.text(data[:,0].min().astype(np.datetime64), T_max_out, f"{T_max_out}")
    ax.text(data[:,0].min().astype(np.datetime64), T_min_out, f"{T_min_out}")
    # Choose your xtick format string
    ax.set_ylabel("T (°C)")
    plt.grid()
    plt.legend()

    ax2 = ax.twinx()
    color = 'tab:cyan'
    ax2.set_ylabel('humidity (%)', color=color)
    ax2.plot_date(data[:,0].astype(np.datetime64), data[:,2], fmt='-', color=color, lw=1)
    ax2.plot_date(data[:,0].astype(np.datetime64), data[:,4], fmt='.--', color=color, lw=1)
    ax2.tick_params(axis='y', labelcolor=color)

    # Use a DateFormatter to set the data to the correct format.
    # date_fmt = '%d-%m-%y %H:%M:%S'
    date_fmt = '%d-%m %H:%M'
    date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
    ax2.xaxis.set_major_formatter(date_formatter)
    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    ax2.set_xlabel("date")

    last_sample_date = datetime.datetime.fromtimestamp(data[-1,0]).strftime('%d-%m %H:%M')
    last_sample_T = data[-1,1]
    last_sample_H = data[-1,2]
    plt.title(f"{last_sample_date}    T={last_sample_T}°C  H={last_sample_H}%")
    # plt.grid()
    plt.savefig(outfile)

data = np.genfromtxt(DATAFILE)


plot_data(data, outfile="/var/www/html/figures/T_year.png", ndays=365)
plot_data(data, outfile="/var/www/html/figures/T_month.png", ndays=31)
plot_data(data, outfile="/var/www/html/figures/T_week.png", ndays=7, compute_gradient=True)
plot_data(data, outfile="/var/www/html/figures/T.png", ndays=1, compute_gradient=True)
