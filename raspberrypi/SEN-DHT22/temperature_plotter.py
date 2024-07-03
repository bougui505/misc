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

def get_period(timesteps, period=24, unit='hour', format=True):
    """
    """
    if unit == 'hour':
        period = period * 60 * 60
    if unit == 'day':
        period = period * 24 * 60 * 60
    periods = [timesteps[-1]]
    while periods[-1] >= timesteps[0]:
        periods.append(periods[-1]-period)
    periods.pop()
    periods = list(reversed(periods))
    periods.pop()
    if format:
        periods = np.asarray(periods).astype(np.datetime64)
    return periods

def get_gradient(data):
    grad = data[1:, :2] - data[:-1, :2]
    # convert from °C/s to °C/h
    grad = (grad[:, 1] / grad[:, 0]) * 60 * 60
    grad = np.insert(grad, 0, grad[0])
    return grad

def get_minmax(Ts: np.ndarray, npts=4):
    if len(Ts) == 0:
        return np.nan, np.nan
    Ts = Ts[~np.isnan(Ts)]
    if len(Ts) == 0:
        return np.nan, np.nan
    sorted = np.sort(Ts)
    T_min = sorted[:npts].mean()
    T_max = sorted[-npts:].mean()
    return T_min, T_max

def get_stats(data, ndays):
    ndays = ndays * 24 * 60 * 60  # s
    sel = data[:, 0] >= time.time() - ndays
    data = data[sel]
    timesteps = data[:,0]
    T_in = data[:, 1]
    H_in = data[:, 2]
    T_out = data[:, 3]
    H_out = data[:, 4]
    periods = get_period(timesteps, period=24, unit='hour', format=False)
    Tin_min, Tin_max, Tout_min, Tout_max = list(), list(), list(), list()
    T_in_t, T_out_t = list(), list()  # temperature at the same time for the other days
    def nonan_append(l, v):
        if not np.isnan(v):
            l.append(v)
    for t1, t2 in zip(periods, periods[1:]):
        sel = np.logical_and(timesteps > t1, timesteps <= t2)
        win_Tin = T_in[sel]
        win_Tout = T_out[sel]
        if len(win_Tin)>0:
            nonan_append(T_in_t, win_Tin[-1])
        if len(win_Tout)>0:
            nonan_append(T_out_t, win_Tout[-1])
        T_min, T_max = get_minmax(win_Tin)
        nonan_append(Tin_min, T_min)
        nonan_append(Tin_max, T_max)
        T_min, T_max = get_minmax(win_Tout)
        nonan_append(Tout_min, T_min)
        nonan_append(Tout_max, T_max)
    return Tin_min, Tin_max, Tout_min, Tout_max, T_in_t, T_out_t

def plot_stats(data, ndays, outfile):
    last_sample_date = datetime.datetime.fromtimestamp(data[-1,0]).strftime('%d-%m %H:%M')
    Tin_min, Tin_max, Tout_min, Tout_max, T_in_t, T_out_t = get_stats(data, ndays=ndays)
    ndays_count = len(Tin_min)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))
    ax1.hist(Tin_min, bins='auto', density=False, color='tab:blue', alpha=0.5, label='min in', edgecolor='black')
    ax1.hist(Tin_max, bins='auto', density=False, color='tab:blue', alpha=0.5, label='max in', hatch='x', edgecolor='black')
    ax1.hist(Tout_min, bins='auto', density=False, color='tab:green', alpha=0.5, label='min out', edgecolor='black')
    ax1.hist(Tout_max, bins='auto', density=False, color='tab:green', alpha=0.5, label='max out', hatch='x', edgecolor='black')
    ax1.axvline(x=data[-1, 1], color='tab:blue', ls=(5, (10, 3)))
    ax1.axvline(x=data[-1, 3], color='tab:green', ls=(5, (10, 3)))
    ax1.legend()
    # ax1.set_xlabel("T (°C)")
    ax1.set_ylabel("Nbre de jours")
    ax1.set_title(f"T min et max journalières sur {ndays_count} jours ({last_sample_date})")
    # plt.savefig(outfile_minmax)
    # plt.clf()
    ax2.hist(T_in_t, bins='auto', density=False, color='tab:blue', alpha=0.5, label='in', edgecolor='black')
    ax2.axvline(x=data[-1, 1], color='tab:blue', ls=(5, (10, 3)))
    ax2.hist(T_out_t, bins='auto', density=False, color='tab:green', alpha=0.5, label='out', edgecolor='black')
    ax2.axvline(x=data[-1, 3], color='tab:green', ls=(5, (10, 3)))
    ax2.set_xlabel("T (°C)")
    ax2.set_ylabel("Nbre de jours")
    ax2.set_title(f"Températures à cette heure sur {ndays_count} jours ({last_sample_date})")
    # ax2.legend()
    plt.savefig(outfile)
    # plt.savefig(outfile_daily)

def plot_scatter(data, ndays, outfile):
    ndays = ndays * 24 * 60 * 60  # s
    sel = data[:, 0] >= time.time() - ndays
    data = data[sel]
    T_in = data[:, 1]
    T_out = data[:, 3]
    T_min = min(np.nanmin(T_in), np.nanmin(T_out))
    T_max = max(np.nanmax(T_in), np.nanmax(T_out))
    plt.clf()
    plt.plot([T_min, T_max], [T_min, T_max])
    plt.scatter(T_out, T_in, s=10, alpha=0.5)
    plt.xlabel("T_out (°C)")
    plt.ylabel("T_in (°C)")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(outfile)
        

def plot_data(data, outfile, ndays=1, compute_gradient=False):
    print(f"plotting for {ndays} day(s)")
    ndays = ndays * 24 * 60 * 60  # s
    sel = data[:, 0] >= time.time() - ndays
    data = data[sel]
    periods = get_period(data[:, 0], period=24, unit='hour')
    if compute_gradient:
        # fig = plt.figure(figsize=[8,9])
        # ax = plt.subplot(212)
        fig, (ax, ax_sub) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=[8,8])
        grad = get_gradient(data)
        for t in periods:
            ax_sub.axvline(x=t, color='k', linewidth=1.5)
        ax_sub.plot_date(data[:,0].astype(np.datetime64), grad, fmt='-')
        # Use a DateFormatter to set the data to the correct format.
        # date_fmt = '%d-%m-%y %H:%M:%S'
        date_fmt = '%d-%m %H:%M'
        date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
        ax_sub.xaxis.set_major_formatter(date_formatter)
        ax_sub.set_ylabel("gradient (°C/h)")
        # Sets the tick labels diagonal so they fit easier.
        fig.autofmt_xdate()
        ax_sub.grid()
    else:
        fig, ax = plt.subplots(figsize=[8,4.5])
    for t in periods:
        ax.axvline(x=t, color='k', linewidth=1.5)
    date_fmt = '%d-%m %H:%M'
    date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
    ax.xaxis.set_major_formatter(date_formatter)
    color = 'tab:blue'
    ax.plot_date(data[:,0].astype(np.datetime64), data[:,1], fmt='-', color=color, lw=3, label='in')
    ax.plot_date(data[:,0].astype(np.datetime64), data[:,3], fmt='.--', color='tab:green', lw=2, label='out')
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
    ax.grid()
    ax.legend()

    # ax2 = ax.twinx()
    # color = 'tab:cyan'
    # ax2.set_ylabel('humidity (%)', color=color)
    # ax2.plot_date(data[:,0].astype(np.datetime64), data[:,2], fmt='-', color=color, lw=1)
    # ax2.plot_date(data[:,0].astype(np.datetime64), data[:,4], fmt='.--', color=color, lw=1)
    # ax2.tick_params(axis='y', labelcolor=color)

    # Use a DateFormatter to set the data to the correct format.
    # date_fmt = '%d-%m-%y %H:%M:%S'
    date_fmt = '%d-%m %H:%M'
    date_formatter = mdate.DateFormatter(date_fmt, tz=tz.gettz('Europe/Paris'))
    # ax2.xaxis.set_major_formatter(date_formatter)
    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    # ax2.set_xlabel("date")

    last_sample_date = datetime.datetime.fromtimestamp(data[-1,0]).strftime('%d-%m %H:%M')
    last_sample_T = data[-1,1]
    last_sample_H = data[-1,2]
    last_sample_T_out = data[-1,3]
    last_sample_H_out = data[-1,4]
    plt.title(f"{last_sample_date} T={last_sample_T}°C H={last_sample_H}% T_out={last_sample_T_out:.1f}°C H_out={last_sample_H_out}%")
    plt.savefig(outfile)

data = np.genfromtxt(DATAFILE)


plot_data(data, outfile="/var/www/html/figures/T.png", ndays=1, compute_gradient=False)
plot_data(data, outfile="/var/www/html/figures/T_week.png", ndays=7, compute_gradient=False)
plot_stats(data, ndays=3*30, outfile="/var/www/html/figures/T_stat.png")
