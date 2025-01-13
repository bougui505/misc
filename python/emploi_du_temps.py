#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Sep  6 16:08:51 2024

import datetime
import itertools
import os
from datetime import timedelta


def log(msg):
    try:
        logging.info(msg)  # type: ignore
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def get_week_dates(base_date, start_day, end_day=5):
    """
    Return entire week of dates based on given date limited by start_day and end_day.
    If end_day is None, return only start_day.

    Use get_week_dates(date.today(), 1, 7) to get current week dates.

    >>> from datetime import date
    >>> get_week_dates(date(2015,1,16), 3, 5)
    [datetime.date(2015, 1, 14), datetime.date(2015, 1, 15), datetime.date(2015, 1, 16)]

    >>> get_week_dates(date(2015,1,15), 2, 5)
    [datetime.date(2015, 1, 13), datetime.date(2015, 1, 14), datetime.date(2015, 1, 15), datetime.date(2015, 1, 16)]
    """
    monday = base_date - timedelta(days=base_date.isoweekday() - 1)
    week_dates = [monday + timedelta(days=i) for i in range(7)]
    return week_dates[start_day - 1:end_day or start_day]

def getweek(week="A", offset=0, print_header=True):
    """
    offset: number of week to add
    to get the next week schedule offset=1

    >>> getweek()
    Subject,Start date,Start time,Description
    Début des cours SA,2/9/2024,10:10,Emploi du temps 6C
    Fin des cours SA,2/9/2024,17:10,Emploi du temps 6C
    Début des cours SA,3/9/2024,8:15,Emploi du temps 6C
    Fin des cours SA,3/9/2024,16:15,Emploi du temps 6C
    Début des cours SA,4/9/2024,8:15,Emploi du temps 6C
    Fin des cours SA,4/9/2024,11:30,Emploi du temps 6C
    Début des cours SA,5/9/2024,8:15,Emploi du temps 6C
    Fin des cours SA,5/9/2024,16:15,Emploi du temps 6C
    Début des cours SA,6/9/2024,8:15,Emploi du temps 6C
    Fin des cours SA,6/9/2024,16:15,Emploi du temps 6C
    >>> getweek(week="B", offset=1)
    Subject,Start date,Start time,Description
    Début des cours SB,9/9/2024,8:15,Emploi du temps 6C
    Fin des cours SB,9/9/2024,17:10,Emploi du temps 6C
    Début des cours SB,10/9/2024,8:15,Emploi du temps 6C
    Fin des cours SB,10/9/2024,16:15,Emploi du temps 6C
    Début des cours SB,11/9/2024,8:15,Emploi du temps 6C
    Fin des cours SB,11/9/2024,11:30,Emploi du temps 6C
    Début des cours SB,12/9/2024,8:15,Emploi du temps 6C
    Fin des cours SB,12/9/2024,17:10,Emploi du temps 6C
    Début des cours SB,13/9/2024,8:15,Emploi du temps 6C
    Fin des cours SB,13/9/2024,16:15,Emploi du temps 6C

    """
    startdate = datetime.date.today() + timedelta(days=offset*7)
    dates = get_week_dates(startdate, 1)
    if print_header:
        print("Subject,Start date,Start time,End time,Description")
    for date in dates:
        if week=="A":
            if date.weekday()==0:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},10:10,17:10,Emploi du temps 6C")
            if date.weekday()==1:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,17:10,Emploi du temps 6C")
            if date.weekday()==2:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,11:30,Emploi du temps 6C")
            if date.weekday()==3:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,16:15,Emploi du temps 6C")
            if date.weekday()==4:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,16:15,Emploi du temps 6C")
        if week=="B":
            if date.weekday()==0:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,17:10,Emploi du temps 6C")
            if date.weekday()==1:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,17:10,Emploi du temps 6C")
            if date.weekday()==2:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,11:30,Emploi du temps 6C")
            if date.weekday()==3:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},9:15,17:10,Emploi du temps 6C")
            if date.weekday()==4:
                print(f"Cours Malo S{week},{date.day}/{date.month}/{date.year},8:15,16:15,Emploi du temps 6C")





if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-w", "--week", help="set week A or B")
    parser.add_argument("-o", "--offset", help="Number of weeks to offset. To get next week -o 1, in 2 weeks -o 2 and so on...", default=0, type=int)
    parser.add_argument("-n", "--nweeks", help="Make the schedule for the given number of weeks. Please provide the -w option to start with week A or B", type=int, default=1)
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()

    if args.nweeks is not None and args.week is not None:
        otherweek = list(set(["A","B"])-set([args.week]))[0]
        weeks = args.week + otherweek
        weeks = itertools.cycle(weeks)
        for w in range(args.nweeks):
            week = next(weeks)
            getweek(week=week, offset=w, print_header=(w==0))
