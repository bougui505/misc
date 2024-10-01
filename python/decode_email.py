#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Oct  1 14:47:26 2024

import email
import os

from bs4 import BeautifulSoup


def log(msg):
    try:
        logging.info(msg)  # type: ignore
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir

def print_body(msg):
    try:
        body = msg.get_payload(decode=True).decode()
    except (AttributeError, UnicodeDecodeError):
        body = msg.get_payload()
    try:
        soup = BeautifulSoup(body, features="lxml")  # type: ignore
        body = soup.get_text()
    except AttributeError:
        pass
    print(body)

def decode_email(emailfilename):
    f = open(emailfilename, 'r', encoding='ISO-8859-1')
    msg = email.message_from_string(f.read())
    print(f"From: {msg['From']}")
    print(f"To: {msg['To']}")
    print(f"Subject: {msg['Subject']}")
    print(f"Date: {msg['Date']}")
    print("")
    if msg.is_multipart():
        for part in msg.get_payload():
            print_body(part)
    else:
        print_body(msg)
    # try:
    #     body = msg.get_payload(decode=True).decode()  # type: ignore
    # except (AttributeError, UnicodeDecodeError):
    #     body = msg.get_payload()
    # if not isinstance(body, list):
    #     body = [body,]
    # for e in body:
    #     try:
    #         soup = BeautifulSoup(e, features="lxml")  # type: ignore
    #         e = soup.get_text()
    #     except AttributeError:
    #         pass
    #     print(e)


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
    parser.add_argument("-e", "--email", help="email file to decode")
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
    if args.email is not None:
        decode_email(args.email)
