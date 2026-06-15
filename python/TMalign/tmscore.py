#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue May 27 09:58:03 2025

import os
import subprocess
import sys
import tempfile
import signal
import atexit


import typer

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.callback()
def callback(debug:bool=False):
    """
    This is a template file for a Python script using Typer.
    It contains a main function and a test function.
    """
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = debug

@app.command()
def usalign():
    """
    Read data from standard input and write to standard output.\n
    Lines contains:\n
    pdb1,pdb2,[selection1],[selection2]\n
    \n
    The command must be run as follows:\n 
    cat input.txt | python tmscore.py usalign\n
    \n
    The command could also be run in parallel using the split command (e.g. for 16 processes in parallel):\n
    cat input.txt | split -n r/16 -u --filter="tmscore usalign"\n
    \n
    Example output:\n
    pdb1='1abc.pdb',pdb2='2xyz.pdb',sel1='all',sel2='all',tmscore=0.1234\n
    ...\n
    """ 

    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                print("Error: At least two arguments are required (pdb1 and pdb2).")
                continue
            try:
                pdb1 = parts[0]
                pdb2 = parts[1]
                sel1 = parts[2] if len(parts) > 2 else None
                sel2 = parts[3] if len(parts) > 3 else None
                if sel1 in ("-", "all"):
                    sel1 = None
                if sel2 in ("-", "all"):
                    sel2 = None
                tmscore = run_usalign(
                    pdb1, pdb2, selmodel=sel1, selnative=sel2
                )
                print(f"{pdb1=},{pdb2=},{sel1=},{sel2=},{tmscore=:.4f}", flush=True)
            except Exception as e:
                print(f"Error processing line '{line}': {e}", file=sys.stderr)
    else:
        print("No input provided. Please pipe data to this script.")

_ACTIVE_TEMPFILES = set()

def _cleanup_tempfiles():
    while _ACTIVE_TEMPFILES:
        filepath = _ACTIVE_TEMPFILES.pop()
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

def _signal_handler(signum, frame):
    _cleanup_tempfiles()
    sys.exit(128 + signum)

# Register cleanups
atexit.register(_cleanup_tempfiles)
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, _signal_handler)

def run_usalign(model, native, selmodel=None, selnative=None, verbose=False):
    """
    """
    selmodel = selmodel if selmodel is not None else "polymer.protein"
    selnative = selnative if selnative is not None else "polymer.protein"
    # model_filename = tempfile.mktemp(suffix=".pdb", dir="/dev/shm")
    # native_filename = tempfile.mktemp(suffix=".pdb", dir="/dev/shm")
    with tempfile.NamedTemporaryFile(
        suffix=".pdb", dir="/dev/shm"
    ) as model_file, tempfile.NamedTemporaryFile(
        suffix=".pdb", dir="/dev/shm"
    ) as native_file:
        _ACTIVE_TEMPFILES.add(model_file.name)
        _ACTIVE_TEMPFILES.add(native_file.name)
        try:
            pymolsave(model, selmodel, model_file.name, verbose=verbose)
            pymolsave(native, selnative, native_file.name, verbose=verbose)
            tmscore = get_tmscore(model_file.name, native_file.name)
        finally:
            _ACTIVE_TEMPFILES.discard(model_file.name)
            _ACTIVE_TEMPFILES.discard(native_file.name)
    return tmscore

def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir

def get_tmscore(modelfile, nativefile):
    scriptdir = GetScriptDir()
    cmd = f"{scriptdir}/USalign -mm 1 -ter 0 -mol prot {modelfile} {nativefile}".split(" ")
    # Options of USalign:
    # -mm  Multimeric alignment option:
    #      1: alignment of two multi-chain oligomeric structures
    # -ter Number of chains to align.
    #      0: align all chains from all models (recommended for aligning
    #         biological assemblies, i.e. biounits)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    if process is None:
        raise RuntimeError("Failed to start USalign process.")

    stdout, stderr = process.communicate()

    if stderr:
        raise RuntimeError(f"USalign error: {stderr.strip()}")

    lines = stdout.splitlines()
    tmscores = []
    for line in lines:
        if line.startswith("TM-score="):
            line = line.strip()
            tmscores.append(float(line.split()[1]))
        if len(tmscores) == 2:
            break
    if not tmscores:
        raise RuntimeError("USalign process did not produce any TM-score outputs.")
    tmscore = max(tmscores)
    return tmscore

def pymolsave(infile, sel, outfile, verbose=False):
    import pymol2

    with pymol2.PyMOL() as p:
        p.cmd.feedback(action="disable", module="all", mask="everything")
        if os.path.exists(infile):
            p.cmd.load(filename=infile, object="mymodel")
        else:
            fetch_path = os.path.expandvars('$HOME/pdb')
            p.cmd.set('fetch_path', fetch_path, quiet=0)
            p.cmd.fetch(infile, name="mymodel")
        p.cmd.remove(f"not (mymodel and ({sel}))")
        # Remove alternate locations, keeping the first available (e.g. 'A' or 'B' if 'A' is missing)
        alts = set()
        p.cmd.iterate("mymodel", "alts.add(alt)", space={"alts": alts})
        alts.discard('')
        best_alt = sorted(list(alts))[0] if alts else ''
        if best_alt:
            p.cmd.remove(f"not alt ''+{best_alt}")
        else:
            p.cmd.remove("not alt ''")
        p.cmd.alter("all", "alt=''")
        ############################
        clean_states(p)
        clean_chains(p, "all", obj="mymodel")
        p.cmd.save(filename=outfile, selection="mymodel", state=-1)

def clean_states(p):
    objlist_ori = set(p.cmd.get_object_list())
    for obj in objlist_ori:
        p.cmd.split_states(obj)
        p.cmd.delete(obj)
        p.cmd.set_name(f"{obj}_0001", obj)
    objlist = p.cmd.get_object_list()
    for obj in objlist:
        if obj not in objlist_ori:
            p.cmd.delete(obj)

def clean_chains(p, sel, obj, nres_per_chain_min=3):
    chains = p.cmd.get_chains(f"{sel} and {obj}")
    for chain in chains:
        nres = p.cmd.select(
            f"{sel} and polymer.protein and chain '{chain}' and name CA and present and {obj}"
        )
        if nres <= nres_per_chain_min:
            p.cmd.remove(f"chain '{chain}' and polymer.protein and {obj}")

if __name__ == "__main__":
    import doctest
    import sys

    @app.command()
    def test():
        """
        Test the code
        """
        doctest.testmod(
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF
        )

    @app.command()
    def test_func(func:str):
        """
        Test the given function
        """
        print(f"Testing {func}")
        f = getattr(sys.modules[__name__], func)
        doctest.run_docstring_examples(
            f,
            globals(),
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF,
        )

    app()
