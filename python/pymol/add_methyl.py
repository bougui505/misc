#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-03-31

import typer
import pymol2

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

def trim_nterm_hydrogen(pymolContext, selection):
    h_sel = f"(neighbor ({selection})) and elem H"
    
    h_list = []
    pymolContext.cmd.iterate(h_sel, "h_list.append(index)", space={'h_list': h_list})
    h_count = len(h_list)
    
    print(f"Found {h_count} hydrogens on N-terminus")
    
    if h_count == 3:
        # We remove the first one found in the index list
        target_h = f"index {h_list[0]}"
        pymolContext.cmd.remove(target_h)
        print(f"Removed 1 hydrogen ({target_h}). New state: NH2.")
    else:
        print("Hydrogen count is not 3; no action taken.")

@app.command()
def main(
        pdb:str = typer.Option(..., help="Input PDB file"),
        chain:str = typer.Option(..., help="Chain identifier"),
        outfilename:str = typer.Option(..., help="Output PDB file name"),
        selection:str = typer.Option("polymer.protein", help="Selection string for the protein"),
        ):
    with pymol2.PyMOL() as pm:
        pm.cmd.load(pdb, "struct")
        # 1. Identify the N-terminal Nitrogen (usually the N of the first residue)
        # We sort by residue index to find the lowest one
        residues = []
        pm.cmd.iterate(f"{selection} and chain {chain} and name N", "residues.append(resi)", space={'residues': residues})
        nterm_resi = sorted(residues, key=int)[0]
        nterm_selection = f"chain {chain} and resi {nterm_resi} and name N"
        trim_nterm_hydrogen(pm, selection=nterm_selection)

        pm.cmd.edit(nterm_selection)
        pm.cmd.attach("C", 1, 1)

        pm.cmd.h_add() # Add hydrogens to the new Carbon
        # pm.cmd.clean(f"chain {chain} and resi {nterm_resi}") # Local energy minimization
        pm.cmd.save(outfilename)



if __name__ == "__main__":

    app()

