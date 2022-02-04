from pymol import cmd


# bind a selection's consecutive atoms

def connect_all(sel):
    cmd.unbond(sel, sel)
    n_atom = cmd.count_atoms(sel)
    for index in range(n_atom):
        cmd.bond(f"{sel} and index {index + 1}", f"{sel} and index {index + 2}")


cmd.extend('connect_consecutive', connect_all)
