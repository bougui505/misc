# Toy RL

### Overview

This is a toy setting for RL. We aim to solve path finding problems in Mazes. Those mazes are defined in mullerenv.py.
Then a features extractor for state representation is defined in feature extractor.

Finally, two scripts for training and making inference are available :

```
./train_muller.py -n test
./predict_muller.py -n test
```

The available options for the settings of the environment, the model and the learning
are described using -h by the argparse.
The different settings that we have been trying out are summed up in a [ggsheet](https://docs.google.com/spreadsheets/d/1fUFJDnLdW443GwGmxf7Qi0cwseGc_ZVvNcS0T__PuFk/edit?usp=sharing).

As a quick summary, we started by implementing a complicated potential to follow and the aim was
to reach the global minima. Then we settled on a less complicated potential, following a quadratic pattern.
Then we added a discrete option for the actions to be just a choice in the grid (8 possibilities) 
rather than a continuous move (that was then quantized on the grid). Finally, we added a binary option
to remove the continuous rewards with constant ones except for the target cell.

In terms of the possibilities for the states, the basic input is just the local neighborhood.
Then we enrich it with the few former states as in DQN, within the 'history' option.
We also add the traj option that denotes having access to the cells that were previously visited.
Finally the 'move' option denotes the possibility to access the previous direction directly.
This is redundant with the history, but is encoded in a more straightforward way (we expect 
trajectory to be kinda linear, this is a smoothness prior).

In the smallest setting, with no extra info and a very easy target cell, most algorithms 
manage to work ok. However, when elaborating on that, we see unstable learning. For instance,
we sometimes have a good performance that completely degrades through time or no learning at all.
DQN for instance fails with the default batch size... This feels weird because 
the task looks extremely simple.

