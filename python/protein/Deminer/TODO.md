# Bugs

```
bougui@ld18-1006> python deminer.py --test_dataset --nviews 1 --batch_size 1 | grep -v 'Detected mmCIF'  ~/source/misc/python/protein/Deminer
Warning: use "from pymol import cmd" instead of "import cmd"
Number of excluded pdb entries: 2604
 PyMOL not running, entering library mode (experimental)
  2%|██▎                                                                                              | 4507/193254 [05:40<7:50:39,  6.68it/s]Traceback (most recent call last):
  File "/c7/home/bougui/anaconda3/lib/python3.7/multiprocessing/queues.py", line 236, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "/c7/home/bougui/anaconda3/lib/python3.7/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
OverflowError: cannot serialize a bytes object larger than 4 GiB
^C
```
