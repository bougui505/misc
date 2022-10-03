# DeMiner: Density-Miner

## Known issues

### OverflowError: cannot serialize a bytes object larger than 4 GiB (SOLVED)

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

Install `pyarrow` package with: `pip install pyarrow` (see: https://github.com/facebookresearch/fairseq/issues/2166#issuecomment-632652607)


## struct.error: 'i' format requires -2147483648 <= number <= 2147483647

```
[130] bougui@ld18-1006> python deminer.py --train --print_each 1 --nviews 5 --n_epochs 5 --batchmemcutoff 30000000 | grep -v ExecutiveLoad-Detail
Warning: use "from pymol import cmd" instead of "import cmd"
60656 data removed by exclude_list
 PyMOL not running, entering library mode (experimental)
Traceback (most recent call last):
  File "/c7/home/bougui/anaconda3/lib/python3.7/multiprocessing/queues.py", line 242, in _feed
    send_bytes(obj)
  File "/c7/home/bougui/anaconda3/lib/python3.7/multiprocessing/connection.py", line 200, in send_bytes
    self._send_bytes(m[offset:offset + size])
  File "/c7/home/bougui/anaconda3/lib/python3.7/multiprocessing/connection.py", line 393, in _send_bytes
    header = struct.pack("!i", n)
struct.error: 'i' format requires -2147483648 <= number <= 2147483647
```

At any rate, the original message seems to be a bug of Python < 3.8 handling large objects badly when it communicates between the processing (see: https://discuss.pytorch.org/t/python-multiprocessing-struct-error-i-format-requires-2147483648-number-2147483647/59684/4).

Try to update python to 3.9.2:
```
conda install python=3.9.2
```
