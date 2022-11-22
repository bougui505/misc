# Running pymol from a jupyter-notebook

See: https://pymolwiki.org/index.php/Jupyter

```python
# open a PyMOL window
import sys
import pymol
_stdouterr = sys.stdout, sys.stderr
pymol.finish_launching(['pymol', '-q'])
sys.stdout, sys.stderr = _stdouterr
```
