# NNindex

Helper class to generate an Annoy index for fast Approximate Nearest neighbors search.

See: https://github.com/spotify/annoy

```python
>>> np.random.seed(0)
>>> nnindex = NNindex(128)
>>> for name in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
...     nnindex.add(np.random.normal(size=(128)), name)
>>> nnindex.build(10)
>>> nnindex.query('c', k=3)
(['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])
```
