# NNindex

- [API](#api)
    - [`NNindex`](#nnindex)
    - [`Mapping`](#mapping)

Helper class to generate an Annoy index for fast Approximate Nearest neighbors search.
See: https://github.com/spotify/annoy


## API

- [`NNindex`](#nnindex)
- [`Mapping`](#mapping)


### `NNindex`

AnnoyIndex(dim, metric) returns a new index that's read-write and stores vector of dim dimensions.
Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"

```python
>>> np.random.seed(0)
>>> nnindex = NNindex(128)
>>> for name in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
...     nnindex.add(np.random.normal(size=(128)), name)
>>> nnindex.build(10)
>>> nnindex.query('c', k=3)
(['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])
```

Try loading:
```python
>>> del nnindex
>>> nnindex = NNindex(128)
>>> nnindex.annoyfilename
'nnindex/annoy.ann'
>>> nnindex.load()
>>> nnindex.query('c', k=3)
(['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])
```

Hash function testing
```python
>>> del nnindex
>>> nnindex = NNindex(128)
>>> for name in ['abc', 'bcd', 'cde', 'ded', 'efg', 'fgh', 'ghi']:
...     nnindex.add(np.random.normal(size=(128)), name)
>>> nnindex.build(10)
>>> nnindex.query('ghi', k=3)
(['ghi', 'bcd', 'cde'], [0.0, 14.775142669677734, 14.855252265930176])
>>> nnindex.mapping.h5f['name_to_index']['a']['b']['c'].attrs['abc']
0
>>> nnindex.mapping.h5f['index_to_name']['1']['0']['0'].attrs['0']
'abc'
```


### `Mapping`

```python
mapping = Mapping('test.h5')
mapping.add(0, 'toto')
```

```python
>>> mapping.index_to_name(0)
'toto'
>>> mapping.name_to_index('toto')
0
```
