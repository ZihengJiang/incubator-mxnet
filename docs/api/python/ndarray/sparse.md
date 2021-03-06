<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Sparse NDArray API

```eval_rst
.. currentmodule:: mxnet.ndarray.sparse
```

## Overview

This document lists the routines of the *n*-dimensional sparse array package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.ndarray.sparse
```

The `CSRNDArray` and `RowSparseNDArray` API, defined in the `ndarray.sparse` package, provides
imperative sparse tensor operations.

An `CSRNDArray` inherits from `NDArray`, and represents a two-dimensional, fixed-size array in compressed sparse row format.

```python
>>> x = mx.nd.array([[1, 0], [0, 0], [2, 3]])
>>> csr = x.tostype('csr')
>>> type(csr)
<class 'mxnet.ndarray.sparse.CSRNDArray'>
>>> csr.shape
(3, 2)
>>> csr.data.asnumpy()
array([ 1.  2.  3.], dtype=float32)
>>> csr.indices.asnumpy()
array([0, 0, 1])
>>> csr.indptr.asnumpy()
array([0, 1, 1, 3])
>>> csr.stype
'csr'
```

A detailed tutorial is available at
[CSRNDArray - NDArray in Compressed Sparse Row Storage Format](https://mxnet.incubator.apache.org/versions/master/tutorials/sparse/csr.html).
<br>

An `RowSparseNDArray` inherits from `NDArray`, and represents a multi-dimensional, fixed-size array in row sparse format.

```python
>>> x = mx.nd.array([[1, 0], [0, 0], [2, 3]])
>>> row_sparse = x.tostype('row_sparse')
>>> type(row_sparse)
<class 'mxnet.ndarray.sparse.RowSparseNDArray'>
>>> row_sparse.data.asnumpy()
array([[ 1.  0.],
       [ 2.  3.]], dtype=float32)
>>> row_sparse.indices.asnumpy()
array([0, 2])
>>> row_sparse.stype
'row_sparse'
```

A detailed tutorial is available at
[RowSparseNDArray - NDArray for Sparse Gradient Updates](https://mxnet.incubator.apache.org/versions/master/tutorials/sparse/row_sparse.html).
<br><br>

```eval_rst

.. note:: ``mxnet.ndarray.sparse`` is similar to ``mxnet.ndarray`` in some aspects. But the differences are not negligible. For instance:

   - Only a subset of operators in ``mxnet.ndarray`` have efficient sparse implementations in ``mxnet.ndarray.sparse``.
   - If an operator do not occur in the ``mxnet.ndarray.sparse`` namespace, that means the operator does not have an efficient sparse implementation yet. If sparse inputs are passed to such an operator, it will convert inputs to the dense format and fallback to the already available dense implementation.
   - The storage types (``stype``) of sparse operators' outputs depend on the storage types of inputs.
     By default the operators not available in ``mxnet.ndarray.sparse`` infer "default" (dense) storage type for outputs.
     Please refer to the [API Reference](#api-reference) section for further details on specific operators.

.. note:: ``mxnet.ndarray.sparse.CSRNDArray`` is similar to ``scipy.sparse.csr_matrix`` in some aspects. But they differ in a few aspects:

   - In MXNet the column indices (``CSRNDArray.indices``) for a given row are expected to be **sorted in ascending order**.
     Duplicate column entries for the same row are not allowed.
   - ``CSRNDArray.data``, ``CSRNDArray.indices`` and ``CSRNDArray.indptr`` always create deep copies, while it's not the case in ``scipy.sparse.csr_matrix``.

```

In the rest of this document, we first overview the methods provided by the
`ndarray.sparse.CSRNDArray` class and the `ndarray.sparse.RowSparseNDArray` class,
and then list other routines provided by the `ndarray.sparse` package.

The `ndarray.sparse` package provides several classes:

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray
    RowSparseNDArray
```

We summarize the interface for each class in the following sections.

## The `CSRNDArray` class

### Array attributes

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.shape
    CSRNDArray.context
    CSRNDArray.dtype
    CSRNDArray.stype
    CSRNDArray.data
    CSRNDArray.indices
    CSRNDArray.indptr
```

### Array conversion

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.copy
    CSRNDArray.copyto
    CSRNDArray.as_in_context
    CSRNDArray.asscipy
    CSRNDArray.asnumpy
    CSRNDArray.asscalar
    CSRNDArray.astype
    CSRNDArray.tostype
```

### Array inspection

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.check_format
```

### Array creation

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.zeros_like
```

### Array reduction

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.sum
    CSRNDArray.mean
    CSRNDArray.norm
```

### Array rounding

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.round
    CSRNDArray.rint
    CSRNDArray.fix
    CSRNDArray.floor
    CSRNDArray.ceil
    CSRNDArray.trunc
```

### Trigonometric functions

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.sin
    CSRNDArray.tan
    CSRNDArray.arcsin
    CSRNDArray.arctan
    CSRNDArray.degrees
    CSRNDArray.radians
```

### Hyperbolic functions

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.sinh
    CSRNDArray.tanh
    CSRNDArray.arcsinh
    CSRNDArray.arctanh
```

### Exponents and logarithms

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.expm1
    CSRNDArray.log1p
```

### Powers

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.sqrt
    CSRNDArray.square
```

### Joining arrays

```eval_rst
.. autosummary::
    :nosignatures:

    concat
```

### Indexing

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.__getitem__
    CSRNDArray.__setitem__
    CSRNDArray.slice
```

### Miscellaneous

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.abs
    CSRNDArray.clip
    CSRNDArray.sign
```

### Lazy evaluation

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.wait_to_read
```

## The `RowSparseNDArray` class

### Array attributes

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.shape
    RowSparseNDArray.context
    RowSparseNDArray.dtype
    RowSparseNDArray.stype
    RowSparseNDArray.data
    RowSparseNDArray.indices
```

### Array conversion

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.copy
    RowSparseNDArray.copyto
    RowSparseNDArray.as_in_context
    RowSparseNDArray.asnumpy
    RowSparseNDArray.asscalar
    RowSparseNDArray.astype
    RowSparseNDArray.tostype
```

### Array inspection

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.check_format
```

### Array creation

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.zeros_like
```

### Array reduction

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.norm
```

### Array rounding

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.round
    RowSparseNDArray.rint
    RowSparseNDArray.fix
    RowSparseNDArray.floor
    RowSparseNDArray.ceil
    RowSparseNDArray.trunc
```

### Trigonometric functions

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.sin
    RowSparseNDArray.tan
    RowSparseNDArray.arcsin
    RowSparseNDArray.arctan
    RowSparseNDArray.degrees
    RowSparseNDArray.radians
```

### Hyperbolic functions

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.sinh
    RowSparseNDArray.tanh
    RowSparseNDArray.arcsinh
    RowSparseNDArray.arctanh
```

### Exponents and logarithms

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.expm1
    RowSparseNDArray.log1p
```

### Powers

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.sqrt
    RowSparseNDArray.square
```

### Indexing

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.__getitem__
    RowSparseNDArray.__setitem__
    RowSparseNDArray.retain
```

### Lazy evaluation

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.wait_to_read
```

### Miscellaneous

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.abs
    RowSparseNDArray.clip
    RowSparseNDArray.sign
```

## Array creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    array
    empty
    zeros
    zeros_like
    csr_matrix
    row_sparse_array
    mxnet.ndarray.load
    mxnet.ndarray.save
```

## Array manipulation routines

### Changing array storage type

```eval_rst
.. autosummary::
    :nosignatures:

    cast_storage
```

### Indexing routines

```eval_rst
.. autosummary::
    :nosignatures:

    slice
    retain
    where
```

## Mathematical functions

### Arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    elemwise_add
    elemwise_sub
    elemwise_mul
    broadcast_add
    broadcast_sub
    broadcast_mul
    broadcast_div
    negative
    dot
    add_n
```

### Trigonometric functions

```eval_rst
.. autosummary::
    :nosignatures:

    sin
    tan
    arcsin
    arctan
    degrees
    radians
```

### Hyperbolic functions

```eval_rst
.. autosummary::
    :nosignatures:

    sinh
    tanh
    arcsinh
    arctanh
```

### Reduce functions

```eval_rst
.. autosummary::
    :nosignatures:

    sum
    mean
    norm
```

### Rounding

```eval_rst
.. autosummary::
    :nosignatures:

    round
    rint
    fix
    floor
    ceil
    trunc
```

### Exponents and logarithms

```eval_rst
.. autosummary::
    :nosignatures:

    expm1
    log1p
```

### Powers

```eval_rst
.. autosummary::
    :nosignatures:

    sqrt
    square
```

### Miscellaneous

```eval_rst
.. autosummary::
    :nosignatures:

    abs
    sign
```

## Neural network

### Updater

```eval_rst
.. autosummary::
    :nosignatures:

    sgd_update
    sgd_mom_update
    adam_update
    adagrad_update
```

### More

```eval_rst
.. autosummary::
    :nosignatures:

    make_loss
    stop_gradient
    Embedding
    LinearRegressionOutput
    LogisticRegressionOutput
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. autoclass:: mxnet.ndarray.sparse.CSRNDArray
    :members: shape, context, dtype, stype, data, indices, indptr, copy, copyto, as_in_context, asscipy, asnumpy, asscalar, astype, tostype, slice, wait_to_read, zeros_like, round, rint, fix, floor, ceil, trunc, sin, tan, arcsin, arctan, degrees, radians, sinh, tanh, arcsinh, arctanh, expm1, log1p, sqrt, square, __neg__, sum, mean, norm, square, __getitem__, __setitem__, check_format, abs, clip, sign

.. autoclass:: mxnet.ndarray.sparse.RowSparseNDArray
    :members: shape, context, dtype, stype, data, indices, copy, copyto, as_in_context, asnumpy, asscalar, astype, tostype, wait_to_read, zeros_like, round, rint, fix, floor, ceil, trunc, sin, tan, arcsin, arctan, degrees, radians, sinh, tanh, arcsinh, arctanh, expm1, log1p, sqrt, square, norm, __getitem__, __setitem__, check_format, retain, abs, clip, sign

.. automodule:: mxnet.ndarray.sparse
    :members:
    :special-members:
    :exclude-members: BaseSparseNDArray, RowSparseNDArray, CSRNDArray

.. automodule:: mxnet.ndarray.sparse
    :noindex:
    :members: array, zeros, empty

.. automodule:: mxnet.ndarray
    :members: load, save

```

<script>auto_index("api-reference");</script>
