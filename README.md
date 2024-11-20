## MAP Optimization Results

================================================================================
Parallel Accelerator Optimizing: Function tensor_map.<locals>._map, /path/to/fast_ops.py (163)
================================================================================

### Parallel Loop Listing for Function `_map`:
---------------------------------------------------------------------|loop #ID
def _map(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    in_storage: Storage,
    in_shape: Shape,
    in_strides: Strides,
) -> None:
    """Applies a function to each element of a tensor."""
    identical = np.array_equal(in_strides, out_strides)

    for i in prange(len(out)): --------------------------------------| #2
        if identical:
            out[i] = fn(in_storage[i])
        else:
            out_index = np.zeros(MAX_DIMS, np.int16) ---------------| #0
            in_index = np.zeros(MAX_DIMS, np.int16) ----------------| #1
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

### Loop Nest Optimization:
- +--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop

### Allocation Hoisting:
Memory allocation hoisted for `out_index` and `in_index`.

---

## ZIP Optimization Results

================================================================================
Parallel Accelerator Optimizing: Function tensor_zip.<locals>._zip, /path/to/fast_ops.py (229)
================================================================================

### Parallel Loop Listing for Function `_zip`:
---------------------------------------------------------------------|loop #ID
def _zip(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """Performs element-wise operations on two tensors."""
    identical = np.array_equal(a_strides, out_strides) and np.array_equal(b_strides, out_strides)
    identical = identical and np.array_equal(a_shape, b_shape)

    for i in prange(len(out)): --------------------------------------| #6
        if identical:
            out[i] = fn(a_storage[i], b_storage[i])
        else:
            out_index = np.zeros(MAX_DIMS, np.int32) ---------------| #3
            a_index = np.zeros(MAX_DIMS, np.int32) -----------------| #4
            b_index = np.zeros(MAX_DIMS, np.int32) -----------------| #5
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

### Loop Nest Optimization:
- +--6 is a parallel loop
   +--3, +--4, +--5 --> rewritten as serial loops

### Allocation Hoisting:
Memory allocation hoisted for `out_index`, `a_index`, and `b_index`.

---

## REDUCE Optimization Results

================================================================================
Parallel Accelerator Optimizing: Function tensor_reduce.<locals>._reduce, /path/to/fast_ops.py (301)
================================================================================

### Parallel Loop Listing for Function `_reduce`:
---------------------------------------------------------------------|loop #ID
def _reduce(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    reduce_dim: int,
) -> None:
    """Performs tensor reduction operation along a specified dimension."""
    reduce_size = a_shape[reduce_dim]

    for i in prange(len(out)): --------------------------------------| #9
        out_index = np.zeros(MAX_DIMS, np.int32) --------------------| #7
        to_index(i, out_shape, out_index)
        o = index_to_position(out_index, out_strides)
        for s in prange(reduce_size): -------------------------------| #8
            out_index[reduce_dim] = s
            j = index_to_position(out_index, a_strides)
            out[o] = fn(out[o], a_storage[j])

### Loop Nest Optimization:
- +--9 is a parallel loop
   +--8, +--7 --> rewritten as serial loops

### Allocation Hoisting:
Memory allocation hoisted for `out_index`.

---

## MATRIX MULTIPLY Optimization Results

================================================================================
Parallel Accelerator Optimizing: Function _tensor_matrix_multiply, /path/to/fast_ops.py (345)
================================================================================

### Parallel Loop Listing for Function `_tensor_matrix_multiply`:
---------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function."""
    for n in prange(out_shape[0]): ----------------------------------| #10
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                sum_result = 0
                a_pos = n * a_batch_stride + i * a_strides[1]
                b_pos = n * b_batch_stride + j * b_strides[2]

                for _ in range(a_shape[-1]):
                    sum_result += a_storage[a_pos] * b_storage[b_pos]
                    a_pos += a_strides[2]
                    b_pos += b_strides[1]

                out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                out[out_pos] = sum_result

### Loop Nest Optimization:
Parallel structure is already optimal.

### Allocation Hoisting:
No allocation hoisting found.
