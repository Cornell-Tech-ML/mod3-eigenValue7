#
"""
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (163) 
-------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                                                    | 
        out: Storage,                                                                                                                            | 
        out_shape: Shape,                                                                                                                        | 
        out_strides: Strides,                                                                                                                    | 
        in_storage: Storage,                                                                                                                     | 
        in_shape: Shape,                                                                                                                         | 
        in_strides: Strides,                                                                                                                     | 
    ) -> None:                                                                                                                                   | 
        """                                                                                                                                      | 
        Applies a function to each element of a tensor.                                                                                          | 
                                                                                                                                                 | 
        This function maps a function `fn` to each element of the input tensor `in_storage` and stores the result in the output tensor `out`.    | 
                                                                                                                                                 | 
        Args:                                                                                                                                    | 
        ----                                                                                                                                     | 
            out (Storage): The output tensor where the result of the mapping will be stored.                                                     | 
            out_shape (Shape): The shape of the output tensor.                                                                                   | 
            out_strides (Strides): The strides of the output tensor.                                                                             | 
            in_storage (Storage): The input tensor to be mapped.                                                                                 | 
            in_shape (Shape): The shape of the input tensor.                                                                                     | 
            in_strides (Strides): The strides of the input tensor.                                                                               | 
        """                                                                                                                                      | 
        # TODO: Implement for Task 3.1.                                                                                                          | 
        #raise NotImplementedError("Need to implement for Task 3.1")                                                                             | 
                                                                                                                                                 | 
        identical = np.array_equal(in_strides, out_strides)                                                                                      | 
                                                                                                                                                 | 
        for i in prange(len(out)): --------------------------------------------------------------------------------------------------------------| #2
            if identical:                                                                                                                        | 
                out[i] = fn(in_storage[i])                                                                                                       | 
                                                                                                                                                 | 
            else:                                                                                                                                | 
                out_index = np.zeros(MAX_DIMS, np.int16)-----------------------------------------------------------------------------------------| #0
                in_index = np.zeros(MAX_DIMS, np.int16)------------------------------------------------------------------------------------------| #1
                to_index(i, out_shape, out_index)                                                                                                | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                                                                        | 
                o = index_to_position(out_index, out_strides)                                                                                    | 
                j = index_to_position(in_index, in_strides)                                                                                      | 
                out[o] = fn(in_storage[j])                                                                                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (195) is 
hoisted out of the parallel loop labelled #2 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int16)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (196) is 
hoisted out of the parallel loop labelled #2 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int16)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (229)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (229) 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                                                                                                                              | 
        out: Storage,                                                                                                                                                                                                                                                      | 
        out_shape: Shape,                                                                                                                                                                                                                                                  | 
        out_strides: Strides,                                                                                                                                                                                                                                              | 
        a_storage: Storage,                                                                                                                                                                                                                                                | 
        a_shape: Shape,                                                                                                                                                                                                                                                    | 
        a_strides: Strides,                                                                                                                                                                                                                                                | 
        b_storage: Storage,                                                                                                                                                                                                                                                | 
        b_shape: Shape,                                                                                                                                                                                                                                                    | 
        b_strides: Strides,                                                                                                                                                                                                                                                | 
    ) -> None:                                                                                                                                                                                                                                                             | 
        """                                                                                                                                                                                                                                                                | 
        Performs element-wise operations on two tensors and stores the result in a out tensor.                                                                                                                                                                             | 
                                                                                                                                                                                                                                                                           | 
        This function applies a fn to each pair of elements from two input tensors `a_storage` and `b_storage`, and stores the result in the output tensor `out`. The operation is specified by the function `fn`, which takes two floats as input and returns a float.    | 
                                                                                                                                                                                                                                                                           | 
        Args:                                                                                                                                                                                                                                                              | 
        ----                                                                                                                                                                                                                                                               | 
            out (Storage): The output tensor where the result will be stored.                                                                                                                                                                                              | 
            out_shape (Shape): The shape of the output tensor.                                                                                                                                                                                                             | 
            out_strides (Strides): The strides of the output tensor.                                                                                                                                                                                                       | 
            a_storage (Storage): The first input tensor.                                                                                                                                                                                                                   | 
            a_shape (Shape): The shape of the first input tensor.                                                                                                                                                                                                          | 
            a_strides (Strides): The strides of the first input tensor.                                                                                                                                                                                                    | 
            b_storage (Storage): The second input tensor.                                                                                                                                                                                                                  | 
            b_shape (Shape): The shape of the second input tensor.                                                                                                                                                                                                         | 
            b_strides (Strides): The strides of the second input tensor.                                                                                                                                                                                                   | 
        """                                                                                                                                                                                                                                                                | 
        # TODO: Implement for Task 3.1.                                                                                                                                                                                                                                    | 
        # raise NotImplementedError("Need to implement for Task 3.1")                                                                                                                                                                                                      | 
        identical = np.array_equal(a_strides, out_strides) and np.array_equal(b_strides, out_strides)                                                                                                                                                                      | 
        identical = identical and np.array_equal(a_shape, b_shape)                                                                                                                                                                                                         | 
                                                                                                                                                                                                                                                                           | 
        for i in prange(len(out)):-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #6
            if identical:                                                                                                                                                                                                                                                  | 
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                                                                                                                    | 
            else:                                                                                                                                                                                                                                                          | 
                out_index = np.zeros(MAX_DIMS, np.int32)-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #3
                a_index = np.zeros(MAX_DIMS, np.int32)---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #4
                b_index = np.zeros(MAX_DIMS, np.int32)---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #5
                to_index(i, out_shape, out_index)                                                                                                                                                                                                                          | 
                o = index_to_position(out_index, out_strides)                                                                                                                                                                                                              | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                                                                                                                    | 
                j = index_to_position(a_index, a_strides)                                                                                                                                                                                                                  | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                                                                                                                    | 
                k = index_to_position(b_index, b_strides)                                                                                                                                                                                                                  | 
                out[o] = fn(a_storage[j], b_storage[k])                                                                                                                                                                                                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)


 
Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (266) is 
hoisted out of the parallel loop labelled #6 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (267) is 
hoisted out of the parallel loop labelled #6 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (268) is 
hoisted out of the parallel loop labelled #6 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (301)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (301) 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                                                                                                       | 
        out: Storage,                                                                                                                                                                  | 
        out_shape: Shape,                                                                                                                                                              | 
        out_strides: Strides,                                                                                                                                                          | 
        a_storage: Storage,                                                                                                                                                            | 
        a_shape: Shape,                                                                                                                                                                | 
        a_strides: Strides,                                                                                                                                                            | 
        reduce_dim: int,                                                                                                                                                               | 
    ) -> None:                                                                                                                                                                         | 
        """                                                                                                                                                                            | 
        Performs tensor reduction operation along a specified dimension.                                                                                                               | 
                                                                                                                                                                                       | 
        This function applies a reduction operation to the input tensor `a_storage` along the dimension specified by `reduce_dim`. The result is stored in the output tensor `out`.    | 
                                                                                                                                                                                       | 
        Args:                                                                                                                                                                          | 
        ----                                                                                                                                                                           | 
            out (Storage): The output tensor where the reduction result will be stored.                                                                                                | 
            out_shape (Shape): The shape of the output tensor.                                                                                                                         | 
            out_strides (Strides): The strides of the output tensor.                                                                                                                   | 
            a_storage (Storage): The input tensor to be reduced.                                                                                                                       | 
            a_shape (Shape): The shape of the input tensor.                                                                                                                            | 
            a_strides (Strides): The strides of the input tensor.                                                                                                                      | 
            reduce_dim (int): The dimension along which the reduction operation is performed.                                                                                          | 
                                                                                                                                                                                       | 
        """                                                                                                                                                                            | 
        # TODO: Implement for Task 3.1.                                                                                                                                                | 
        #  raise NotImplementedError("Need to implement for Task 3.1")                                                                                                                 | 
                                                                                                                                                                                       | 
                                                                                                                                                                                       | 
        reduce_size = a_shape[reduce_dim]                                                                                                                                              | 
                                                                                                                                                                                       | 
        for i in prange(len(out)):-----------------------------------------------------------------------------------------------------------------------------------------------------| #9
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------------------------------------------------------------------------------------------------------| #7
            to_index(i, out_shape, out_index)                                                                                                                                          | 
            o = index_to_position(out_index, out_strides)                                                                                                                              | 
            for s in prange(reduce_size):----------------------------------------------------------------------------------------------------------------------------------------------| #8
                out_index[reduce_dim] = s                                                                                                                                              | 
                j = index_to_position(out_index, a_strides)                                                                                                                            | 
                out[o] = fn(out[o], a_storage[j])                                                                                                                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #9, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--9 is a parallel loop
   +--8 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (serial)
   +--7 (serial)


 
Parallel region 0 (loop #9) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (333) is 
hoisted out of the parallel loop labelled #9 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (345)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/takadanana/Desktop/mle/mod3-eigenValue7/minitorch/fast_ops.py (345) 
---------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                 | 
    out: Storage,                                                                            | 
    out_shape: Shape,                                                                        | 
    out_strides: Strides,                                                                    | 
    a_storage: Storage,                                                                      | 
    a_shape: Shape,                                                                          | 
    a_strides: Strides,                                                                      | 
    b_storage: Storage,                                                                      | 
    b_shape: Shape,                                                                          | 
    b_strides: Strides,                                                                      | 
) -> None:                                                                                   | 
    """NUMBA tensor matrix multiply function.                                                | 
                                                                                             | 
    Should work for any tensor shapes that broadcast as long as                              | 
                                                                                             | 
    ```                                                                                      | 
    assert a_shape[-1] == b_shape[-2]                                                        | 
    ```                                                                                      | 
                                                                                             | 
    Optimizations:                                                                           | 
                                                                                             | 
    * Outer loop in parallel                                                                 | 
    * No index buffers or function calls                                                     | 
    * Inner loop should have no global writes, 1 multiply.                                   | 
                                                                                             | 
                                                                                             | 
    Args:                                                                                    | 
    ----                                                                                     | 
        out (Storage): storage for `out` tensor                                              | 
        out_shape (Shape): shape for `out` tensor                                            | 
        out_strides (Strides): strides for `out` tensor                                      | 
        a_storage (Storage): storage for `a` tensor                                          | 
        a_shape (Shape): shape for `a` tensor                                                | 
        a_strides (Strides): strides for `a` tensor                                          | 
        b_storage (Storage): storage for `b` tensor                                          | 
        b_shape (Shape): shape for `b` tensor                                                | 
        b_strides (Strides): strides for `b` tensor                                          | 
                                                                                             | 
    Returns:                                                                                 | 
    -------                                                                                  | 
        None : Fills in `out`                                                                | 
                                                                                             | 
    """                                                                                      | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                   | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                   | 
                                                                                             | 
    # TODO: Implement for Task 3.2.                                                          | 
    #raise NotImplementedError("Need to implement for Task 3.2")                             | 
    assert a_shape[-1] == b_shape[-2], "inner dim has to be same"                            | 
                                                                                             | 
    for n in prange(out_shape[0]):-----------------------------------------------------------| #10
        for i in range(out_shape[1]):  # Rows of a                                           | 
            for j in range(out_shape[2]):  # Columns of b                                    | 
                sum_result = 0                                                               | 
                a_pos = n * a_batch_stride + i * a_strides[1]                                | 
                b_pos = n * b_batch_stride + j * b_strides[2]                                | 
                                                                                             | 
                for _ in range(a_shape[-1]):  # Columns of a and rows of b                   | 
                    sum_result += a_storage[a_pos] * b_storage[b_pos]                        | 
                    a_pos += a_strides[2]                                                    | 
                    b_pos += b_strides[1]                                                    | 
                                                                                             | 
                out_pos = n * out_strides[0] + i * out_strides[1] + j*out_strides[2] # Ba    | 
                out[out_pos] = sum_result                                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None"""
