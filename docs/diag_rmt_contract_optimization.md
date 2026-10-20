# DiagRMT prepared-contraction optimization plan

## Goal

Optimize `_contract_prepared_compress_sector` for the common contraction case:

- one prepared RMT is `DiagRMT{T,3}`;
- the other prepared RMT is dense `Array{T,3}`;
- the diagonal prepared RMT has matrix form `axis == (1, 2)`;
- the diagonal side has trivial outer multiplicity, so `size(diag_rmt, 3) == 1`.

This should preserve the existing contraction workflow:

1. build `permuted_rmts1/2`;
2. prepare factor arrays;
3. call `_contract_prepared_compress_sector`;
4. return a dense result RMT.

Do not introduce a new planner or broad trait system for this first
optimization.

## Common case

Prepared RMT layout is:

```julia
A::DiagRMT{T,3}, axis == (1, 2)  # first tensor diagonal
B::Array{T,3}                    # second tensor dense
K::Array{Float64,3}              # factor array
```

or the symmetric case:

```julia
A::Array{T,3}
B::DiagRMT{T,3}, axis == (1, 2)
K::Array{Float64,3}
```

For `axis == (1, 2)`, the diagonal RMT represents a matrix over
`(free_dim, contracted_dim)` and has singleton OM dimension:

```julia
size(D) == (d, d, 1)
D[f, c, 1] != 0 only when f == c
```

Since only one leg of the diagonal tensor is contracted, each output sector
should normally receive at most one sector pair in this case. The implemented
optimized path uses `iszero(beta)` as this guard:

```julia
beta == 0
```

If accumulation is needed (`beta != 0`), the code falls back to the existing
generic diagonal-aware contraction.

## First tensor diagonal

For:

```julia
D::DiagRMT{T1,3}, axis == (1, 2)
B::Array{T2,3}
K::Array{Float64,3}
```

the dense generic contraction is:

```julia
out[f, g, r] = sum(D[f, c, o1] * B[g, c, o2] * K[r, o1, o2]
                   for c, o1, o2)
```

Because `D[f,c,o1]` is nonzero only at `c == f` and `o1 == 1`:

```julia
out[f, g, r] = D.diag[f] * sum(B[g, f, o2] * K[r, 1, o2] for o2)
```

Implementation outline:

1. Assert `D.axis == (1, 2)` and `size(D, 3) == 1`.
2. Use this path only when `beta == 0`; otherwise use the generic fallback.
3. Contract the dense RMT with the factor matrix first and save the intermediate
   result in `contract_temp`.

   For this case `o1dim == 1`, so use the factor slice:

   ```julia
   Kmat = reshape(K, rdim, o2dim)
   ```

   For each contracted/free index on the dense tensor, multiply:

   ```julia
   tmp[g, c, r] = sum(B[g, c, o2] * Kmat[r, o2] for o2)
   ```

   This can be implemented with `mul!` by viewing/reshaping the dense tensor as
   a matrix whose rows are `(g, c)` and whose columns are `o2`.
4. Permute/write the temporary into the output layout with permutation
   `(2, 1, 3)`:

   ```julia
   out[f, g, r] = tmp[g, f, r]
   ```

   because the diagonal enforces `c == f`.
5. Multiply each `f` slice in-place by `D.diag[f]`.

The implementation should avoid expanding `D` to a dense array.

### Temp size for first tensor diagonal

Divide temporary-buffer cases inside `_rmt_contract_temp_len`. For now, only
two paths need nonzero temporary storage:

- dense/dense: existing generic temporary-size logic;
- any diagonal first tensor with dense second tensor.

For the optimized `D.axis == (1, 2)` path, the temporary stores `tmp[g, c, r]`,
so its length is:

```julia
size(B, 1) * size(B, 2) * size(K, 1)
```

Reserve this same size for all `(DiagRMT, Array)` cases, even when the prepared
axis form later falls back to the generic diagonal-aware contraction. The buffer
is at most on the order of the largest prepared RMT and avoiding axis-dependent
pre-cache logic keeps the interface simpler.

Every other prepared-RMT type combination should return `0` for now, unless it
routes to the existing dense/dense generic path.

## Second tensor diagonal

For:

```julia
A::Array{T1,3}
D::DiagRMT{T2,3}, axis == (1, 2)
K::Array{Float64,3}
```

the contraction reduces to:

```julia
out[f, g, r] = D.diag[g] * sum(A[f, g, o1] * K[r, o1, 1] for o1)
```

Implementation outline:

1. Assert `D.axis == (1, 2)` and `size(D, 3) == 1`.
2. Use this path only when `beta == 0`; otherwise use the generic fallback.
3. Contract the dense RMT with the factor matrix directly into `out`.

   For this case `o2dim == 1`, so use:

   ```julia
   Kmat = reshape(K, rdim, o1dim)
   ```

   Compute:

   ```julia
   out[f, g, r] = sum(A[f, g, o1] * Kmat[r, o1] for o1)
   ```

   using `mul!` over the dense RMT OM dimension.
4. Multiply each `g` slice in-place by `D.diag[g]`.

No additional temporary is needed for the second-tensor-diagonal optimized path
because the dense-times-factor product already has the final output layout.

## Dispatch shape

Keep the generic fallback for all other cases:

- both prepared RMTs are dense;
- both prepared RMTs are diagonal;
- diagonal RMT has vectorized form `axis[2] == 0`;
- diagonal RMT has any matrix axis other than `(1, 2)`;
- factor dimensions do not match the singleton-OM assumptions.

Suggested helper split:

Use one pre-cache temp-size interface and divide cases inside it:

```julia
_rmt_contract_temp_len(
    ::Type{AType},
    ::Type{BType},
    fdim::Int,
    gdim::Int,
    cdim::Int,
    o1dim::Int,
    o2dim::Int,
    K::AbstractArray,
) where {AType<:AbstractArray, BType<:AbstractArray}
```
The function should branch internally on `AType` and `BType`:

- `(Array, Array)`: use the existing dense generic temp-size logic;
- `(DiagRMT, Array)`: return `gdim * cdim * size(K, 1)` for `tmp[g,c,r]`,
  regardless of the eventual prepared diagonal axis form;
- all other cases: return `0` for now.

Example shape:

```julia
_rmt_contract_temp_len(::Type{<:Array}, ::Type{<:Array}, dims..., K) =
    _rmt_contract_temp_len_dense(dims..., K)

_rmt_contract_temp_len(::Type{<:DiagRMT}, ::Type{<:Array}, dims..., K) =
    gdim * cdim * size(K, 1)

_rmt_contract_temp_len(::Type{<:Array}, ::Type{<:DiagRMT}, dims..., K) = 0
_rmt_contract_temp_len(::Type{<:DiagRMT}, ::Type{<:DiagRMT}, dims..., K) = 0
```

```julia
_contract_prepared_compress_sector(...)
```

continues to allocate `result_data` and `result_view`, then dispatches each pair
to:

```julia
_contract_RMT_pair_into!(out, A, B, K, temp, threshold, beta)
```

Add optimized overloads:

```julia
_contract_RMT_pair_into!(out, D::DiagRMT, B::Array, K, temp, threshold, beta)
_contract_RMT_pair_into!(out, A::Array, D::DiagRMT, K, temp, threshold, beta)
```

Inside those overloads:

- use the optimized path only for `D.axis == (1, 2)` and `size(D, 3) == 1`;
- use the optimized path only when `iszero(beta)`;
- use the unified `_rmt_contract_temp_len` during `max_temp_len`
  calculation;
- for first-tensor diagonal, use `contract_temp` as `tmp[g,c,r]`, then
  permute/write to `out[f,g,r]` and scale by `D.diag[f]`;
- for second-tensor diagonal, write dense-times-factor directly to `out` and
  scale by `D.diag`;
- otherwise call the existing generic diagonal-aware fallback.

If the `length(sector_pairs) == 1` invariant is easiest to enforce in
`_contract_prepared_compress_sector`, add a small branch before the pair loop:

```julia
if _can_use_diag_dense_single_pair(first_data1, first_data2)
    @assert length(sector_pairs) == 1
end
```

## Validation

Add focused tests comparing against dense materialization:

- first tensor diagonal, `axis == (1, 2)`, dense second tensor;
- dense first tensor, second tensor diagonal, `axis == (1, 2)`;
- verify the optimized path rejects or falls back for `axis[2] == 0`;
- verify multi-pair input continues to be handled by the generic fallback.

Run:

```julia
julia --project=Telum -e 'using Telum, LinearAlgebra; include("Telum/test/common.jl"); test_contract_compress_sector_diag_rmt(); test_contract_diag_rmt_tlarray()'
```

Also inspect:

```julia
@code_warntype Telum._contract_RMT_pair_into!(...)
```

for the new diagonal/dense overloads.
