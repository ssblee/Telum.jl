# Contraction plan: larger-side prepared RMT buffer

## Goal

Reduce peak memory in `src/contract.jl` by avoiding persistent storage of every
prepared/permuted RMT for both input tensors.

The current contraction path uses `permuted_rmts1` and `permuted_rmts2` as
sector-indexed caches. Once a source sector is needed, its RMT is permuted into
the contraction layout `(free..., contracted..., OM...)`, reshaped to a
rank-3 prepared RMT, and retained until the end of `contract`.

The proposed optimization is sound if it keeps one side cached and streams the
larger-memory side through a reusable buffer:

1. At the start of `contract`, compute the total prepared RMT size for each
   input tensor.
2. If the first tensor is larger, allocate one reusable buffer large enough for
   the largest prepared RMT on the first tensor side.
3. Whenever a first-side prepared RMT is needed, write its permuted data into
   that buffer, use it for all calculations that need that sector, then reuse
   the buffer for the next first-side sector. Apply the symmetric strategy when
   the second tensor is larger.
4. Keep the smaller side in the current sector cache so repeated accesses do not
   repeatedly permute it.

This should preserve public behavior and result sector layout. It should be an
internal memory optimization only.

## Detailed implementation plan

### 1. Measure active prepared RMT sizes

Add a small helper that scans active source sectors and computes:

- `total_len`: sum of prepared dense element counts over all nonzero sectors;
- `max_len`: maximum prepared dense element count over all nonzero sectors.

Use `sector_slots(q)` and skip `is_sector_zero(q, sector)`. Do not use
`nsectors(q)` in the hot scan. The helper should use `_rmt_layout_sizes` or
`_rmt_contract_dims` with the same `perm`, `NF`, `CN`, and `N` used by
contraction, so the count matches the rank-3 prepared representation:

```julia
fdim * cdim * odim
```

Do not stream a side whose sector RMT storage is `DiagRMT`. Diagonal sectors are
small and have specialized contraction paths, so this helper is only used for
streamable dense-side capacity planning.

### 2. Choose cache side and stream side

After `perm1` and `perm2` are known, first decide whether each side is
streamable:

- a `DiagRMT` side is not streamable;
- a side whose processed dense RMT can be reused without allocating separate
  storage is not streamable.

For a concrete `TLArray`, the effective permutation is the contraction
permutation `perm1` or `perm2`. For a `TLArrayView`, the effective permutation
is the composition of the view permutation and the contraction permutation, the
same mapping currently used by `prepared_sector_rmt`:

```julia
source_perm = ntuple(pos -> perm[pos] <= QD ? q.perm[perm[pos]] : perm[pos], Val(RD))
```

Use identity comparison on this effective RMT permutation, not only on the
required contraction permutation. A concrete `TLArray` or real, non-conjugating
`TLArrayView` with an identity effective permutation is not streamable, because
there is no separate processed dense RMT allocation to eliminate. A
`TLArrayView` with `q.conj == true` and complex RMT element type is streamable
even when the effective permutation is identity and the scale is one: the
processed prepared RMT must contain conjugated data and therefore needs separate
memory unless it is left as a lazy wrapper, which the streamed path must not do.

Then choose the streaming side:

```julia
can_stream1 = _can_stream_prepared_rmt(q1, perm1, Val(nf1), Val(CN), Val(N))
can_stream2 = _can_stream_prepared_rmt(q2, perm2, Val(nf2), Val(CN), Val(N))
total1, max1 = can_stream1 ? _prepared_rmt_storage_size(q1, perm1, Val(nf1), Val(CN), Val(N)) : (0, 0)
total2, max2 = can_stream2 ? _prepared_rmt_storage_size(q2, perm2, Val(nf2), Val(CN), Val(N)) : (0, 0)

streaming_side =
    can_stream1 && !can_stream2 ? 1 :
    can_stream2 && !can_stream1 ? 2 :
    !can_stream1 && !can_stream2 ? 0 :
    total1 > total2 || (total1 == total2 && max1 >= max2) ? 1 : 2
```

If `streaming_side == 1`:

- keep `permuted_rmts2` as the persistent smaller-side cache;
- replace persistent first-side storage with a reusable first-side work buffer.

If `streaming_side == 2`:

- keep `permuted_rmts1` as the persistent smaller-side cache;
- stream the second side.

If `streaming_side == 0`, do not stream either side. Use the existing
sector-indexed prepared-RMT caches for both tensors.

If exactly one side is not streamable because it is `DiagRMT` or because its
processed dense RMT can be reused without separate storage, stream the other
side. If both sides are not streamable, set `streaming_side = 0`.

When both sides are streamable and the total sizes are equal, compare `max1` and
`max2`. Cache the side with the smaller maximum prepared RMT size and stream the
side with the larger maximum. If both total and maximum sizes are equal, stream
the first tensor and cache the second.

### 3. Enter a streaming-side function barrier

After choosing the streaming side, call a helper with the side encoded in the
type domain:

```julia
_contract_with_streaming_side(..., Val(streaming_side))
```

Use `streaming_side == 1` for the first tensor, `streaming_side == 2` for the
second tensor, and `streaming_side == 0` for the no-streaming fallback. Inside
this barrier, the compiler can see which side is streamed and which side is
cached.

The concrete working storage should then be selected locally:

- `streaming_side == 1`: one large reusable dense buffer for the first tensor
  and the existing typed sector-indexed prepared-RMT vector for the second;
- `streaming_side == 2`: one large reusable dense buffer for the second tensor
  and the existing typed sector-indexed prepared-RMT vector for the first;
- `streaming_side == 0`: existing typed sector-indexed prepared-RMT vectors for
  both tensors.

This avoids carrying both storage layouts through the outer `contract` method.
It also keeps `PermutedRMT1` and `PermutedRMT2` available at compile time for
dispatch into `_rmt_contract_temp_len` and `_contract_RMT_pair_into!`.

### 4. Add a buffer-writing prepared RMT path

Introduce a function-level helper, not a new storage type:

```julia
_prepare_sector_rmt_into!(buffer, q, sector, perm, Val(NF), Val(CN), Val(N))
```

For dense `Array` sector storage:

- obtain `rmt, alpha = sector_rmt_with_scale(q, sector)`;
- permute into `buffer[1:fdim*cdim*odim]`;
- return a rank-3 `reshape(view(buffer, 1:len), (fdim, cdim, odim))` plus
  `alpha`.

The actual permutation should use `_hptt_permutedims!(dest, A, perm)` and write
into the existing destination buffer. Use an allocating `permutedims` fallback
only for cases where destination-buffer HPTT cannot be used. Effective identity
permutation is usually handled by not streaming that side, so this helper is
primarily for nontrivial effective permutations. The exception is a complex
conjugating `TLArrayView`: it may need to write copied-and-conjugated data into
the buffer even when the effective permutation is identity.

For `TLArrayView`, preserve the current `prepared_sector_rmt` semantics:

- map `perm` through `q.perm`;
- multiply by `q.scale`;
- for `q.conj == true` with complex RMT element type, first copy or
  destination-permute the source data into the reusable buffer, then conjugate
  that buffered data in place. The source RMT must never be modified.

When streaming a `TLArrayView`, store the concrete prepared RMT in the buffer:
the RMT must already be effectively permuted, scaled, and conjugated. Do not
return a lazy conjugated wrapper or defer the scale into later pair contraction
state for the streamed side.

The returned streamed dense prepared RMT may be a reshaped view into the
reusable buffer, for example `reshape(view(buffer, 1:len), (fdim, cdim, odim))`.
Treat this as a dense `AbstractArray{T,3}`/`SubArray` input in the pair
contraction methods rather than introducing a wrapper type.

Do not stream `DiagRMT`. If a side stores diagonal RMTs, keep the existing
cached `DiagRMT{T,3}` prepared representation. If both sides are nonstreamable,
`streaming_side == 0` keeps both prepared-RMT caches.

### 5. Replace per-output sector-pair storage with a sorted work list

Instead of keeping `prepared_sector_pairs` as a vector indexed only by output
sector, build one flat work list after output intervals and QR/factor data are
known:

```julia
work_items = Vector{Tuple{Int, Int, Int}}(...)
```

Each entry is:

```julia
(idx1, idx2, out_pos)
```

where `idx1` is the first tensor sector, `idx2` is the second tensor sector, and
`out_pos` is the resulting output sector. If the factor arrays remain stored
per output sector, add the local pair/factor position as a fourth field:

```julia
(idx1, idx2, out_pos, factor_pos)
```

or maintain an aligned vector containing the corresponding factor-array index.
The key point is that the final contraction loop should be able to recover the
result sector and the correct factor array from each work item without rebuilding
`sector_pairs`.

Sort the work list by the streamed side when streaming is enabled:

```julia
if streaming_side == 1
    sort!(work_items; by = item -> item[1], alg = MergeSort)
elseif streaming_side == 2
    sort!(work_items; by = item -> item[2], alg = MergeSort)
end
```

Use stable sorting so pair order within equal streamed-sector runs remains
predictable. Reordering across different streamed sectors only changes
floating-point accumulation order for output sectors that receive contributions
from multiple streamed sectors. For `streaming_side == 0`, no streamed-sector
ordering is needed.

### 6. Precompute metadata and allocate result RMTs before payload work

Prepare all non-payload metadata before the final RMT contraction loop:

- output q-label columns;
- result w-matrix metadata and `result_wmatdata`;
- prepared QR/factor arrays for every valid output sector;
- cached-side prepared RMTs for every cached-side sector that appears in the
  work list;
- result RMT arrays for every valid output sector;
- the flat, sorted work list.

Fill the cached-side prepared-RMT vector sequentially before the final streaming
loop. The streamed side is not stored sector-by-sector; it is prepared into the
single reusable buffer one streamed-sector run at a time. For
`streaming_side == 0`, fill both prepared-RMT vectors and use the current
cache-based contraction loop.

Allocating result RMTs up front makes the final streaming loop simple: each work
item writes into `result_RMTs[out_pos]` and uses a per-output `first_pair` flag
or contribution counter to choose `beta = zero(RT)` for the first contribution
and `beta = one(RT)` afterward.

### 7. Compute result RMTs by streamed-sector runs

The final payload loop should iterate through runs of equal streamed-sector
index in the sorted work list.

When `streaming_side == 1`:

1. Prepare `q1[idx1]` once into the reusable streamed-side buffer.
2. For every work item in that `idx1` run, fetch or prepare cached
   `q2[idx2]`.
3. Recover `out_pos` and the matching factor array.
4. Accumulate into `result_RMTs[out_pos]`.
5. Move to the next `idx1` run and reuse the same buffer.

When `streaming_side == 2`, use the symmetric loop: prepare `q2[idx2]` once per
run and fetch or prepare cached `q1[idx1]`.

When `streaming_side == 0`, skip streamed-sector runs and contract from the two
prepared-RMT caches. This is the required path when both sides are
nonstreamable, for example both are diagonal or both have effective identity RMT
permutations.

This gives the streamed side a single live prepared dense buffer while still
allowing the cached side to reuse prepared sectors across many streamed-sector
runs.

With this scheme, streamed-side recomputation should not occur in the final
payload loop: each streamed sector is permuted, every work item involving that
prepared RMT is completed, and only then is the buffer reused for the next
streamed sector.

Keep this final output-sector contraction loop serial. Sector sizes vary
substantially, so dividing output-sector work evenly across threads is not
expected to be reliable. The single streamed buffer is therefore part of the
intended serial execution model, not a temporary limitation to remove in this
refactor.

### 8. Fold temp-size calculation into the same access policy

The current prepass computes `max_temp_len` by preparing both sides through
`_cached_prepared_sector_rmt!`. That would defeat the memory optimization.

Replace that prepass with dimension-only helpers:

- get `(fdim, cdim, o1dim)` from `_rmt_contract_dims(q1 sector, perm1, ...)`;
- get `(gdim, cdim, o2dim)` from `_rmt_contract_dims(q2 sector, perm2, ...)`;
- call `_rmt_contract_temp_len(PermutedRMT1, PermutedRMT2, ...)`.

This keeps the contraction temporary sizing exact without materializing any
prepared RMTs during planning.

### 9. Keep result RMT allocation ahead of payload work

It is acceptable, and probably simpler, to allocate each result RMT before
processing its sector pairs. This matches the current
`_contract_prepared_compress_sector` shape:

```julia
result_data = Array{RT, RD_out}(undef, result_shape)
result_view = reshape(result_data, fdim, gdim, rdim)
```

Track whether each result RMT has received at least one contribution with a
`BitVector` whose length is the number of result RMTs:

```julia
result_has_contribution = falses(res_nsectors)
```

For each work item, map its output sector to `result_pos` and choose:

```julia
beta = result_has_contribution[result_pos] ? one(RT) : zero(RT)
```

After `_contract_RMT_pair_into!` runs for that result sector, set:

```julia
result_has_contribution[result_pos] = true
```

This keeps `result_data = Array{RT, RD_out}(undef, result_shape)` valid without
requiring the first contribution for an output sector to appear at any
particular pair index. That matters once the flat work list is sorted by the
streamed source sector rather than grouped by output sector.

Drop impossible output sectors before allocating final result metadata. If an
output sector is discovered to be exactly zero only after dense contraction,
assign a zero RMT array for that result position instead of compacting
`result_qlabels`, `result_wmatinfo`, `result_wmatdata`, and `result_RMTs`.
Such late zero payloads should be rare, and retaining them keeps result-side
metadata aligned without a final compaction pass.

### 10. Preserve the smaller-side cache

Keep a typed sector-indexed cache for the smaller side:

```julia
permuted_rmts_small = Vector{Tuple{PermutedRMTSmall, Tsmall}}(undef, sector_count(qsmall))
```

Fill it sequentially for the cached-side sectors referenced by `work_items`.
This keeps repeated smaller-side sectors cheap during the final streamed loop
while bounding larger-side memory to:

```julia
max_larger_prepared_rmt_len + cached_smaller_prepared_rmt_total + result/temp
```

### 11. Validate behavior and performance

Required checks:

- run the contraction tests that compare against sparse materialization;
- run the DMRG test required for internal contraction refactors:

```julia
julia --project=Telum Telum/test/dmrg_runtests.jl
```

- inspect `@code_warntype` for the new buffer-writing helper and the modified
  output-sector merge helper;
- benchmark representative contractions and report peak allocation change;
- verify the DMRG final energies remain:
  `-82.37493703992092` for `Nkeep=20` and `-82.39703156828021` for `Nkeep=50`,
  up to numerical precision.

## Expected caveats

### Destination-buffer permutation is required for the intended memory profile

The main win is reducing the amount of prepared RMT data retained at one time.
Use `_hptt_permutedims!(dest, rmt, perm)` for streamed dense sectors unless it
cannot be used for the sector's element type or array representation.

If the effective RMT permutation is the identity, usually do not stream that
side. There is no permutation allocation to eliminate, and returning or caching
the original prepared RMT is cheaper than copying it into the streaming buffer.
The exception is a complex `TLArrayView` with `q.conj == true`: the processed
RMT must contain conjugated values, so streaming can still replace a persistent
materialized/conjugated prepared RMT with one reusable buffer. In that case,
copy or destination-permute the source into the buffer first, then conjugate the
buffered values in place. This preserves the original RMT and avoids allocating
a second dense array just for conjugation.

If the implementation falls back to allocating `permutedims(rmt, perm)` and then
copies into the reusable streamed-side buffer, the old temporary permutation
allocation still exists for that streamed sector. That fallback is acceptable
only when destination-buffer HPTT is unavailable; benchmark reports should make
clear when the fallback path is used.

The destination-buffer HPTT helpers should cover the same dense element types as
the current `_hptt_permutedims` methods: `Float64`, `Float32`, `ComplexF64`, and
`ComplexF32`. The real and complex HPTT call signatures differ, so keep these as
small concrete methods rather than a runtime element-type branch in hot code.

This is not a separate optional optimization for the dense streamed path. The
streaming scheme is intended to be: permute one streamed RMT into the reusable
buffer, complete every contraction involving it, then overwrite the buffer with
the next streamed RMT.

### Accumulation order changes are accepted

Sorting the flat work list by streamed sector changes the order in which some
output RMTs receive contributions. The mathematical result is the same, but
floating-point addition is not associative, so exact bitwise equality should not
be expected for output sectors with multiple contributing pairs.

This is inevitable for the streamed-sector run scheme and is accepted as part of
the refactor.

Tests should compare with numerical tolerances, as the existing sparse
equivalence and DMRG checks do. If a test currently assumes bitwise equality for
multi-pair contraction output, that test is too strict for this refactor.

For reproducibility, use stable sorting (`MergeSort`) and deterministic
tie-breaking inside equal streamed-sector runs. This keeps results repeatable
for a fixed thread count and work-list construction order, even if the exact
summation order differs from the old implementation.
