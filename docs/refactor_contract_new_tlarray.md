# Refactor plan: contraction for field-backed TLArray

## Goal

Refactor `contract` in `src/contract.jl` to use the current field-backed
`TLArray` layout more directly:

```julia
qlabels[leg, sector]
wmats[nonabelian_slot, sector]
RMTs[sector]
```

The first target is the sector matching and X-symbol preparation stage. The RMT
contraction/compression stage should keep the current behavior and structure
until a later optimization pass.

The intended result is fewer small transient allocations before RMT contraction,
less GC pressure in DMRG contractions, and a layout that can later be split by
output sector for multithreading.

## Terminology

This document uses two different q-label levels:

- `product q-label`: the q-label for one physical TLArray leg, containing all
  product-symmetry components. Its type is `QT`.
- `symmetry q-label`: the q-label for one symmetry component only. For
  non-Abelian X-symbol work its type is `NTuple{NZ, Int}`.

The pair table stores product q-labels (`QT`). X-symbol cache keys,
impossible-set keys, and CGT metadata use symmetry q-labels
(`NTuple{NZ, Int}`) after projecting a product q-label to one symmetry index.

## Non-goals

- Do not change tensor semantics or numerical results.
- Do not change public `contract` APIs.
- Do not introduce new structs, wrapper types, traits, or abstraction layers.
- Do not rewrite `_contract_RMT_pair_into!`, `_prepare_compress_sector`,
  `_contract_prepared_compress_sector`, or RMT compression in this phase.
- Do not reconstruct whole CGR objects to recover CGT leg order.
- Do not change TLArray storage format in this phase.
- Do not use out-of-place sorting in the contraction hot path.

## Current problems

- `contract` discovers matching sector pairs and immediately contracts
  X-symbols into per-pair w-matrices. This creates many small arrays and
  dictionaries before the output sectors are known.
- Output sectors are accumulated through dictionaries keyed by free product
  q-labels.
  That is convenient but allocates heavily for large numbers of sector pairs.
- X-symbol cache keys are currently correct and should stay based on
  per-symmetry q-label tuples:
  `(contracted symmetry q-labels, free symmetry q-labels of q1,
  free symmetry q-labels of q2)`.
  The optimization target is to avoid repeated CGT metadata reconstruction, not
  to change cache key semantics.
- CGT metadata is only needed when an X-symbol cache miss calls
  `getNsave_Xsymbol`. It should not be reconstructed for cache hits.
- Rejected fusion channels are discovered pair by pair. Once a non-Abelian
  symmetry returns `nothing` for an output free symmetry q-label tuple, every
  sector pair with the same output free symmetry q-label tuple can be skipped.

## Invariants to preserve

- `_contract_sector_infos` remains the starting point and keeps its current
  sorted-by-contracted-product-q-label behavior for finding overlapping
  contraction keys.
- Output physical leg order remains `(free legs of q1..., free legs of q2...)`.
- The order of output product-q-label sectors is not a semantic requirement.
- Result `qlabels` must be built in the current TLArray field format:
  `Matrix{QT}(undef, QD_out, n_output_sectors)` with
  `qlabels[leg, sector]`.
- CGT stored-leg metadata must follow the TLArray invariant:
  incoming physical legs sorted by symmetry q-label, outgoing physical legs
  sorted by symmetry q-label, and equal symmetry q-labels with equal direction
  keeping physical leg order.
- X-symbol cache sharing must be by symmetry type, not by symmetry index, so
  repeated non-Abelian symmetry types reuse one cache as in the current code.
  The distinct non-Abelian symmetry types, cache-slot mapping, key types, and
  dictionary types are all known from `PS` at compile time.
- X-symbol caches must not use `Any` in their concrete type definitions.
- Abelian symmetries do not occupy `wmats` slots and should not create X-symbol
  work.
- If any non-Abelian symmetry reports `nothing`, the corresponding output
  product-q-label sector is invalid and no RMT contribution should be emitted
  for it.
- The final RMT and w-matrix construction path must remain behavior-compatible
  with the current implementation.
- Every sort in the contraction path must be in-place, for example `sort!` or
  `sortperm!` into a preallocated buffer. Avoid `sort`, `sortperm`, or
  `collect(...); sort` patterns in hot helpers.
- In-place sorting is not sufficient by itself if the sorted input is allocated
  every time. Hot metadata helpers should also avoid per-sector leg-list
  construction with comprehensions such as `Int[...]`; fill reusable buffers or
  use tuple-based helpers where practical.
- New helpers must be type-stable at their call boundaries. Do not introduce
  `Vector{Any}`, `Dict{Any,Any}`, abstractly typed fields, or return values that
  change type across Abelian/non-Abelian branches.
- The "same output free symmetry q-label tuple is impossible" shortcut is an
  explicit fusion invariant at the symmetry-q-label level: for a fixed
  non-Abelian symmetry and fixed resulting free symmetry q-label tuple,
  `getNsave_Xsymbol(...) === nothing` means every source-sector pair with that
  resulting free symmetry q-label tuple is forbidden for that symmetry. Validate
  this against representative mixed and repeated-symmetry-q-label cases.

## Reviewed workflow

### 1. Keep sector-info construction

Keep `_free_legs`, `_contract_sector_infos`, and `_contracted_qlabel_run` as the
front end.

The two sorted sector-info vectors already provide:

- source sector index;
- free physical-leg product q-labels (`QT`);
- contracted physical-leg product q-labels (`QT`).

No change is needed before the two-pointer scan.

### 2. Record overlapping contracted-label runs

Run the existing two-pointer scan over contracted-product-q-label runs once,
without doing X-symbol or RMT work.

For each overlapping contracted key `ct`, record:

```julia
(ct, run1, run2)
```

where `run1` and `run2` are the `UnitRange{Int}` ranges into the sorted
sector-info vectors. Also add:

```julia
length(run1) * length(run2)
```

to `n_possible`.

This pass should only advance through `_contracted_qlabel_run` results. It
should not allocate per-pair data, and `_fill_possible_pairs!` should reuse the
recorded overlap ranges rather than repeating the same two-pointer scan.

### 3. Build and sort the possible-pair table

Allocate one vector of length `n_possible`. Each entry should contain:

```julia
(contracted product qlabels,
 (free product qlabels of q1..., free product qlabels of q2...),
 sector_index_q1,
 sector_index_q2)
```

Use the current free-label tuple types:

```julia
FreeKey1 = NTuple{nf1, QT}
FreeKey2 = NTuple{nf2, QT}
OutKey = Tuple{FreeKey1, FreeKey2}
```

or the flattened equivalent if it is measurably better. Prefer the representation
that keeps sorting type-stable and avoids `Vector{Any}`.

Sort the table in-place by output free product q-labels with `sort!`. This sort
is only to group equal output free product q-labels into contiguous intervals;
the final output sector order is not required to be sorted.

The contracted product q-labels must be carried in each pair-table entry even
though the table is grouped by output free product q-labels. Later X-symbol
lookup projects them to symmetry q-labels for the cache key.

After sorting, record:

- `result_qlabel_sectors::Vector{NTuple{QD_out, QT}}`;
- an interval table giving the contiguous pair-table range for each output
  sector.

The final `qlabels` matrix should be materialized only after invalid sectors
have been filtered or skipped:

```julia
result_qlabels = Matrix{QT}(undef, QD_out, length(result_qlabel_sectors))
```

and filled as `result_qlabels[leg, sector_index]`.

Review note: invalid sectors should be tracked with an explicit validity vector
or invalid marker aligned with the output product-q-label list and interval
table. Do not use swap-and-pop for invalid-sector removal in the final
implementation. Keeping the sector tables index-stable makes later prepared
metadata, direct-fill buffers, and validation easier to reason about. Final
materialization should iterate only valid sectors or compact once at the
materialization boundary.

Type-stability note: the pair-table entry type should be concrete and fixed by
`NF1`, `NF2`, and `QT`. Filling the preallocated vector must not rely on
temporary heterogeneous tuples or abstract containers.

### 4. Prepare CGT input metadata only on cache misses

Do not build a full matrix of:

```julia
((sorted incoming symmetry qlabels),
 (sorted outgoing symmetry qlabels),
 contracted_stored_legs)
```

for every `(nonabelian_slot, sector)` as part of the main workflow. Since the
X-symbol cache key remains:

```julia
(contracted symmetry qlabels,
 free symmetry qlabels of q1,
 free symmetry qlabels of q2)
```

this metadata is not part of the cache key. It is needed only on a cache miss to
call `getNsave_Xsymbol`.

On a cache miss, compute the exact plain-data input needed by
`getNsave_Xsymbol`:

```julia
((sorted incoming symmetry qlabels),
 (sorted outgoing symmetry qlabels),
 contracted_stored_legs)
```

The sorted incoming/outgoing q-labels are symmetry q-labels, not product
q-labels.
The contracted stored legs are computed from the canonical stored-leg order and
the physical contracted legs.

Important details:

- The metadata helper is called for a product-symmetry index `n`, not a distinct
  cache slot. Repeated non-Abelian symmetry types share one X-symbol cache, but
  still have distinct per-index CGT metadata because their product-symmetry
  positions differ.
- The helper must skip Abelian symmetries by construction; it should be called
  only from generated non-Abelian branches.
- Any sorting used to create stored incoming/outgoing leg order must be in-place.
  Reuse the existing `_stored_leg_order` behavior or a helper that fills and
  sorts preallocated leg-index buffers.
- The current `_stored_leg_order` shape uses allocated `Int[...]` vectors before
  `sort!`. That is acceptable as a correctness oracle, but it should not be the
  final hot-path implementation for this refactor.
- If profiling shows repeated cache misses for the same `(n, sector)` metadata
  are significant, add an optional lazy cache of metadata. That cache must be
  per product-symmetry index or otherwise type-stable; do not add a
  `Matrix{Any}` or abstract metadata table.

The cache-miss helper must use the existing `_stored_leg_order` /
`_sector_cgt_metadata` logic or equivalent function-level helpers, not
reconstructed CGR objects.

### 5. Build X-symbol caches without w-matrix contraction

Iterate the sorted possible-pair table by output-sector interval.

For each non-Abelian symmetry:

1. Check whether this output free symmetry q-label tuple is already in that
   symmetry's impossible set. The impossible-set key should be the
   symmetry-projected output free symmetry q-label tuple. Using the full product
   `OutKey` is correct but misses pruning opportunities across different
   Abelian labels.
2. If not impossible, load or reuse the X-symbol with the existing cache key:

```julia
(contracted symmetry qlabels,
 free symmetry qlabels of q1,
 free symmetry qlabels of q2)
```

3. If `getNsave_Xsymbol` returns `nothing`, mark this output sector impossible
   for the symmetry and skip the rest of the interval. Do not save `nothing` in
   the X-symbol cache; failed fusion channels belong only in the impossible set.

Cache layout should preserve current sharing behavior while removing `Any`.
There should be one cache dictionary for each distinct non-Abelian symmetry
type, stored in `caches` as a generated tuple of dictionaries:

```julia
caches::Tuple{CacheDict1, CacheDict2, ...}
```

where every `CacheDictK` is a concrete `Dict` type for one distinct
non-Abelian symmetry type and its symmetry-q-label shape (`NTuple{NZ, Int}` for
that symmetry). The value type should be the raw X-symbol array type, not
`Union{Nothing, ...}`. Repeated non-Abelian symmetry indices of the same symmetry
type must map to the same tuple slot.

Everything needed for this layout is known from the product symmetry type `PS`
at compile time:

- distinct non-Abelian symmetry types in first-occurrence order;
- product-symmetry index `n` to distinct-cache slot;
- the symmetry-q-label type for each cache key;
- the concrete `Dict` type for each cache slot.

Use generated/`Val`-specialized helpers:

```julia
_new_xsym_caches(::Type{PS}) where {PS<:ProductSymm}
_xsym_cache_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
_xsym_cache_for(caches::Tuple, ::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
```

`_new_xsym_caches(PS)` should return the concrete tuple of dictionaries. The
`Val(n)` accessor should index a compile-time constant tuple slot, so the hot
lookup path sees the exact dictionary type for product-symmetry index `n`.

There should also be one impossible set for each distinct non-Abelian symmetry
type, in exactly the same tuple-slot order as `xsym_caches`. For example, if the
X-symbol caches are ordered as `(SU2 cache, SU3 cache)`, then the impossible
sets must be ordered as `(SU2 set, SU3 set)`. The same product-symmetry-index to
distinct-non-Abelian-cache-slot helper should index both tuples.

The cache key is unchanged from the current algorithm and is expressed in
symmetry q-labels. CGT metadata is computed only to call `getNsave_Xsymbol` on a
cache miss.

Do not call `_contract_xsym_wmat` in this phase. Store or retrieve only raw
X-symbol arrays. Abelian symmetries implicitly contribute the trivial factor.

Review note: do not hide different dictionary shapes behind `Vector{Any}`,
`Vector{AbstractDict}`, or a vector with a finite union element type. A generated
tuple is preferred because tuple-slot access with `Val(n)` preserves the exact
dictionary type.

### 6. Produce valid output-sector metadata

During the X-symbol pass, keep only output sectors not rejected by any
non-Abelian symmetry.

For each valid output sector, retain:

- the output physical-leg product-q-label tuple;
- the interval/range of possible sector pairs;
- access to the X-symbol cache entries needed for each pair and symmetry.

The interval should identify source sector pairs in the sorted possible-pair
table. Avoid copying per-sector pair vectors unless the later unchanged RMT path
requires `Vector{Tuple{Int, Int}}`.

Final invalidation behavior:

- when a non-Abelian symmetry returns `nothing`, skip the current interval;
- mark the current output-sector slot invalid in a `Vector{Bool}` or equivalent
  concrete validity table;
- keep output product q-labels, intervals, direct-fill buffers, and any prepared
  metadata index-aligned;
- final output-sector order should follow the surviving table order unless a
  later explicit compaction step changes it consistently;
- when starting a new output product q-label, first project to symmetry
  q-labels and check the per-symmetry impossible sets so already rejected
  symmetry q-label tuples are skipped without X-symbol work.

### 7. Reuse the current RMT/w-matrix materialization path

This phase deliberately keeps the current final stage.

For each valid output sector:

1. Convert the interval to the representation currently expected by
   `_prepare_compress_sector` and `_contract_prepared_compress_sector`.
2. Build contracted w-matrices from:
   - source `sector_wmat(q1, i, n)`;
   - cached raw X-symbol arrays for non-Abelian symmetries;
   - the Abelian trivial factor.
3. Reuse existing QR compression and RMT contraction code.
4. Push result product q-labels, w-matrices, and RMTs into the same output
   buffers used today.

This preserves correctness while isolating the first optimization to sector
enumeration, fusion filtering, and X-symbol caching.

`_prepare_compress_sector` can still return `nothing` after X-symbol filtering.
That sector must be removed from the same output product-q-label and interval
tables before final `qlabels`, `wmats`, and `RMTs` are materialized.

## Implementation phases

### Phase 0: baseline and guards

Before editing:

- Run the DMRG test from `AGENTS.md`.
- Record current allocation/time data for the representative contraction
  benchmarks in `docs/speed_after_refactor.md`, if those helpers still run.
- Add a focused local check or debug assertion that the new pair count equals
  the number of entries filled into the possible-pair vector.
- Capture `@code_warntype` for current `contract` and the helpers that will be
  changed, so later phases can prove they did not add instability.

### Phase 1: possible-pair table

Implement only the counting pass, pair-table fill, sorting, and interval
construction. Keep the current contraction path available until the new table is
validated.

Validation:

- Compare the set of `(contracted_key, out_key, sector_index_q1,
  sector_index_q2)` pairs against the current nested-loop discovery before
  X-symbol filtering on representative contractions.
- Confirm each pair-table entry carries contracted product q-labels (`QT`) and
  that X-symbol lookup projects them to contracted symmetry q-labels
  (`NTuple{NZ, Int}`).
- Check type stability for the pair-table construction helper.
- Confirm every sort in the helper is in-place.

### Phase 2: cache-miss CGT input metadata

Add function-level helpers that compute non-Abelian CGT input metadata for q1
and q2 on an X-symbol cache miss.

Validation:

- For representative sectors and non-Abelian symmetries, compare metadata
  against `_sector_cgt_metadata` plus `_stored_contract_legs`.
- Include repeated equal symmetry q-labels on same-direction legs.
- Check type stability for the helper and for the contract entry point.
- Confirm stored-leg ordering does not allocate through out-of-place sorting.
- Confirm no required full metadata matrix is built in the hot path.
- If an optional lazy metadata cache is added, confirm it is not `Matrix{Any}`,
  `Vector{Any}`, or an abstract element matrix.

### Phase 3: raw X-symbol cache and impossible-sector filtering

Use the pair-table intervals to load raw X-symbol arrays and reject impossible
output sectors before w-matrix contraction.

Use an explicit validity vector or invalid marker to represent rejected output
sectors. This is the final intended representation for filtering; do not switch
to swap-and-pop later.

Validation:

- Compare accepted/rejected output sectors with the current `contract` result.
- Confirm repeated non-Abelian symmetry types share one cache tuple slot and
  distinct non-Abelian symmetry types use distinct tuple slots.
- Include a dedicated duplicated-non-Abelian-symmetry case. A concrete source is
  a local space built from:

```julia
option = FermionSOption(1, :SU2, :SU2, nothing)
space = getLocalSpace(option)
```

  Use tensors derived from this space to verify that the two `SU2` product
  symmetry indices share one X-symbol cache tuple slot while still using
  distinct per-index CGT metadata.
- Confirm the product-symmetry-index to cache-slot helper is type-stable for
  `Val(n)`. Prefer a non-generated implementation when it remains fully
  inferred; use `@generated` only when needed to materialize a compile-time
  constant.
- Confirm `_new_xsym_caches(PS)` returns a concrete tuple of concrete dictionary
  types.
- Confirm the cache accessor returns the concrete dictionary type for `Val(n)`,
  not a union or abstract type visible to the hot lookup path.
- Confirm Abelian-only contractions do not create non-Abelian cache work.
- Confirm cache type definitions contain no `Any`, `Vector{Any}`,
  `Dict{Any,...}`, finite-union vector element type, or abstract dictionary
  element type.
- Confirm the impossible-set shortcut rejects exactly the same sectors as
  pair-by-pair X-symbol loading when keyed by output free symmetry q-labels.
- Check `@code_warntype` for the X-symbol lookup/filtering loop.
- Confirm invalid output sectors are marked in a concrete validity table and
  skipped consistently during final materialization.

### Phase 4: connect to unchanged final materialization

For each valid output sector, adapt the interval data to the current
`new_wmats` and `sector_pairs` inputs, then call the existing preparation and
RMT contraction helpers.

Validation:

- Run the DMRG test from `AGENTS.md`.
- Run relevant contraction tests.
- Compare representative contraction outputs before and after, ignoring output
  product-q-label sector order when the tensor data is otherwise equivalent.
- Inspect `@code_warntype` for `contract` and new hot helpers.
- Record allocation and GC impact for the benchmark cases in
  `docs/speed_after_refactor.md`.
- Confirm no new out-of-place sorting remains in modified contraction helpers.

### Phase 5: direct contracted-wmat concat and QR

After Phase 4 is correct, remove the remaining per-pair contracted-wmat
materialization from contraction. The final code should look native to the
current `TLArray.wmats` format:

```julia
Vector{NTuple{M, Matrix{Float64}}}
```

where `M` is the number of non-Abelian product-symmetry entries stored in the
TLArray w-matrix tuple. Do not keep phase-local names or containers that imply
the old `NTuple{N, Vector{...}}` storage shape.

#### Index tables and invalid-sector tracking

Keep the possible-pair table and output-sector interval table index-stable.
When an invalid output sector is detected, do not swap-and-pop. Instead,
allocate a concrete validity table aligned with `out_keys` and `out_intervals`:

```julia
valid_output = trues(length(out_keys))
```

or an equivalent concrete invalid marker. If any non-Abelian symmetry reports
`nothing`, set `valid_output[out_pos] = false` and skip the rest of that
interval. Later loops must test `valid_output[out_pos]` and continue without
materializing q-labels, w-matrices, or RMTs for invalid sectors. If compact
output arrays are desired, compact once at the final materialization boundary.

The same validity table should also handle sectors for which QR/compression
returns no usable rank. Do not maintain one invalidation path for X-symbol
failures and another path for QR failures.

Final output storage should also be index-aligned with `out_keys` during the
work phase. Allocate the candidate output w-matrix and RMT vectors to
`length(out_keys)` with undefined elements:

```julia
final_wmats = Vector{NTuple{M, Matrix{Float64}}}(undef, length(out_keys))
final_RMTs  = Vector{LurTensor{RT, RD_out, Array{RT, RD_out}}}(undef, length(out_keys))
```

Only fill `final_wmats[out_pos]` and `final_RMTs[out_pos]` for valid output
sectors. Final compact TLArray construction should then copy or push only slots
where `valid_output[out_pos]` is true. This keeps all candidate-sector tables
aligned during computation while preserving compact final TLArray storage.

#### Symmetry index mapping helpers

Define small helpers that convert between the three symmetry-index spaces used
in contraction:

- product-symmetry index `n`, in `1:N`;
- non-Abelian w-matrix tuple index `m`, in `1:M`;
- distinct non-Abelian symmetry cache index `c`, in `1:C`, where `C` is the
  number of distinct non-Abelian symmetry types.

Required helpers:

```julia
wmat_tuple_slot(::Type{PS}, ::Val{n})              # product n -> m or nothing
product_symmetry_index_from_wmat_slot(::Type{PS}, ::Val{m})  # m -> product n
xsym_cache_slot(::Type{PS}, ::Val{n})              # product n -> c or nothing
product_symmetry_index_from_xsym_cache_slot(::Type{PS}, ::Val{c})  # c -> first product n
```

These helpers should be evaluable at compile time for concrete `PS` and
`Val(...)` arguments. In validation, use `@inferred` or `@code_warntype` to
confirm that calls such as `wmat_tuple_slot(PS, Val(n))`,
`product_symmetry_index_from_wmat_slot(PS, Val(m))`,
`xsym_cache_slot(PS, Val(n))`, and
`product_symmetry_index_from_xsym_cache_slot(PS, Val(c))` return compile-time
constant results with no runtime union visible to hot callers.

`xsym_cache_slot` must map repeated non-Abelian symmetry types to the same
cache index. For example, for `(U1, SU2, U1, SU2)`:

```julia
wmat_tuple_slot(PS, Val(1)) === nothing
wmat_tuple_slot(PS, Val(2)) == 1
wmat_tuple_slot(PS, Val(3)) === nothing
wmat_tuple_slot(PS, Val(4)) == 2

xsym_cache_slot(PS, Val(1)) === nothing
xsym_cache_slot(PS, Val(2)) == 1
xsym_cache_slot(PS, Val(3)) === nothing
xsym_cache_slot(PS, Val(4)) == 1
```

Use the inverse helpers when constructing tuples or iterating over stored
containers:

```julia
ntuple(Val(M)) do m
    n = product_symmetry_index_from_wmat_slot(PS, Val(m))
    # build one stored non-Abelian w-matrix for product symmetry n
end
```

Prefer ordinary methods, `Val` dispatch, tuple recursion, and function barriers.
Use `@generated` only when a non-generated implementation cannot keep the
mapping result as a compile-time constant under `@inferred` or
`@code_warntype`. Any generated helper should be a small constant-returning
index helper, not a large algorithm.

#### Per-sector factor storage

Store the per-output-sector split factors in a sector-major tuple format.
This is not the final `TLArray.wmats` storage; each tuple entry is the 3D
factor tensor produced after direct-fill QR splitting for one stored
non-Abelian w-matrix slot:

```julia
sector_wmats = Vector{NTuple{M, Array{Float64, 3}}}(undef, length(out_keys))
```

Only assign `sector_wmats[out_pos]` for valid output sectors with a nonzero
retained rank. Invalid entries remain unassigned and must never be read.
Final materialization iterates over `out_pos in eachindex(out_keys)` and pushes
only entries where `valid_output[out_pos]` is true.

Do not allocate or push Abelian trivial w-matrices. Abelian symmetries have no
entry in `sector_wmats[out_pos]`; their contribution to the RMT contraction is
the same scalar/trivial factor semantics as before.

The final output `TLArray.wmats` remains `Vector{NTuple{M, Matrix{Float64}}}`.
Those matrices are the common QR isometries, while `sector_wmats[out_pos][m]`
stores the split `R` factors with shape `(rank_m, om1_m, om2_m)` used to build
the per-pair `K` tensors for RMT contraction.

`out_intervals[out_pos]` gives the source sector-pair interval, and
`sector_wmats[out_pos]` gives the per-slot split factors for that output sector.
Together they should provide enough information to run the conventional final
RMT compression/accumulation path for that valid output sector without carrying
an additional pushed `sector_reps` vector.

#### Direct-fill algorithm

For one valid output sector interval:

1. Read source sector pairs directly from the possible-pair table:

```julia
(sector_index_q1, sector_index_q2)
```

2. For each stored non-Abelian w-matrix slot `m = 1:M`, map to product-symmetry
   index `n = product_symmetry_index_from_wmat_slot(PS, Val(m))`.
3. Look up or load the raw X-symbol array through the distinct-cache slot
   `xsym_cache_slot(PS, Val(n))`. If lookup/load returns `nothing`, mark the
   output sector invalid and stop processing the interval.
4. Determine the concatenated matrix width from source w-matrix column counts:

```julia
sum(size(sector_wmat(q1, i, n), 2) *
    size(sector_wmat(q2, j, n), 2) for (i, j) in sector_pairs)
```

5. Allocate one dense concatenation matrix for the current output sector and
   stored slot `m`. Its row count is the X-symbol output multiplicity dimension.
6. Fill the concatenation matrix directly with the same contraction equation
   currently used by `_contract_xsym_wmat`:

```julia
concat[a, col(b, c, pair)] =
    xarr[bb, cc, a] * wm1[bb, b] * wm2[cc, c]
```

   The equation, index order, and numerical semantics must not change.
7. Run QR on the concatenation matrix and apply the same rank-retention rule as
   `_qr_shared_isometry`.
8. Save the common isometry matrix into the final output w-matrix tuple for the
   sector.
9. Split `R` into per-source-pair factors with logical shape:

```julia
(rank_m, om1_m, om2_m)
```

10. Store the split factor tensor for slot `m` in `sector_wmats[out_pos][m]`.
11. For each source-sector pair, combine all per-slot split factors using the
    existing `_combined_om_factor_array` logic, then call
    `_contract_RMT_pair_into!` with the unchanged RMT contraction algorithm and
    cost-order selection.
12. Write the common QR isometry tuple into `final_wmats[out_pos]` and the
    accumulated RMT into `final_RMTs[out_pos]`.

This phase may define contraction-specific helpers for direct fill and QR
instead of changing `_qr_shared_isometry`, which is also used by SVD. Suitable
helper boundaries are:

```julia
_fill_contract_wmat_concat!(concat, xarr, wm1, wm2, col_offset)
_qr_shared_isometry_for_contract(concat::Matrix{Float64}, pair_shapes, tol)
```

or equivalent function-level helpers. Do not add wrapper structs for factor
views unless explicitly justified and approved.

#### Implementation constraints

- `_contract_wmats` should not remain the per-pair X-symbol/w-matrix
  contraction boundary. Split it into X-symbol lookup/filtering support and
  direct-fill support, or remove it.
- Keep `sector_wmats` sector-major and tuple-shaped:
  `Vector{NTuple{M, Array{Float64, 3}}}`.
- Allocate `sector_wmats` to the known number of output-sector candidates; do
  not push sector tuples as they are discovered.
- Allocate final candidate `wmats` and `RMTs` vectors to `length(out_keys)` and
  assign by `out_pos`. Do not push final outputs during per-sector processing.
- Keep invalid-sector state in a concrete validity table; do not swap-and-pop.
- Use `out_intervals[out_pos]` plus `sector_wmats[out_pos]` as the primary
  inputs to final conventional compression/accumulation for that output sector.
- Store split `R` factors using concrete array/vector element types. Do not use
  `Vector{AbstractArray}` or `Vector{Any}`.
- Preserve the existing truncation rule based on row norms of `R`.
- Allocation savings should come from removing per-pair contracted-wmat arrays
  and `SV_split`, not from changing numerical semantics.

#### Validation

- Compare direct-filled concatenation matrices against concatenating
  `_contract_xsym_wmat` outputs for representative Abelian, mixed, and
  non-Abelian contractions.
- Compare `Q`, retained rank, split factors, and final RMT against the current
  `_prepare_compress_sector` plus `_contract_prepared_compress_sector` path for
  small deterministic sectors.
- Include repeated non-Abelian symmetry types, especially the two-`SU2` local
  space case.
- Check `@inferred` or `@code_warntype` for every index conversion helper:
  product index to wmat tuple slot, wmat tuple slot to product index, product
  index to X-symbol cache slot, and X-symbol cache slot to product index.
- Check `@code_warntype` on the new direct-fill helper, QR helper, and
  per-sector merge helper.
- Confirm invalid output sectors are marked and skipped without reading
  unassigned `sector_wmats`, `final_wmats`, or `final_RMTs` entries.
- Confirm compact final TLArray construction preserves alignment between
  `qlabels`, `wmats`, and `RMTs` after skipping invalid candidate slots.
- Run the DMRG test from `AGENTS.md`.
- Record allocation and GC impact against Phase 4 and the original baseline.

#### Suggested sub-phases

If the full Phase 5 patch becomes too large, split it into these smaller phases:

1. Add compile-time index helpers and tests:
   product symmetry index, non-Abelian wmat tuple index, and distinct
   X-symbol-cache index conversions.
2. Replace swap-and-pop invalidation with `valid_output`, leaving the existing
   pushed materialization path otherwise unchanged.
3. Preallocate candidate `sector_wmats`, `final_wmats`, and `final_RMTs` to
   `length(out_keys)` and assign by `out_pos`, still using the conventional
   compression helpers.
4. Replace pushed `sector_reps`-style state with reads from
   `out_intervals[out_pos]` and the possible-pair table.
5. Implement direct-fill QR and `sector_wmats` split-factor generation.
6. Switch final RMT accumulation to consume `out_intervals[out_pos]` plus
   `sector_wmats[out_pos]`, then compact valid candidate slots into the final
   TLArray fields.

## Risks

- The current `qlabels` matrix is `(leg, sector)`. Accidentally constructing
  `(sector, leg)` will produce subtle sector-label errors.
- Swap-and-pop invalid-sector removal changes sector order. This is allowed, but
  every associated table must be updated together: output product q-labels,
  intervals, and any per-sector prepared metadata.
- The X-symbol cache key intentionally omits CGT stored-leg positions. On a
  cache miss, the `getNsave_Xsymbol` arguments must still be built from CGT
  metadata that includes the stored contracted-leg positions.
- X-symbol cache keys must stay as symmetry-q-label keys. Accidentally switching
  to CGT metadata keys would change cache semantics and can reduce sharing.
- Repeated non-Abelian symmetry types need shared X-symbol caches but distinct
  per-index cache-miss metadata.
- Over-specializing dictionary key types may make the implementation brittle;
  under-specializing them may move type instability into the hot path. Use
  function barriers and `@code_warntype` to choose the minimal stable version.
- A helper that returns different concrete types for Abelian and non-Abelian
  branches can poison the outer loop. Keep branch-specific logic separated or
  use generated/`Val`-specialized helpers where the existing code already does.
- Out-of-place sorting can hide large allocation regressions even when the
  algorithmic structure is improved.
- A single metadata matrix across all non-Abelian slots is unnecessary and can
  silently become abstract if per-symmetry q-label component types differ.
- A vector of cache dictionaries can still expose abstract or union element
  types to the hot lookup path. Use a generated tuple of concrete dictionary
  types instead.
- Validity-vector filtering can accidentally read unassigned `sector_wmats`
  entries if final materialization does not check the validity table first.
  Every loop over candidate output sectors must explicitly skip invalid slots.
- Treating X-symbol `nothing` as an output-free-symmetry-q-label impossibility
  is only valid under the fusion invariant stated above; if validation finds a
  counterexample, the impossible-set key must be refined.
- `Matrix(F.R)` allocates the QR `R` parent matrix. This is acceptable for the
  first direct-fill implementation because the removed allocations are the
  per-pair contracted w-matrices and `SV_split`. A later LAPACK-level
  QR workspace optimization should be separate.
- Saving split `R` factors preserves the current convention and avoids factor-view
  lifetime issues in the first Phase 5 implementation. If this copy remains
  significant after direct concat fill, optimize it as a separate pass.

## Acceptance criteria

- `contract` returns the same tensor data and product-q-label sector labels as
  before for representative Abelian, mixed Abelian/non-Abelian, and repeated
  product-q-label cases.
- DMRG `run()` in `example/DMRG_central.jl` completes with final energies:
  - `Nkeep=20`: `-82.37493703992092`
  - `Nkeep=50`: `-82.39703156828021`
- The contraction path no longer contracts X-symbols into w-matrices during
  sector-pair discovery.
- Impossible fusion sectors are skipped by output free symmetry q-label tuple,
  not repeatedly rediscovered for every sector pair.
- X-symbol cache keys remain `(contracted symmetry q-labels,
  free symmetry q-labels of q1, free symmetry q-labels of q2)`.
- X-symbol cache containers and keys avoid `Any`.
- There is one X-symbol cache tuple slot per distinct non-Abelian symmetry type,
  with a compile-time-specialized helper from product-symmetry index to cache
  slot.
- Invalid output sectors are marked in a concrete validity table and skipped
  during final materialization; do not use swap-and-pop invalidation.
- Final output product q-label order is not required to be sorted, but all
  materialized output tables must remain mutually aligned.
- Output sectors removed by `_prepare_compress_sector === nothing` are handled
  consistently with X-symbol-impossible sectors.
- `@code_warntype` shows no new type instability in the contract entry point or
  modified hot helpers.
- Modified contraction helpers use only in-place sorting.
- Allocation and GC measurements are recorded against the baseline.
- After Phase 5, contracted w-matrices are filled directly into the QR
  concatenation buffers for contraction, and resulting sector w-matrices are
  saved as `Vector{NTuple{M, Matrix{Float64}}}` entries. `R` is split/saved
  using the old convention, without changing the X-symbol/w-matrix contraction
  equation or the RMT contraction algorithm.
