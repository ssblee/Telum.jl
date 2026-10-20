# SVD CGTSVD counted preassembly plan

## Goal

Optimize the CGTSVD preassembly stage in `src/svd.jl` by separating discovery
from dense payload construction:

1. get the CGTSVD cache and count possible product split rows;
2. construct row/signature and payload vectors with known capacity;
3. fill those vectors in a separate pass.

The main target is to avoid append-driven discovery in the hot preassembly
path and to make later parallelization straightforward. The refactor should
preserve current SVD semantics, truncation behavior, physical-q-label cache
keys, and one CGTSVD cache per distinct non-Abelian symmetry.

## Current Shape

The current implementation already uses:

- physical qlabels in tensor-leg order as the CGTSVD cache key;
- one cache dictionary per distinct non-Abelian symmetry type;
- `_SVDSplitRow{L,R,QT}` for product row metadata;
- separate vectors for non-Abelian `left_isos`, `right_isos`, and `cores`.

The remaining inefficiency is that split discovery and payload creation are
still tightly coupled. The code appends to per-symmetry split vectors while it
discovers choices, then expands product rows from those vectors.

The desired workflow counts possible product rows before allocating/filling
the product row and payload vectors.

Because CGTSVD keys are needed in both the counting and fill phases, the plan
also stores precomputed key vectors. This avoids reconstructing
physical-leg-ordered keys repeatedly.

## Invariants

- For fixed `svd_std(q, left_legs, ...)`, physical qlabels ordered by tensor leg
  are sufficient as the CGTSVD cache key.
- There is one CGTSVD cache for each distinct non-Abelian symmetry type in
  `productsymm(q)`. Abelian symmetries have no CGTSVD cache dictionary.
- Repeated occurrences of the same non-Abelian symmetry type reuse the same
  cache dictionary.
- CGTSVD keys should be saved per non-Abelian product symmetry occurrence. A
  repeated non-Abelian symmetry type reuses the same cache dictionary, but each
  product slot has its own physical-key stream.
- Side signatures are full product qlabels in physical leg order:

```julia
left_signature  = ntuple(i -> physical_key[left_legs[i]], Val(L))
right_signature = ntuple(i -> physical_key[right_legs[i]], Val(R))
```

- `_SVDSplitRow{L,R,QT}` stores only metadata:

```julia
sector_index::Int
q::QT
left_signature::NTuple{L,QT}
right_signature::NTuple{R,QT}
```

- Dense payload vectors remain aligned with `split_rows`: entry `i` in
  `split_rows`, `left_isos`, `right_isos`, and `cores` describes the same
  product split row.
- Abelian symmetry slots contribute to `row.q::QT`, but not to non-Abelian
  payload tuples.

## Phase 1: Count CGTSVD Split Choices

Add a counting pass that walks active TLArray sectors and obtains the CGTSVD
objects before payload allocation.

Before walking sectors, construct one typed vector whose element is the tuple
of all non-Abelian product-slot keys for one active TLArray sector:

```julia
KeyTuple = _svd_cgtsvd_key_tuple_type(PS, QT, Val(QD))
keys_by_row = Vector{KeyTuple}()
```

where `_svd_cgtsvd_key_tuple_type` should be a generated helper that returns:

```julia
Tuple{
    _cgtsvd_cache_key_type(QT, Val(n1), Val(QD)),
    _cgtsvd_cache_key_type(QT, Val(n2), Val(QD)),
    ...
}
```

`KeyTuple` contains exactly `M` keys. It does not contain keys for Abelian
product symmetry slots. It must include duplicate entries when the same
non-Abelian symmetry type appears multiple times in the product symmetry. Here
`M` is the TLArray non-Abelian w-matrix tuple width: one entry per
non-Abelian product symmetry occurrence.

Here `M` is the number of non-Abelian product symmetry occurrences represented
by TLArray w-matrix slots. This is intentionally not the number of distinct
CGTSVD cache dictionaries. For example, if the same non-Abelian symmetry type
appears twice in `PS`, both product slots may map to the same cache dictionary,
but their entries inside each row key tuple remain distinct because they read
different qlabel components from `QT`.

Example:

```julia
PS = ProductSymm{Tuple{SU2, SU2}}
M == 2
KeyTuple == Tuple{
    _cgtsvd_cache_key_type(QT, Val(1), Val(QD)),
    _cgtsvd_cache_key_type(QT, Val(2), Val(QD)),
}
```

There is still only one CGTSVD cache dictionary for `SU2`, but each active
TLArray row stores two keys in `KeyTuple`, one for each `SU2` product slot.

The element type of `keys_by_row` is known at compile time from `PS`, `QT`, and
`QD`:

```julia
Tuple{
    NTuple{QD, fieldtype(QT, n1)},
    NTuple{QD, fieldtype(QT, n2)},
    ...
}
```

where `n1`, `n2`, ... are the product symmetry slots represented by
non-Abelian w-matrix slots.

If `M == 0`, the key tuple type is `Tuple{}` and `keys_by_row` can be skipped
or kept as `Vector{Tuple{}}` by the same generic path.

Use `sizehint!` with the number of active nonzero TLArray sectors:

```julia
nactive = count(ri -> !q.iszero[ri], sector_slots(q))
sizehint!(keys_by_row, nactive)
```

For each active `sector_index`:

1. For each product symmetry slot `n`, build the symmetry-slot physical key:

```julia
physical_key_n = _svd_physical_qlabel_key(QT, q, sector_index, Val(n))
```

For non-Abelian product slots, insert this key into the row key tuple. After
all non-Abelian slots are processed for the active sector, push one tuple:

```julia
row_keys = ntuple(Val(M)) do slot
    n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
    _svd_physical_qlabel_key(QT, q, sector_index, Val(n))
end
push!(keys_by_row, row_keys)
```

The vector index is the active-sector position, not the raw `sector_index`.
Keep a parallel `active_sector_indices` vector if later phases need to map back
to TLArray sector slots.

2. Get the CGTSVD cache dictionary for slot `n`:

```julia
cache = _cgtsvd_cache_for(cgtsvd_caches, symm(q)[n], productsymm(q), Val(n))
```

For Abelian symmetry slots, the cache is `nothing`.

3. Get the CGTSVD object or Boolean shortcut:

```julia
cgtsvd = _svd_get_cgtsvd(S, physical_key_n, upsp, dnsp, left_legs_canon, cache)
```

4. Count split choices for this symmetry slot:

```julia
nsplits_n = if isabelian(S)
    1
elseif cgtsvd isa LurCGT.CGTSVD
    length(cgtsvd.bond_sps)
else
    1
end
```

The `Bool` shortcut case counts as one split choice.

5. The possible product split-row count for the TLArray sector is:

```julia
nrows_sector = prod(nsplits_n for n in 1:N)
```

The counted logical object is:

```julia
(left_full_qlabels, right_full_qlabels, intermediate_qlabels)
```

where the number of possible `intermediate_qlabels` is known from the CGTSVD
split counts. The concrete `left_full_qlabels` and `right_full_qlabels` are
not built in this phase; they are built during fill from the full physical
product key.

Sum `nrows_sector` over all active nonzero TLArray sectors to get
`total_rows`.

### Phase 1 Data To Keep

Keep a small typed description per active sector. It should contain:

- `sector_index`;
- per-symmetry split counts;
- optionally the cached CGTSVD objects or Boolean shortcuts needed for the fill
  pass;
- the output row range assigned to the sector.

The saved key vector can be kept outside this per-sector description:

```julia
keys_by_row::Vector{KeyTuple}
```

where each field of `KeyTuple` is generated from the product symmetry slot
represented by that w-matrix slot. During fill, recover the key for a product
slot with:

```julia
slot = nonabelian_wmat_slot(PS, Val(n))
physical_key_n = keys_by_row[active_position][slot]
```

This row-major layout has better locality for the fill pass because the loop
usually processes one active TLArray sector at a time and needs all
non-Abelian keys for that row.

Do not store reduced `left_iso`, `right_iso`, or `core` tensors during the
counting pass. Dense payload construction belongs to the fill pass.

## Phase 2: Compute Output Ranges

From the per-sector counts, compute deterministic output ranges:

```julia
sector_ranges[i] = offset + 1:offset + nrows_sector
offset += nrows_sector
```

This is the preferred shape for parallelization. Each sector owns an exclusive
output range, so fill work can later be parallelized without an atomic counter
or shared `push!`.

If exact preallocation is not introduced immediately, use the same count as a
capacity hint:

```julia
sizehint!(split_rows, total_rows)
sizehint!(left_isos, total_rows)
sizehint!(right_isos, total_rows)
sizehint!(cores, total_rows)
```

Exact preallocation is the stronger target:

```julia
split_rows = Vector{_SVDSplitRow{L,R,QT}}(undef, total_rows)
left_isos  = Vector{NTuple{M,Matrix{Float64}}}(undef, total_rows)
right_isos = Vector{NTuple{M,Matrix{Float64}}}(undef, total_rows)
cores      = Vector{NTuple{M,Array{Float64,3}}}(undef, total_rows)
```

If a counted row later reduces to numerical zero during payload construction,
there are two acceptable approaches:

1. Use `sizehint!` and `push!` within a sector-local temporary vector, then copy
   into the final range.
2. Keep exact preallocation and record a `valid::BitVector`, then compact rows
   and payload vectors after fill.

Prefer the first approach unless exact indexed writes are needed immediately for
parallel work. It keeps the implementation smaller and avoids carrying invalid
entries through sorting/class construction.

## Phase 3: Fill Sector Rows And Payloads

The fill pass performs the dense work.

For each active sector:

1. Reuse the saved row-major CGTSVD key tuples and CGTSVD cache results from
   Phase 1.
2. Build the full physical product key for this sector:

```julia
physical_key = _svd_physical_product_key(QT, q, sector_index)
```

3. Build full left/right side signatures:

```julia
left_signature, right_signature =
    _svd_physical_key_side_signatures(physical_key, left_legs, right_legs)
```

4. Build per-symmetry split choices for that sector.
5. Enumerate the Cartesian product of per-symmetry choices.
6. Build the full intermediate product qlabel:

```julia
q = ntuple(n -> choice[n].q, Val(N))::QT
```

7. Build the non-Abelian payload tuples:

```julia
left_payload = ntuple(Val(M)) do slot
    n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
    choice[n].left_iso
end
```

Use the same pattern for `right_payload` and `core_payload`.

8. Write one aligned row:

```julia
split_rows[row_index] = _SVDSplitRow(
    sector_index, q, left_signature, right_signature)
left_isos[row_index]  = left_payload
right_isos[row_index] = right_payload
cores[row_index]      = core_payload
```

For a `sizehint!` implementation, this same write can be expressed as
`push!` into the aligned vectors. The important requirement is that discovery,
capacity estimation, and payload fill remain separate phases.

## Phase 4: Sort Rows Before QR Sharing

After fill, sort rows and payload vectors together by:

```julia
(row.q, row.left_signature, row.right_signature, row.sector_index)
```

Apply the same permutation to:

```julia
split_rows
left_isos
right_isos
cores
```

Then derive intermediate-q intervals:

```julia
intermediate_qs, intermediate_intervals = _svd_q_intervals(split_rows)
```

Sorting first is required because QR sharing should operate on isometries with
the same corresponding intermediate qlabels. The sorted row layout makes those
groups easier to find and keeps later class construction aligned with the
shared-isometry basis.

## Phase 5: QR Sharing

Shared-isometry QR should still happen per non-Abelian symmetry slot.

Run QR sharing after rows and payload vectors have been sorted together.

Grouping keys must encode the exact sharing condition:

1. the full intermediate product qlabel `row.q` is the same;
2. the qlabel for the corresponding symmetry slot and side is the same.

For product symmetry slot `n`, the side qlabel is obtained by projecting the
full side signature to slot `n`:

```julia
left_side_qs_n  = ntuple(i -> row.left_signature[i][n], Val(L))
right_side_qs_n = ntuple(i -> row.right_signature[i][n], Val(R))
```

Therefore QR sharing keys are:

- left side: `(row.q, left_side_qs_n)`;
- right side: `(row.q, right_side_qs_n)`.

The full `row.q` component is mandatory. Rows with the same side qlabels for
slot `n` but different full intermediate product qlabels must not share the
same QR basis.

If payload arrays are aliased across product rows, the implementation must
ensure each unique payload is transformed exactly once. The sorted row layout
does not by itself solve aliasing; it only gives the correct grouping order and
intermediate-q boundaries.

The QR factor is absorbed into the matching core axis:

- left sharing: `_contract_om_axis(core, factor, 1)`;
- right sharing: `_contract_om_axis(core, factor, 2)`.

## Phase 6: Build Classes

Within each interval, build connected components using the existing class rule:

- rows are connected if they share `left_signature`;
- rows are connected if they share `right_signature`;
- each connected component is one class matrix.

Class construction should continue to read all dense payloads by row index from
the aligned payload vectors.

## Phase 7: Parallelization Shape

This refactor should make the heavy fill path parallelizable without another
major rewrite.

Parallel-friendly properties:

- Phase 1 counting has no dense payload writes.
- Phase 2 assigns disjoint output ranges per active sector.
- Phase 3 can fill each sector range independently.
- Later sorting, QR sharing, and class grouping are global coordination steps
  and can remain serial initially.

Avoid:

- shared append state in dense fill loops;
- loop-carried global row counters in payload fill;
- mutating the same payload array through multiple product-row aliases.

## Phase 8: Cleanup Targets

After the counted preassembly is implemented:

- remove append-driven product-row construction from the hot path;
- keep helper return types concrete;
- avoid `Vector{Any}` and `Dict{Any,Any}`;
- keep Abelian payloads implicit;
- keep CGTSVD cache construction and cache-slot mapping shared with the
  physical-key preassembly design;
- update names so helpers distinguish counting from filling, for example:

```julia
_count_svd_cgtsvd_splits(...)
_fill_svd_cgtsvd_splits!(...)
```

Names can change to match the existing local style, but the count/fill
separation should remain explicit.

## Validation

For this internal optimization, the DMRG smoke test is sufficient:

```julia
julia --project=Telum -e 'using Telum, LinearAlgebra; include("Telum/example/DMRG_central.jl"); run(time_blocks=false)'
```

Expected final energies:

- `Nkeep_DMRG = 20`: `-82.37493703992092` up to numerical precision;
- `Nkeep_DMRG = 50`: `-82.39703156828021` up to numerical precision.
