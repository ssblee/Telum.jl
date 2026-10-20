# Refactor plan: contraction candidate storage and zero-sector filtering

## Goal

Reduce allocation and redundant work in `src/contract.jl` after the
field-backed `TLArray` refactor.

The target is the path that groups possible sector pairs, builds QR factor
data from X-symbols and source w-matrices, contracts RMT blocks, and
materializes the resulting `TLArray`.

This pass should preserve tensor semantics and public APIs. It should not add
new structs, wrapper objects, traits, or broad architectural layers.

## Review of proposed changes

The proposal is sound overall. The current code still has avoidable allocation
and some leftover product-symmetry-wide containers from the old representation.
The highest-value changes are:

- use `Val(n)` w-matrix slot lookup in generated or tuple-building paths;
- store candidate QR `R` factors as sector-major
  `Vector{NTuple{M, Array{Float64, 3}}}`, where `M` is the number of stored
  non-Abelian w-matrix slots;
- use `possible_pairs` plus `out_intervals` directly instead of copying
  `candidate_pairs`;
- build output q-label candidates directly from `out_keys`;
- mark invalid sectors before final materialization;
- treat zero RMT results as invalid sectors;
- replace small-relative-sector dropping with exact zero-sector dropping, and
  allow trusted construction paths to skip that check.

Terminology used in this document:

- `w-matrix`: the shared isometry `Q` from the QR compression. These matrices
  are stored in the per-sector `wmats` vector of the resulting `TLArray`.
- `factor`: a per-contraction-pair slice of the retained QR `R` matrix. These
  factors are temporary contraction data and are stored in `candidate_factors`.
- `candidate_factors`: the sector-major temporary container for QR `R` slices.
  This name should replace the old temporary name `candidate_wmats`, because
  the stored objects are not result w-matrices. For one output sector and one
  stored non-Abelian slot, `candidate_factors[out_pos][m]` stores the `R`-factor
  data for the contraction pairs in `out_intervals[out_pos]`; pair boundaries
  or shapes should be recovered from the interval and source sector pairs, not
  from a pushed vector of factors.
- If QR factor data is prepared and immediately converted into the combined
  `factor_arrays` consumed by RMT contraction, a persistent
  `candidate_factors` vector is not required. In that case, use
  `candidate_factors` only as the conceptual name for the per-pair QR `R`
  slices, not as a required long-lived storage object.

Two cautions:

1. Do not use swap-and-pop while candidate tables are still being filled or
   consumed. `out_keys`, `out_intervals`, candidate factors, final w-matrices,
   and candidate RMTs must remain index-aligned. Use `valid_output::Vector{Bool}`
   during the work phase. If swap-and-pop is used, reserve it for a final compact
   phase where every aligned table is updated together. A final one-pass compact
   materialization is simpler and less error-prone than mutating several
   candidate tables.
2. Skipping the zero-sector check in the `TLArray` constructor is only valid for
   construction paths that have already removed invalid sectors. Keep the
   default constructor behavior conservative.

For `contract`, invalid sectors should be removed at the very end of the
function, immediately before constructing the result `TLArray`. The final
compaction loop is responsible for removing impossible output q-label sectors,
zero-rank QR sectors, and exact-zero accumulated RMT sectors. After that point,
`contract` may call an internal trusted `TLArray` construction path that skips
constructor-level zero-sector removal.

## Current issues

- `sector_wmat(q, sector, n::Int)` calls `nonabelian_wmat_slot(PS, n::Int)`,
  which loops over product symmetries at runtime. The generated
  `_contract_wmats` path already knows `n` statically and should use the
  `Val(n)` slot path wherever possible.
- The old temporary named `candidate_wmats` was
  `Vector{NTuple{N, Vector{LurTensor{Float64, 3, ...}}}}`. It included Abelian
  trivial factors and pushed into each per-symmetry vector. The refactor should
  remove this persistent container or replace it with local `candidate_factors`
  terminology for QR `R` slices.
- `candidate_pairs` duplicates source-sector pair indices already available in
  `possible_pairs[out_intervals[out_pos]]`.
- `_out_key_intervals` pushes output keys and intervals as distinct keys are
  discovered. This is small compared with RMT contraction, but it can be made
  more predictable by first counting distinct output keys and then allocating
  final-sized containers.
- Final output q-labels are currently recovered through
  `get_new_cgt_metadata` and `_physical_qlabels_from_cgt_metadata`, even though
  the physical output product q-label tuple is already represented by the
  `out_key`.
- Construction used to route through `_field_tlarray`, which normalized,
  oriented, and called `_drop_small_sectors!`. This should be folded into the
  `TLArray` constructor path with exact zero-sector filtering before
  construction.

## Invariants

- Output physical leg order remains `(free legs of q1..., free legs of q2...)`.
- `out_key` order must match that physical output leg order.
- `TLArray.qlabels` remains a concrete `Matrix{QT}` with `QD_out` rows. After
  the `nsectors` field is added, its column count may be larger than the number
  of live sectors; `nsectors(q)` defines the valid sector range.
- `TLArray.wmats` remains `Vector{NTuple{M, Matrix{Float64}}}`, where each
  matrix is a shared isometry for one stored non-Abelian w-matrix slot.
- Temporary QR `R` factor data for one candidate output sector should be stored
  as `NTuple{M, Array{Float64, 3}}`, one entry per stored non-Abelian w-matrix
  slot.
- Abelian symmetries must not allocate trivial w-matrix entries in candidate
  storage.
- X-symbol cache keys remain `(contracted symmetry q-labels,
  free symmetry q-labels of q1, free symmetry q-labels of q2)`.
- Repeated non-Abelian symmetry types share an X-symbol cache slot but still use
  distinct product-symmetry indices for sector metadata and source w-matrices.
- Invalid output sectors must remove q-labels, w-matrices, and RMTs together.
- Hot helpers must avoid `Any`, abstract container element types, and dynamic
  dispatch in inner loops.

## Phase 1: w-matrix slot lookup cleanup

### Change

Remove hot-path dependence on the `Int` version of `nonabelian_wmat_slot` where
the symmetry index is known statically.

Use existing compile-time helpers:

```julia
sector_wmat(q, sector, Val(n))
wmat_tuple_slot(PS, Val(n))
product_symmetry_index_from_wmat_slot(PS, Val(m))
```

The `Int` method can remain for public or non-hot dynamic callers, but
contraction should not use it in generated branches or `ntuple(Val(...))`
construction.

### Validation

- Inspect `@code_warntype` for the modified contraction helpers.
- Confirm calls from generated contraction code use `Val(n)`.
- Do not remove the `Int` method unless all non-hot callers are audited.

## Phase 2: output key interval allocation

### Change

Rewrite `_out_key_intervals` as an exact-allocation helper over already sorted
`possible_pairs`.

First pass:

- count distinct output keys with one linear sweep, since equal output keys are
  already contiguous after sorting by output key;
- optionally assert each equal-key run is contiguous.

Second pass:

- allocate `out_keys` with exact length;
- allocate `out_intervals` with exact length;
- fill by index.

In addition, create a q-label candidate matrix directly from `out_keys`:

```julia
candidate_qlabels = Matrix{QT}(undef, QD_out, length(out_keys))
```

For each output position, fill the column from `out_key[1]` followed by
`out_key[2]`.

### Validation

- Compare old and new `out_keys` / interval ranges on representative
  contractions.
- Confirm `candidate_qlabels[:, out_pos]` equals the physical output product
  q-labels implied by the current `get_new_cgt_metadata` path.

## Phase 3: sector-major candidate factor storage

### Change

Replace:

```julia
candidate_wmats = Vector{NTuple{N, WmatVec}}(undef, length(out_keys))
candidate_pairs = Vector{SectorPairVec}(undef, length(out_keys))
```

with:

```julia
candidate_factors = Vector{NTuple{M, Array{Float64, 3}}}(undef, length(out_keys))
```

Do not store `candidate_pairs`. Source sector pairs are read from:

```julia
for pair_pos in out_intervals[out_pos]
    pair = possible_pairs[pair_pos]
    i1 = pair[3]
    i2 = pair[4]
end
```

This phase may initially keep the old QR preparation logic behind a function
barrier, but the final shape should be sector-major and stored-slot-major, not
product-symmetry-major.

The name `candidate_factors` is intentional: entries are per-pair QR `R` factor
slices, not shared isometry w-matrices. The shared isometry matrices belong in
the resulting tensor's `wmats` storage.

`candidate_factors[out_pos][m]` should be interpreted together with
`out_intervals[out_pos]`. The container is sector-major and slot-major; it
should not reintroduce a `Vector` that pushes one factor object per contraction
pair.

If the direct-fill implementation prepares one output sector at a time and
immediately builds the per-pair combined `factor_arrays` for RMT contraction,
skip allocating this persistent `candidate_factors` vector entirely. The
important invariant is that the QR `R` slices are not called w-matrices and are
not pushed into a product-symmetry-wide `NTuple{N, Vector{...}}` container.

### Validation

- Confirm Abelian-only contractions allocate no candidate factor tuple entries.
- Confirm `M == n_nonabelian_symmetries(PS)` and all candidate tuples are
  `NTuple{M, Array{Float64, 3}}`.
- Confirm no `push!` is needed inside per-output-sector candidate tuple
  elements.

## Phase 4: direct interval-driven QR factor generation

### Change

This phase fuses the current step 3 and step 4 of `contract` for QR factor
generation. In the current code, step 3 calls `_contract_wmats` for every
possible source-sector pair and stores one per-pair temporary tensor. Step 4
then calls `_prepare_compress_sector`, concatenates those temporary tensors,
runs QR, saves the shared isometry as the result w-matrix, and splits the `R`
factors.

The intended refactor removes that intermediate per-pair temporary tensor
materialization. For each output sector interval, contraction should load or
reuse the required X-symbols, fill the QR concatenation matrix directly, run QR,
and split the retained `R` factors for later RMT contraction.

For each valid output sector and each stored non-Abelian slot `m`:

1. Map `m` to product-symmetry index `n` with
   `product_symmetry_index_from_wmat_slot(PS, Val(m))`.
2. Determine the total concatenation width from all source pairs in
   `out_intervals[out_pos]`.
3. Allocate one dense concatenation matrix.
4. Fill it directly with the same equation used by `_contract_xsym_wmat`:

```julia
concat[a, col(b, c, pair)] =
    xarr[bb, cc, a] * wm1[bb, b] * wm2[cc, c]
```

5. Run QR.
6. Keep only nonzero rows of `R`.
7. Store the common isometry `Q` in `final_wmats[out_pos][m]`. This is the
   result w-matrix for this output sector and stored non-Abelian slot.
8. Split the retained `R` rows into per-pair factors with shape
   `(rank_m, om1_m, om2_m)`.
9. Store those split factors in `candidate_factors[out_pos][m]`.

After this phase, `_contract_wmats` should no longer be the boundary that
contracts X-symbols and source w-matrices into one temporary tensor per pair.
Its remaining responsibilities, if any, should be split into smaller helpers for
X-symbol lookup/filtering and direct concat fill.

This removes the per-pair `_contract_xsym_wmat` allocation, the
`NTuple{N, Vector{...}}` push pattern, and the separate `_prepare_compress_sector`
pass over already-materialized temporary tensors.

### Validation

- Compare direct-filled concatenation matrices against concatenating
  `_contract_xsym_wmat` outputs for small deterministic cases.
- Compare retained rank, `Q`, split factors, and final RMT against the current
  `_prepare_compress_sector` path.
- Include mixed Abelian/non-Abelian and repeated non-Abelian symmetry types.

## Phase 5: invalid-sector tracking

### Change

Use one aligned validity table:

```julia
valid_output = trues(length(out_keys))
```

Mark an output sector invalid when:

- any non-Abelian symmetry returns `nothing` from X-symbol loading;
- QR has zero retained rank for any stored non-Abelian slot;
- the final accumulated RMT for the sector has exact zero norm.

Do not read unassigned entries of `candidate_factors`, `final_wmats`, or
`final_RMTs` unless `valid_output[out_pos]` is true.

Invalid sectors are removed only at the very end of `contract`. During all
intermediate phases, keep candidate tables index-aligned by `out_pos` and only
set `valid_output[out_pos] = false`.

At final materialization, build compact result arrays by scanning
`valid_output`. This final compaction must copy q-labels, w-matrices, and RMTs
together, and must skip every invalid output sector. After this scan, the result
arrays are trusted to contain no impossible sectors and no exact-zero sectors.

### Validation

- Force an X-symbol-impossible sector and confirm q-labels, w-matrices, and RMTs
  are all skipped.
- Force a zero-rank QR case and confirm it follows the same invalidation path.
- Force a zero accumulated RMT case and confirm the sector is removed.

## Phase 6: output q-label materialization from out_keys

### Change

Remove final dependence on:

```julia
get_new_cgt_metadata(...)
_physical_qlabels_from_cgt_metadata(...)
```

for output q-label construction.

Use the candidate q-label columns derived from `out_keys`. During final
compaction:

```julia
result_qlabels[:, sector_index] = candidate_qlabels[:, out_pos]
```

or fill by explicit leg loops to avoid view allocation.

Keep `get_new_cgt_metadata` only if other code paths still need CGT metadata.
Do not reconstruct metadata just to recover physical output q-labels.

### Validation

- Compare result sector q-labels before and after this phase.
- Include equal q-labels on same-direction legs to confirm physical output
  q-label order still follows `(free1..., free2...)`.

## Phase 7: zero-sector construction policy

### Change

Replace `_drop_small_sectors!` with an exact zero-sector policy for raw
field-backed construction:

```julia
_drop_zero_sectors!(symm, qlabels, wmats, RMTs, nsector)
```

It should remove only sectors whose effective sector norm is exactly zero. For
`QD == 2`, keep the current effective norm convention:

```julia
_sector_cgt_size_2d(q, sector_index) * sum(abs2, RMT)
```

For other ranks, use:

```julia
sum(abs2, RMT)
```

Move the canonicalization work formerly performed by `_field_tlarray` into the
constructor path, so call sites construct field-backed tensors with
`TLArray(...)` directly.

Currently `_field_tlarray` does the following after raw construction:

```julia
q = _raw_field_tlarray(symm, qlabels, wmats, RMTs, inds, spaces)
_normalize_wmats!(q)
_orient_wmats!(q)
return _drop_small_sectors!(q)
```

The preferred refactor is to make this the default constructor behavior, with
small-sector dropping replaced by exact zero-sector filtering on the raw fields
before constructing the `TLArray` object. Do not construct a `TLArray`, mutate
it, and then construct another `TLArray` inside the zero-sector filter. The
zero-sector filter should operate on fields before `new(...)`.

To allow in-place swap-and-pop filtering without allocating a compact q-label
matrix, add an explicit sector count field to `TLArray`:

```julia
struct TLArray{...}
    nsectors::Int
    qlabels::Matrix{QT}
    wmats::Vector{NTuple{M, Matrix{Float64}}}
    RMTs::RMTS
    ...
end
```

The constructor invariant becomes:

```julia
size(qlabels, 1) == QD
nsectors == length(wmats) == length(RMTs)
size(qlabels, 2) >= nsectors
```

The old invariant `size(qlabels) == (QD, length(RMTs))` should be replaced.
The extra q-label columns beyond `nsectors` are scratch capacity and must never
be treated as sectors.

Preferred constructor order:

```julia
function TLArray(symm, qlabels, wmats, RMTs, inds, spaces;
                 drop_zero_sectors::Bool = true)
    nsector = length(RMTs)
    if drop_zero_sectors
        nsector = _drop_zero_sectors!(
            symm, qlabels, wmats, RMTs, inds, spaces, nsector)
    end

    q = new{...}(nsector, qlabels, wmats, RMTs, inds, typed_spaces)
    _normalize_wmats!(q)
    _orient_wmats!(q)
    return q
end
```

Filtering before normalization is intended. `_normalize_wmats!` and
`_orient_wmats!` only rescale or sign-flip existing sector data, so exact zero
RMT sectors remain zero under those operations.

`_drop_zero_sectors!` should filter the field arrays in-place using
swap-and-pop:

1. Keep `last_live = nsector`.
2. Scan `sector_index <= last_live`.
3. If the sector norm is exactly zero, swap q-label column `sector_index` with
   q-label column `last_live`, swap `wmats[sector_index]` with
   `wmats[last_live]`, and swap `RMTs[sector_index]` with `RMTs[last_live]`.
4. Decrement `last_live` without incrementing `sector_index`.
5. If the sector is nonzero, increment `sector_index`.
6. After the scan, `resize!(wmats, last_live)` and `resize!(RMTs, last_live)`.
7. Return `last_live` as the stored `nsectors` field.

Do not allocate a compact q-label matrix. The q-label matrix may keep extra
columns as capacity, while `nsectors(q)` defines the valid sector range.

Preferred shape:

```julia
TLArray(...; drop_zero_sectors::Bool = true)
```

The public/default behavior should stay conservative: zero-sector removal is
enabled unless a trusted internal caller explicitly disables it.

`contract` may call:

```julia
TLArray(...; drop_zero_sectors = false)
```

only after its final compaction loop has removed:

- impossible resulting q-label sectors;
- zero-rank QR sectors;
- exact-zero accumulated RMT sectors.

This skip option should be treated as an internal trusted path, not as a public
promise that arbitrary callers can bypass zero-sector validation.

### Validation

- Confirm zero sectors are removed.
- Confirm small but nonzero sectors are no longer dropped by relative threshold.
- Confirm contraction calls the trusted skip path only after the final
  compaction loop has filtered impossible and zero sectors.
- Add a debug assertion or focused test that contraction does not pass exact-zero
  RMT sectors into `TLArray(...; drop_zero_sectors = false)`.
- Audit existing constructor callers before changing the default behavior.
- Confirm `_drop_zero_sectors!` filters raw fields before object construction and
  never constructs or mutates a `TLArray`.
- Confirm zero-sector filtering before `_normalize_wmats!` and `_orient_wmats!`
  gives the same zero-sector decisions as filtering after them.
- Audit every use of `size(q.qlabels, 2)` and replace sector-count assumptions
  with `nsectors(q)`. `size(q.qlabels, 2)` becomes q-label column capacity, not
  the number of live sectors.
- Confirm loops over sectors use `1:nsectors(q)` or `eachindex(q.RMTs)`, not
  q-label matrix column count.

## Phase 8: RMT accumulation and zero result detection

### Change

Accumulate each valid output sector RMT using `out_intervals[out_pos]` and
`candidate_factors[out_pos]`.

After accumulation, compute exact norm:

```julia
iszero(sum(abs2, result_RMT.data))
```

If zero, set `valid_output[out_pos] = false` and do not assign final q-label,
w-matrix, or RMT entries for that sector.

This is intentionally exact-zero filtering, not tolerance filtering.

### Validation

- Compare numerical output against the pre-refactor implementation for ordinary
  sectors.
- Add a minimal regression test if a real zero-result sector case is available.
- Run DMRG validation from `AGENTS.md`.

## Suggested implementation order

1. Change static w-matrix slot lookup in contraction to `Val(n)`.
2. Make `_out_key_intervals` exact-size and build candidate q-label columns from
   `out_keys`.
3. Remove `candidate_pairs` and read source pairs from `possible_pairs` /
   `out_intervals`.
4. Introduce sector-major `Vector{NTuple{M, Array{Float64, 3}}}` candidate
   factor storage.
5. Fuse the current `_contract_wmats` and `_prepare_compress_sector` work for
   one stored non-Abelian slot by directly filling the QR concat matrix from
   X-symbols and source w-matrices.
6. Switch all stored slots to direct fill.
7. Remove final q-label recovery through CGT metadata.
8. Add zero-RMT invalidation.
9. Move `_field_tlarray` canonicalization into the `TLArray` constructor path,
   add a stored `nsectors` field, replace small-sector dropping with exact
   zero-sector swap-and-pop filtering on fields, remove ordinary
   `_field_tlarray` call sites, and add the trusted skip option to the
   constructor.
10. Switch `contract` to the trusted skip path only after final compaction is
    proven to remove every invalid sector.

## Tests and checks

- Run the DMRG test from `AGENTS.md`:
  `run()` in `Telum/example/DMRG_central.jl`, with final energies
  `-82.37493703992092` for `Nkeep=20` and `-82.39703156828021` for `Nkeep=50`.
- Run relevant contraction tests.
- Add or update a focused test for the contract invariant: invalid sectors are
  removed in the final compaction loop before calling the trusted constructor
  path.
- Use `@code_warntype` or JET on `contract` and new hot helpers.
- Measure allocations before and after on representative contraction benchmarks.
- Check that modified helpers do not introduce `Vector{Any}`,
  `Dict{Any, Any}`, abstract element containers, or unstable return types.
- Confirm no out-of-place sort is introduced in the contraction hot path.

## Risks

- Removing `candidate_pairs` is safe only if every later loop consistently uses
  `out_intervals[out_pos]` and `possible_pairs[pair_pos]`.
- Final invalid-sector removal must keep q-labels, w-matrices, and RMTs aligned.
- Direct QR fill must preserve `_contract_xsym_wmat` index order exactly.
- Replacing relative small-sector dropping may expose tiny nonzero sectors that
  were previously discarded. That is intended by this plan, but existing tests
  may implicitly assume the old cutoff behavior.
- Skipping the zero-sector constructor check from an untrusted caller can create
  invalid `TLArray`s. Keep the skip option internal and opt-in.
- If `contract` calls the trusted skip path before final compaction, invalid
  sectors can leak into result tensors. Keep the trusted `TLArray` constructor
  call physically adjacent to the final compact result arrays, not in the work
  phase.
- Moving the old `_field_tlarray` behavior into the constructor can accidentally make
  low-level reconstruction recursive if zero-sector filtering constructs another
  `TLArray`. Avoid this by making `_drop_zero_sectors!` a field-level helper
  that mutates raw fields with swap-and-pop and returns the live sector count
  before `new(...)` is called.
- Adding `nsectors::Int` changes the meaning of `size(q.qlabels, 2)`: it is no
  longer guaranteed to equal the number of live sectors. Missing an audit site
  can expose stale q-label columns as fake sectors.
