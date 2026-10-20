# TLArray QR decomposition plan

## Goal

Add a symmetry-adapted QR decomposition for `TLArray`, shaped like the current
CGTSVD-based path in `src/svd.jl`.

The target API should factor a tensor across a left/right leg partition:

```julia
result = qr(q, left_legs, bond_tag="qr")
Q = result.Q
R = result.R
```

with reconstruction:

```julia
Q * R ≈ q
```

after the same leg-order normalization used by the SVD tests.

Leg selection is required. Calling `qr(q)` without either explicit `left_legs`
or keyword selectors is an error, matching the SVD requirement that the
decomposition partition is specified by the caller.

The first implementation should prioritize correctness, type stability, and
readability. Truncation is not part of QR, so the implementation can be simpler
than `svd_std` after class matrices are assembled. Keep the QR implementation in
its own file even when the algorithm follows the same conventions as SVD; avoid
cross-file helper reuse when it makes the QR control flow harder to read.

## Proposed factor convention

For `q::TLArray` of rank `QD`, with `left_legs::NTuple{L,Int}` and
`right_legs::NTuple{QD-L,Int}`:

- `Q` legs: original left legs followed by the new bond leg.
- `R` legs: the new bond leg followed by original right legs.
- `Q` bond index: `TLIndex(bond_tag, '-')`.
- `R` bond index: `TLIndex(bond_tag, '+')`.
- `Q.spaces[end]` and `R.spaces[1]` should use the same bond space list.
- Original physical legs keep their `TLIndex` metadata and spaces.

This makes the direct contraction `Q * R` match the SVD convention already used
by `U * S` in `svd_std`: an outgoing bond on the left factor contracts with an
incoming bond on the right factor.

The QR right factor is not the same object as SVD's `Vd`. `R` represents the
fused right-side factor with legs `(bond, unselected legs)`, so its CGT metadata
must be derived from the QR two-CGT factorization for this direct `Q * R`
convention rather than copied from the SVD `Vd` assembly.

Use a single `bond_tag` for QR. Unlike SVD, there is no middle diagonal tensor
and no second bond interface, so separate left/right tags would be redundant.

## LQ convention

Do not implement a separate LQ decomposition path initially. LQ is QR with the
complementary leg selection and a different factor-order convention:

- `qr(q, left_legs)` produces `Q(left_legs, bond) * R(bond, right_legs)`.
- `lq(q, left_legs)` can be a thin wrapper around QR on `right_legs`, returning
  the factors reordered as `L(left_legs, bond) * Q(bond, right_legs)`.

The wrapper still needs tests because TLArray leg directions, dual qlabels, and
CGT metadata must be adjusted consistently. It should not need separate
CGT-side machinery or a duplicated dense decomposition implementation.

## Public API

Add `LinearAlgebra.qr` overloads rather than a package-specific name. Support
two public call forms:

1. explicit leg selection:
   `qr(q, left_legs, bond_tag="qr")`;
2. keyword leg selection:
   `qr(q, bond_tag="qr"; dir=nothing, itag=nothing, plev=nothing,
   lock=nothing, rev=false)`.

```julia
struct QRResult{TQ, TR}
    Q::TQ
    R::TR
end

function LinearAlgebra.qr(q::TLArray, left_legs, bond_tag::AbstractString="qr")
    left_legs_ = Tuple(_normalize_qr_left_legs(left_legs, ndims(q)))
    return qr_std(q, left_legs_, bond_tag)
end
```

The keyword wrapper should have a method signature that cannot be confused with
the explicit `left_legs` method. If no selector keyword is supplied, it should
throw an `ArgumentError`.

```julia
function LinearAlgebra.qr(q::TLArray, bond_tag::AbstractString="qr";
                          dir=nothing, itag=nothing, plev=nothing,
                          lock=nothing, rev=false)
    left_legs = _select_qr_left_legs(q; dir=dir, itag=itag, plev=plev,
                                     lock=lock, rev=rev)
    return qr(q, left_legs, bond_tag)
end
```

`QRResult` is justified even under the "minimal type" rule because it mirrors
`SVDResult`, gives callers named fields, and avoids overloading a raw tuple
whose order is less self-documenting. It carries no storage invariant beyond
returning the two factors.

## Internal typed entry

Follow the SVD function-barrier pattern:

```julia
function qr_std(
    q::TLArray{T,QD,N,RD,QT,PS,M,RMT},
    left_legs::NTuple{L,Int},
    bond_tag::AbstractString = "qr",
) where {T,QD,N,RD,QT,PS,M,RMT,L}
```

Inside `qr_std`:

1. assert `left_legs` is sorted and unique;
2. compute `right_legs::NTuple{QD-L,Int}` with a QR-local helper matching
   SVD's right-leg convention;
3. use typed `CGTQR` caches to enumerate QR split rows;
4. assemble class matrices with the same Telum-side grouping shape as SVD;
5. run dense QR per class matrix;
6. build `Q` and `R` TLArrays from the QR class results.

Keep the public `qr` methods as normalization wrappers only. Do not let vector
or predicate-selected legs reach the hot implementation.

## Standalone implementation in `src/qr.jl`

Write the Telum QR implementation in `src/qr.jl`. Do not reuse or rename SVD
helpers from `src/svd.jl` in the first implementation. QR may duplicate small
normalization, ordering, qlabel, side-metadata, row-grouping, and TLArray
assembly helpers when that keeps the QR code readable and locally auditable.

The duplicated QR helpers should still follow the SVD conventions exactly where
those conventions affect observable behavior:

- left-leg normalization and selected-leg sorting;
- right-leg complement order;
- physical qlabel and product-key construction;
- CGT/TLArray stored-leg ordering;
- class sorting and sector ordering;
- dual-sector convention for the right factor, when required by contraction;
- non-Abelian irrep dimension scaling, if QR needs it for isometry.

Keep the QR helpers prefixed with `_qr_`. Avoid broad shared-helper refactors
until QR is working, tested, and the duplicated code has stabilized.

## Dense per-class QR

SVD currently builds one dense matrix per intermediate product sector class in
`_build_svd_cgtsvd_class`, then runs:

```julia
F = svd(mat; full=false)
```

QR should use the same class-matrix layout as SVD, implemented with QR-local
helpers, and replace only the dense factorization step:

```julia
F = LinearAlgebra.qr(mat)
k = min(size(mat)...)
Qmat = Matrix(F.Q[:, 1:k])
Rmat = Matrix(F.R[1:k, :])
```

Use economy QR. The bond dimension for each class is `k = min(m, n)`. Do not
materialize `Matrix(F.Q)` before slicing, because that can allocate the full
`m x m` orthogonal/unitary factor for tall matrices. `F.Q[:, 1:k]` is sufficient
for the first implementation; consider a LAPACK-backed economy helper only if
benchmarks show this extraction is too slow or allocates too much for
representative class sizes.

Record class results with enough metadata to assemble the two TLArrays:

```julia
struct _QRCGTClassResult{QT, LI, RI, Q, R}
    sector::QT
    left_infos::LI
    right_infos::RI
    Q::Q
    R::R
end
```

This internal type is justified if QR follows the SVD class-result assembly
pattern:

1. A function-only return would otherwise be a large tuple threaded through
   multiple hot assembly helpers.
2. It represents one QR factorization result for one product intermediate
   sector class.
3. It improves dispatch and vector element concreteness:
   `Vector{_QRCGTClassResult{...}}`.
4. Type stability can be checked with `@code_warntype qr(q, left_legs)`.
5. It is better than reusing `_SVDCGTClassResult`, whose `S`/`Vt` fields encode
   SVD-specific meaning.

If the implementation can keep QR class assembly in one function without
threading results through multiple helpers, prefer a local typed tuple and skip
this struct.

## Bond-space assembly

SVD derives kept bond ranges from singular-value selection. QR has no
selection step, so construct ranges directly:

1. For each class result, `k = size(result.Q, 2)`.
2. Group class bond ranges by `result.sector`.
3. Sort sectors with the same convention as SVD before assigning ranges.
4. Assign contiguous ranges within each sector in sorted class order.
5. Build:

```julia
bond_splist = [(sector, total_k_for_sector), ...]
```

with the same sector ordering as SVD. Do not assign ranges in one order and then
sort `bond_splist` in another order; `bond_splist`, class ranges, `qlabels`,
RMT payloads, and w-matrix metadata must all use the same sector ordering.

`Q` stores each left-info block as a matrix slice:

```julia
part = result.Q[info.range, :]
full = zeros(eltype(result.Q), length(info.range), sector_count)
full[:, class_range] = part
```

then reshapes/permutes with the same pattern as SVD `U`.

`R` stores each right-info block as:

```julia
part = result.R[:, info.range]
full = zeros(eltype(result.R), sector_count, length(info.range))
full[class_range, :] = part
```

then reshapes as the QR right factor with legs `(bond, right_legs...)`. Do not
copy SVD `Vd`'s dual-sector metadata. For the planned direct `Q * R`
convention, `R.spaces[1]` and `Q.spaces[end]` use the same bond space list, and
the QR right-side CGT metadata must be generated consistently with that choice.

Validate the bond qlabel convention with reconstruction tests before relying on
it elsewhere. If implementation shows contraction requires a dual-sector bond on
`R`, adjust both `R.spaces[1]` and `R` CGT metadata together and update this
factor convention.

## Non-Abelian normalization

Use the same normalization convention as `CGTSVD` and Telum SVD, adapted to the
single-bond QR split.

On the CGT side, `CGTQR` should be unitary. Scale the X-symbol split-basis blocks
by `sqrt(dimq)`, where `dimq` is the irrep dimension of the QR bond sector. This
matches the CGTSVD split-basis construction and keeps the CGT basis transform
unitary in the intended inner product.

On the Telum RMT side, after applying dense QR to each concatenated class matrix,
multiply the dense `Q` part by `sqrt(dimq)` for that QR bond sector. The dense
`R` part is not given the same factor. This mirrors the SVD convention, where
the non-Abelian dimension factor is carried by the side tensor factors rather
than treated as an ordinary dense matrix singular-value normalization.

For reference, SVD applies per-sector degeneracy scaling around the dense SVD:

```julia
scale = sqrt(dimq)
U = F.U .* scale
S = F.S ./ scale
Vt = F.Vt .* scale
```

QR must preserve reconstruction and the desired isometry convention. Start with
the same class matrix assembly and the QR scaling above:

```julia
Q = Qmat .* sqrt(dimq)
R = Rmat
```

Then check:

- `Q * R ≈ q`;
- `Q' * Q ≈ I` on the fused QR bond;
- behavior for `SU2`, `U1`, product symmetries, and no symmetry.

Document this invariant in code where the CGTQR X-symbol scaling and dense
`Q`-part scaling are applied. If reconstruction or isometry fails, first check
that the same `dimq` convention and bond-sector ordering are used on both sides
before changing the scaling rule.

## `CGTQR` is expected

Implement QR with a separate CGT-side decomposition object, tentatively named
`LurCGT.CGTQR`.

The reason is structural, not cosmetic. `CGTSVD` decomposes the original CGT
into three CGT pieces:

1. a left CGT;
2. an all-incoming 1j bridge carrying `q` and `dual(q)`;
3. a right CGT.

That three-piece construction is appropriate for SVD because the decomposition
has two bond interfaces around the diagonal `S` tensor. QR has only one bond
between `Q` and `R`, so its CGT-side factorization should be a two-piece split:

1. a left `Q`-side CGT with the QR bond;
2. a right `R`-side CGT with the same bond interface.

Reusing `CGTSVD` would force QR through an SVD-specific bridge basis and row
ordering. Even if reconstruction could be made to work, it would encode the
wrong invariant and make QR assembly harder to reason about.

`CGTQR` should represent exactly the CGT-only basis transform needed to rotate a
canonical CGT part basis, then split it into `Q`-side and `R`-side CGT factors
for a fixed leg partition. Build it with the same broad construction pattern as
`CGTSVD`, but with QR-specific semantics:

1. enumerate the ordered list of valid QR bond qlabels for the normalized leg
   partition, following SVD's sector ordering convention;
2. get the required X-symbols for the left/right split basis;
3. scale the X-symbol split-basis blocks by `sqrt(dimq)` so `CGTQR` is unitary;
4. concatenate the split-basis blocks by bond qlabel and outer multiplicity;
5. apply the possible stored-leg permutation so the CGT-side order matches the
   TLArray/CGT leg-order invariant;
6. rotate the CGT part basis with the resulting transform;
7. divide the rotated basis into QR left and right payloads in the same spirit
   as SVD, but without introducing an SVD-specific 1j bridge.

It should store concrete metadata comparable to `CGTSVD`, but with QR-specific
semantics:

```julia
struct CGTQR{S<:NonabelianSymm, U, D, L, NZ}
    qr_arr::Matrix{Float64}
    upsp::NTuple{U, NTuple{NZ, Int}}
    dnsp::NTuple{D, NTuple{NZ, Int}}
    leftlegs::NTuple{L, Int}
    bond_sps::Vector{Tuple{NTuple{NZ, Int}, Int, Int}}
    size_byte::Int
end
```

The final field names may change after deriving the exact two-CGT data layout.
The important requirement is that `bond_sps` describes the QR bond sectors and
outer-multiplicity ranges needed to assemble the TLArray `Q` and `R` factors,
and that the object exposes enough typed payload metadata for Telum to build the
left and right side CGT metadata without rediscovering row ordering.

The core payload contract should mirror the useful part of `CGTSVD`: `qr_arr`
maps original CGT basis rows to concatenated `(left_om, right_om)` split-basis
rows ordered by `bond_sps`. Store it with split-basis rows as the first matrix
axis and original canonical CGT rows as the second matrix axis, matching
`CGTSVD`'s `basis_rows[split_row, original_row]` orientation. Telum should be
able to use this ordering to derive QR split rows, left payloads, and right
payloads deterministically. For each entry `(q, omL, omR)` in `bond_sps`, the
corresponding block has `omL * omR` rows in the concatenated split basis, with
`left_om` varying fastest and `right_om` varying next, following the existing
CGTSVD `reshape(sector_rows, omL * omR, total)` convention.

The new type is justified under the repository type policy:

1. A function-only patch is insufficient because `CGTSVD` encodes a
   three-CGT/1j-bridge split, while QR needs a two-CGT split and different
   cached metadata.
2. The invariant represented by `CGTQR` is the QR-specific CGT basis transform
   and bond-sector ordering for one symmetry, one canonical `(upsp, dnsp)`, and
   one normalized leg partition.
3. It improves memory layout and dispatch by giving Telum typed cache
   dictionaries such as `Dict{Key, Union{Bool, LurCGT.CGTQR{S}}}` instead of
   unstructured tuples or `Any`.
4. Type stability should be checked with `@code_warntype qr(q, left_legs)` and
   with the generated CGTQR cache constructor.
5. This is better than a minimal function-level patch because the CGT
   decomposition itself differs; sharing the SVD object would preserve the
   wrong mathematical structure.

Add `CGTQR` in `LurCGT/`, preferably as a new `src/CGTQR.jl` included from
`src/Base.jl` and exported from `src/LurCGT.jl`. Add SQLite cache support
parallel to `CGTSVD`, with separate keys such as `cgtqr_v1_...`.

## No-symmetry and Abelian cases

Construct QR through the same QR-local general path for all symmetry counts,
including `N == 0`. Do not add a separate no-symmetry fast path in the first
implementation.

The general path must handle the empty product symmetry
`ProductSymm{Tuple{}}`, empty qlabels `()`, empty non-Abelian w-matrix tuples,
and empty CGT payload products without indexing into a nonexistent symmetry
slot. For Abelian symmetries, no CGT payload matrices are needed. Implement
QR-local `Bool`/trivial-isometry branches equivalent to the SVD branches.

## Files to change

Likely Telum changes:

- `src/qr.jl`: QR result type, public overloads, typed `qr_std`, QR-local
  normalization, selection, ordering, class assembly, and TLArray assembly
  helpers.
- `src/Telum.jl`: include `qr.jl` and export any package-level names if needed.
  `LinearAlgebra.qr` itself does not need export.
- `src/svd.jl`: no changes in the first QR implementation.
- `test/test_factorization.jl` or `test/test_utils.jl`: focused QR regression
  tests.
- `test/test_tlarray_structure.jl`: keyword leg-selection tests if the API
  mirrors SVD selection.

`LurCGT/` must change before the non-Abelian Telum QR path can be completed:
add `CGTQR`, `getNsave_CGTQR`, and cache support. Telum should still construct
one general QR path that handles `N == 0`, Abelian, and non-Abelian cases.

## Test plan

Add behavior tests before or with implementation:

1. No symmetry:
   `getLocalSpace(SpinOptions(nothing, 1))`, build a small contracted tensor,
   run `qr`, check reconstruction.
2. Abelian:
   `SpinOptions(:U1, 1)` and a fermion local space, check reconstruction and
   bond dimensions.
3. Non-Abelian:
   `SpinOptions(:SU2, 1)` and one `FermionSOptions(1, :U1, :SU2, nothing)`
   case, check reconstruction.
4. Product non-Abelian:
   if runtime is acceptable, include a small `FermionSOptions(3, :U1, :SU2,
   :SU3)` smoke or mark it as a focused slower test.
5. Keyword selection:
   mirror the SVD tests for `dir`, `itag`, `plev`, `lock`, and `rev`.
6. Error handling:
   empty left leg list, all legs selected, duplicate legs, out-of-range legs.

Also add `N == 0` coverage for existing Telum operations that QR relies on or
will be compared against:

- contraction against sparse/dense reference;
- sum/addition against sparse/dense reference;
- any SVD/factorization helper behavior used as a QR convention reference.

For reconstruction, check `norm(q - Q * R)` directly. TLArray subtraction
matches the legs of the two tensors before subtracting. Also check sector-state
invariants:

- no stored zero RMT payloads;
- all defined RMTs in each output share one concrete RMT type;
- no `isdefined && iszero` sector.

Dense QR is not rank-revealing, and QR factors have implementation-dependent
gauge/sign/phase choices. Tests should therefore:

- check per-class QR bond dimensions against economy QR, `k = min(m, n)`, not
  numerical rank;
- avoid exact entrywise expectations for `Q` or `R`; check reconstruction,
  isometry, dimensions, ordering, and sector invariants instead.

## Type stability and performance checks

Run focused checks first:

```julia
julia --project=Telum -e 'using Telum, LinearAlgebra; include("Telum/test/common.jl"); q = getLocalSpace(SpinOptions(:SU2, 1)); A = q.S; @code_warntype qr(A, (1,))'
julia --project=Telum -e 'using Telum, LinearAlgebra; include("Telum/test/common.jl"); q = getLocalSpace(SpinOptions(:SU2, 1)); A = q.S; @code_warntype Telum.qr_std(A, (1,), "qr")'
```

Then run relevant tests:

```julia
julia --project=Telum Telum/test/runtests.jl
```

If QR is used in DMRG or contraction-heavy paths later, add an allocation
comparison against SVD-like setup on representative tensors. Keep row
enumeration and class matrix assembly structured as independent per-sector
work so later parallelization does not require a rewrite.

## Implementation phases

### Phase 1: LurCGT `CGTQR`

- Add `LurCGT/src/CGTQR.jl`.
- Define `CGTQR` with concrete fields and cached `size_byte`.
- Implement `getNsave_CGTQR` and a standardized internal entry.
- Build the two-CGT split basis and `bond_sps` for a normalized leg partition
  by enumerating bond qlabels, getting X-symbols, scaling and concatenating
  basis blocks, applying any required permutation, rotating the CGT part basis,
  and dividing it into QR-side payloads.
- Add SQLite cache support with separate `cgtqr` keys.
- Add focused LurCGT tests for small `SU2` and `SU3` split cases.

### Phase 2: Minimal Telum QR

- Add `src/qr.jl`.
- Add `QRResult`.
- Add public `qr` wrappers and `qr_std`.
- Add QR split preassembly using typed `CGTQR` caches.
- Build `Q` and `R`.
- Add reconstruction tests for no symmetry (`N == 0`), `U1`, and `SU2`.

### Phase 3: Isometry and normalization validation

- Add `Q' * Q` tests where an identity comparison is already available.
- Confirm non-Abelian degeneracy scaling.
- Adjust scaling only if reconstruction or isometry fails.

### Phase 4: Cleanup review

- Review duplicated QR/SVD helper logic only after QR is working and covered.
- Keep QR readability as the default; do not merge helpers if the shared form is
  harder to audit.
- Do not change public SVD behavior.
