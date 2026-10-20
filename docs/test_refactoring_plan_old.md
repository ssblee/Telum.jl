# Test refactoring plan

## Goal

Replace the current broad legacy test file with a small set of focused tests
that cover the tensor behaviors Telum relies on:

- real and complex element types;
- dense and `DiagRMT` reduced matrix tensor storage;
- operations mixing tensors whose inputs have different RMT storage types;
- Hermitian and non-Hermitian diagonalization;
- contraction, SVD, direct sums, slicing, and index metadata;
- large-leg and DMRG-derived tensors used as integration stress cases.

The refactor should keep tests simple. Each test should protect one invariant or
one user-visible behavior, and expensive DMRG-derived cases should live in a
small integration/stress layer rather than the normal unit-test path.

## Problems in the current suite

`test/test_main.jl` mixes several different responsibilities:

- local-space construction;
- index/tag/lock helpers;
- contraction correctness;
- SVD internals;
- eigenvalue behavior;
- zero-sector behavior;
- old storage assumptions and legacy row-style helper code;
- dense-array reference comparisons;
- ad hoc large-rank checks.

This makes it hard to add a new edge case without either duplicating setup or
making an already broad test more brittle. It also hides which tests are unit
tests and which tests are integration or performance-sensitive smoke tests.

Existing focused files should remain:

- `test/test_oplus.jl`;
- `test/test_qlabel_orientation.jl`;
- `test/test_row_subset.jl`;
- `test/test_hdf5_io.jl`;
- `test/common.jl` for shared references and factories.

The main refactor should move behavior-specific tests out of
`test/test_main.jl` into focused files.

## Test classification

Classify tests by behavioral layer, not by function name. A single public
function can touch multiple layers, but each test should state which layer it is
protecting.

Recommended layout:

```text
test/common.jl

test/test_localspace.jl
test/test_index_metadata.jl
test/test_tlarray_storage.jl
test/test_tensor_algebra.jl
test/test_contraction.jl
test/test_factorization.jl
test/test_decomposition.jl
test/test_direct_sum.jl
test/test_io.jl

test/test_integration_small.jl
test/test_stress_large.jl
```

Responsibilities:

- `test_localspace.jl`:
  `SpinOptions`, `FermionOptions`, `FermionSOptions`, generated operator names,
  symmetry info, and cross-symmetry commutation validation.
- `test_index_metadata.jl`:
  `TLIndex`, tags, primes, locks, directions, `findleg`, `findlegs`,
  `matching`, `contractable`, and `uncontractable`.
- `test_tlarray_storage.jl`:
  TLArray constructor invariants, q-label orientation, sector defined/zero
  state, uniform concrete RMT type, `DiagRMT` storage rules, complex RMT payload
  construction, `empty_qspace`, `zero`, and `zero_qlabels`.
- `test_tensor_algebra.jl`:
  scalar multiplication, addition/subtraction, conjugation/adjoint, norm,
  `getsub`, `addSingleton`, `deleteSingleton`, and tensor product `⊗`.
- `test_contraction.jl`:
  TLArray contraction correctness, lock reduction, mismatched-space rejection,
  dense/diagonal RMT input pairs, complex contraction, no stored w-matrices for
  Abelian symmetries, multiple non-Abelian symmetries, and non-Abelian
  X-symbol/cache behavior only when it affects contraction semantics.
- `test_factorization.jl`:
  `svd`, `svd_leg`, reconstruction, truncation, `DiagRMT` singular-value tensor
  storage, and complex SVD.
- `test_decomposition.jl`:
  `eig`/`eigen`, Hermitian and non-Hermitian reconstruction, truncation/discard,
  eigenvalue tensor storage, and complex non-Hermitian cases.
- `test_direct_sum.jl`:
  `oplus`, vector and matrix direct sum, missing entries, dense/`DiagRMT`
  mixing, and space inference.
- `test_io.jl`:
  HDF5 round trips for dense, complex, diagonal, and undefined-zero-sector
  tensors.
- `test_integration_small.jl`:
  small workflows involving multiple subsystems, such as small MPS/MPO
  construction, contraction followed by SVD, or contraction followed by eigen.
- `test_stress_large.jl`:
  DMRG-derived large tensors, large-leg contraction/SVD smoke tests, and
  allocation/runtime-sensitive checks. This should be opt-in unless it is cheap.

The important split is:

- storage: the tensor is represented correctly;
- algebra: simple operations preserve values and metadata;
- contraction/factorization/decomposition: major numerical algorithms work;
- integration/stress: realistic workflows survive.

With this classification, `test/test_main.jl` should disappear after migration,
except as a temporary include shim.

## Proposed migration layout

Use one runner and small behavior files:

```julia
test/runtests.jl
test/common.jl
test/test_localspace.jl
test/test_index_metadata.jl
test/test_tlarray_storage.jl
test/test_tensor_algebra.jl
test/test_contraction.jl
test/test_factorization.jl
test/test_decomposition.jl
test/test_direct_sum.jl
test/test_io.jl
test/test_integration_small.jl
test/test_stress_large.jl
test/test_qlabel_orientation.jl
```

Keep `test/test_oplus.jl` as either `test/test_direct_sum.jl` or a compatibility
include during migration.

## Shared test factories

Put tensor construction helpers in `test/common.jl`. Prefer a few named
factories over ad hoc setup inside every test.

Recommended factories:

- `spin_space(; sym=:U1, spin=1)`:
  returns a small spin local space.
- `fermion_space(; channels=1, syms=(:U1, :SU2, nothing))`:
  returns a representative mixed Abelian/non-Abelian local space.
- `same_nonabelian_product_space()`:
  uses `FermionSOptions(2, :U1, :SU2, :SU2)` to generate tensors with repeated
  non-Abelian symmetry types.
- `different_nonabelian_product_space()`:
  uses `FermionSOptions(3, :U1, :SU2, :SU3)` to generate tensors with different
  non-Abelian symmetry types.
- `small_dense_tensor(; T=Float64, option=fermion_space())`:
  returns a nontrivial dense-RMT `TLArray`.
- `small_diag_tensor(; T=Float64)`:
  returns a tensor whose sectors use `DiagRMT`.
- `complex_copy(q)`:
  returns a value-equivalent tensor with `ComplexF64` RMT payloads.
- `nonhermitian_matrix_tensor(; T=Float64)`:
  returns a rank-2 TLArray with non-Hermitian sector RMTs.
- `large_leg_tensor(; nkeep=20, source=:dmrg)`:
  returns a large tensor generated from DMRG or from contraction of DMRG MPS/MPO
  tensors.

Factories should return concrete `TLArray` values and should avoid global
mutation. Expensive factories should be opt-in.

## Coverage matrix

Each operation should be checked across the smallest useful set of element and
storage variants.

| Operation | No symmetry | Real dense | Complex dense | DiagRMT | Dense + DiagRMT | Multiple non-Abelian symmetries | Large legs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| construction/invariants | yes | yes | yes | yes | no | yes | yes |
| `+`, `sum`, `oplus` | yes | yes | yes | yes | yes | yes | yes |
| contraction | yes | yes | yes | yes | yes | yes | yes |
| `svd` | yes | yes | yes | yes for `S` storage | mixed via inputs | yes | yes |
| `eig`/`eigen` Hermitian | yes | yes | yes | yes | no | yes | optional |
| `eig`/`eigen` non-Hermitian | yes | yes | yes | yes if supported | no | yes | optional |
| tag/lock/prime metadata | yes | yes | no | yes | no | yes | no |
| slicing/getsub | yes | yes | yes | yes | no | yes | yes |
| HDF5 I/O / save-load | yes | yes | yes | yes | no | yes | no |

Do not test every Cartesian product in normal CI. Instead, give each operation:

- one small real dense correctness case;
- one no-symmetry case, because every operation should also work when there are
  no q-label or w-matrix dimensions;
- one complex-valued case where conjugation, adjoint, or arithmetic matters;
- one `DiagRMT` preservation or materialization case;
- one mixed dense/diagonal input case only for operations that accept mixed
  inputs;
- one multiple-non-Abelian-symmetry case, because every operation should handle
  product symmetries with more than one stored w-matrix slot. Cover both
  repeated non-Abelian symmetry types via `FermionSOptions(2, :U1, :SU2, :SU2)`
  and different non-Abelian symmetry types via
  `FermionSOptions(3, :U1, :SU2, :SU3)`;
- one large-leg smoke test only when the operation has known scaling or
  sector-matching risk.

## Contraction tests

Keep contraction tests focused on three invariants:

1. Value correctness against `to_sparse_array` or another dense reference.
2. Metadata correctness: output legs, lock reduction, q-label spaces.
3. Storage correctness: zero sectors are not defined, result RMT types are
   concrete and expected.

Suggested testsets:

- `contract matches dense reference for real tensors`;
- `contract matches dense reference for complex tensors`;
- `contract handles dense/DiagRMT input pairs`;
- `contract does not store w-matrices for Abelian symmetries`;
- `contract handles multiple non-Abelian symmetries`;
- `contract rejects mismatched spaces`;
- `contract handles DMRG-derived large central tensors`.

For large tensors, use existing example code:

```julia
include("support/dmrg.jl")
MPS, _, _ = init_MPS(XYMPO(1.0, 8), 8)
large = MPS[4] * MPS[5]
```

Keep this out of the normal unit runner unless runtime is acceptable. A separate
stress runner such as `test/dmrg_runtests.jl` is better.

## SVD tests

SVD tests should avoid checking internal class details unless the internal
contract is the thing being protected.

Suggested testsets:

- `svd reconstructs real TLArray`;
- `svd reconstructs complex TLArray`;
- `svd truncation keeps largest singular sectors`;
- `svd S tensor uses DiagRMT when eligible`;
- `svd handles large DMRG-derived two-site tensor`;
- `svd stored-leg ordering follows TLArray directions and q-label order`.

For reconstruction, use:

```julia
result = svd(q, left_legs, "left", "right"; Nkeep=nothing)
@test norm(result.U * result.S * result.Vd - q) < tol
```

Use dense-array comparison only when TLArray arithmetic cannot express the
reference cleanly.

## Eigen tests

Split Hermitian and non-Hermitian paths explicitly.

Suggested testsets:

- `eigen Hermitian reconstructs A = V * D * V'`;
- `eigen non-Hermitian reconstructs A = V * D * Vinv`;
- `eigen autodetects Hermitian storage when requested`;
- `eigen truncation preserves selected sector dimensions`;
- `eigen produces DiagRMT D when eligible`;
- `eigen handles complex non-Hermitian sector RMTs`.

For non-Hermitian tests, do not symmetrize the RMT payload. Build a small
rank-2 tensor with sector matrices such as:

```julia
[1.0 2.0; 0.0 3.0]
```

and a complex variant:

```julia
ComplexF64[1 2im; 0 3 - im]
```

The test should verify reconstruction and expected eigenvalues per sector, not
only that the function returns.

## Sum and mixed RMT storage

Direct sum and tensor sum tests should cover the cases that are easy to break:

- dense + dense;
- diagonal + diagonal;
- dense + diagonal, where the result must have one concrete RMT storage type;
- complex + real promotion;
- missing entries in matrix `oplus`.

When dense and diagonal inputs are mixed, assert both:

- dense numerical value matches reference;
- resulting `RMTs` container has a concrete element type, not `Vector{Any}` or
  `Vector{AbstractArray}`.

## Large tensor validation

Large tests should be opt-in and generated, not stored as fixtures.

Two useful generators:

1. DMRG MPS/MPO tensors:

```julia
MPO = XYMPO(1.0, 8)
MPS, _, _ = init_MPS(MPO, 8)
two_site = MPS[4] * MPS[5]
```

2. Repeated contraction/direct-sum growth:

```julia
q = small_dense_tensor()
large = oplus([q, q, q, q], (1,))
```

Large tests should check structural invariants and absence of errors:

- `sector_count(q)` is nonzero;
- all active sectors have defined RMTs;
- no active sector stores a zero RMT;
- all defined RMTs have the same concrete type;
- selected operations return within reasonable memory/runtime.

Do not compare full dense arrays for very large tensors unless the dense
reference is intentionally small enough.

## Migration order

1. Create `test/test_indices.jl` and move tag, lock, prime, matching, and
   contractable tests out of `test_main.jl`.
2. Create `test/test_localspaces.jl` and move local-space construction and
   validation tests.
3. Create `test/test_eig.jl` and move eigen tests. Add the missing
   non-Hermitian real and complex cases.
4. Create `test/test_svd.jl` and move SVD tests. Keep only one or two internal
   CGTSVD tests that protect real invariants.
5. Create `test/test_contraction.jl` and move contraction tests. Add
   dense/diagonal mixed input coverage.
6. Create `test/test_storage.jl` for TLArray sector-state and RMT storage
   invariants.
7. Keep expensive DMRG/large-leg tests in `test/dmrg_runtests.jl` or another
   opt-in stress runner.
8. Delete migrated sections from `test/test_main.jl`; keep it only as a
   temporary include shim during migration.

## Test naming

Name testsets after behavior, not implementation detail:

- good: `"contract handles dense and diagonal RMT inputs"`;
- good: `"eigen reconstructs non-Hermitian complex tensors"`;
- bad: `"calls _contract_RMT_pair_into! with DiagRMT"`.

Internal helper tests are allowed when the helper encodes a core invariant, but
they should be grouped separately and kept minimal.

## Validation policy during refactor

Because this is an internal refactor of tests first:

- do not change production behavior while moving tests;
- run `julia --project=Telum Telum/test/runtests.jl` after each migration
  step;
- run `julia --project=Telum Telum/test/dmrg_runtests.jl` only when production
  code changes or when adding DMRG/stress coverage;
- after the refactor, use the focused files to decide which subset to run for
  future code changes.

For later production changes touching contraction, SVD, eigen, RMT storage, or
index direction logic, run the corresponding focused test file plus the DMRG
validation requested in `AGENTS.md`.

## Required failure report

At the end of every test-refactoring step:

1. Run the newly created or newly moved tests.
2. List every test that does not pass.
3. For each failing test, include the testset name, file path, failure message,
   and whether the failure is caused by the refactor or by pre-existing
   behavior.

This is the most important requirement of the test refactor. A migration step
is not complete until the new tests have been run and every non-passing test is
explicitly reported.
