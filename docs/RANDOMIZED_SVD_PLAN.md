# Randomized SVD plan

## Goal

Add an opt-in randomized SVD path to Telum's existing symmetry-aware
`LinearAlgebra.svd(q, ...)` API. The exact implementation remains the default.
The initial use case is a large tensor split where only a small bond dimension
is required and a controlled approximation is preferable to computing every
singular vector.

## Proposed API

Add the following keywords to both public `svd` overloads and forward them
through `svd_cgtsvd`:

```julia
svd(q, left_legs; randomize=false, random_mode=:proportional,
    random_tol=0.1, Nkeep=nothing, random_niter=2, random_rng=Random.default_rng(), ...)
```

- `randomize=false` preserves today's exact behavior and default results.
- `randomize=true` selects randomized partial SVD for eligible dense
  SVD-class matrices.
- `random_mode` chooses the candidate allocation policy when randomization is
  enabled. It defaults to `:proportional`; `:per_class` is the alternative.
- `random_tol` is the fractional extra candidate budget for
  `random_mode=:proportional`; the default `0.1` requests approximately 10%
  extra candidates across classes. This is an oversampling fraction, not the
  existing SVD `cutoff`. It has no effect when `random_mode=:per_class`.
- `Nkeep` supplies the requested global cap and is required when
  `randomize=true`; otherwise there is no target rank with which to reduce the
  amount of work. It must be positive in randomized mode.
- Under `random_mode=:per_class`, each class requests up to `Nkeep` candidates.
- `random_niter` sets the number of power iterations, with a modest default
  (initially 2).
- `random_rng` supplies the random-number generator, so callers can make
  results repeatable with a seeded RNG.

When `randomize=false`, ignore all `random_*` options; they must not change
exact SVD behavior or trigger randomized-option validation. When
`randomize=true`, validate the randomized options and require `Nkeep > 0`.

In randomized mode, validate positive `Nkeep`, positive `random_tol`, and
nonnegative `random_niter`. Reject `randomize=true` with `Nkeep=nothing` using
an explanatory `ArgumentError`. Validate `random_mode` against `:proportional`
and `:per_class`. The keyword-only leg-selector overload must forward all
randomized options. Use `randomize` as the enable flag. All other randomized SVD options
start with `random_`; `Nkeep` remains the existing shared SVD keyword. For each
class matrix `Aᵢ` with dimensions `mᵢ × nᵢ`, define its available singular-value
count as `dᵢ = min(mᵢ, nᵢ)`. Derive these counts from class metadata before
constructing the dense class matrices. In proportional mode, request the
`ceil(Nkeep * (1 + random_tol) * dᵢ / sumⱼ(dⱼ))` largest singular values from
class `i`, capped at `dᵢ`. Per-class rounding up can make the summed requests
slightly greater than the nominal `Nkeep * (1 + random_tol)` budget. In
per-class mode, request the `min(dᵢ, Nkeep)` largest singular values from every
class. Per-class mode is the accuracy-oriented policy: each class has enough
candidate slots to contribute to the global top `Nkeep`, though randomized
approximation can still affect accuracy. Use exact SVD for a class when its
requested candidate count reaches `dᵢ`. Sort combined candidates and return at
most the globally selected `Nkeep` values and vectors, subject to existing
cutoff behavior.

## Matrix-level library choice

Prefer `LowRankApprox.jl`'s partial SVD for the matrix-level randomized
factorization rather than maintaining a new randomized SVD implementation in
Telum. It exposes a partial SVD API and supports sketching and power iterations.
Do not use `RandomizedLinAlg.jl`'s `rsvd` as the default: its documentation
warns that the routine may return large singular pairs without reliably
finding the largest ones. Before implementation, verify that the selected
LowRankApprox API supports the planned rank, power-iteration count, complex
matrices, and explicit RNG. If it does not, use a small Telum helper based on
Gaussian sketching, QR orthogonalization, and exact SVD of the projected
matrix.

## Algorithm and symmetry boundary

The decomposition pipeline in `src/svd.jl` first forms dense matrices per
symmetry/CGT class and currently computes an exact SVD for each class before
applying the global `cutoff`/`Nkeep` selection. Randomization should operate
inside that existing class boundary: it must not mix inequivalent symmetry
sectors or change CGT metadata construction. For each eligible class matrix
`A`, use a randomized range finder (Gaussian sketch, optional power iterations,
orthogonalization), compute the small projected matrix, and take its exact SVD.
Feed the resulting approximate singular values and factors into the existing
selection and output-building stages.

Because `Nkeep` is global while randomized subspaces are local to classes,
allocation affects both cost and the candidate values available to global
selection. Per-class allocation requests up to `Nkeep` candidates from each
class and guarantees enough candidate slots for each class to contribute
to the global top `Nkeep`, subject to randomized approximation error.
Proportional allocation is the default and can be much faster, but may allocate fewer than `Nkeep` candidates to a class and
miss large singular values from that class. Skip zero class matrices. As in
the existing path, singular values smaller than `10^-12` times the largest
singular value are automatically truncated; empty classes contribute no
candidates. `Nkeep=0` is invalid when `randomize=true`.

## Unsymmetric tensor path

The feature also applies to unsymmetric tensors. Route its dense matrix
factorization through the randomized matrix-SVD helper; changing only the
symmetric CGT class call site is insufficient. Preserve the unsymmetric path's
existing factor shapes, index and bond conventions, cutoff behavior, `Nkeep`
behavior, `get_lists` semantics, and input nonmutation.

For an unsymmetric tensor, use the randomized method to find the global
`Nkeep` largest singular values and corresponding vectors of its dense matrix.
Add integration coverage for both symmetric and
unsymmetric inputs so enabling randomization does not leave one path on exact
SVD by accident.

## Accuracy and truncation semantics

- `cutoff` continues to be applied to the candidate singular values using the
  existing selection logic, but candidates from randomized decompositions are
  approximate.
- `Nkeep` remains a global maximum across classes, not a per-class rank.
- If cutoff removes candidates, return fewer than `Nkeep`; do not grow the
  randomized candidate pool to replace them.
- When `get_lists=true`, `kept_list` and `trunc_list` contain only computed
  singular-value candidates. Computed values excluded by cutoff or global
  `Nkeep` selection may appear in `trunc_list`; singular values outside the
  randomized candidate pool were never computed and appear in neither list.
- Reconstructing `U * S * Vd` is approximate when randomization is active.
- The randomized path must preserve output shapes, index tags, bond-sector
  conventions, `get_lists` structure, and nonmutation of the input.
- Explain that `random_tol`, `random_niter`, and `random_rng` affect
  approximation quality and reproducibility, while exact mode remains deterministic under the
  existing dense linear algebra backend.

## Implementation outline

1. Extract the dense class SVD into a helper that accepts either exact mode or
   randomized options, without changing assembly of symmetry-aware classes.
2. Implement/test a randomized matrix SVD helper for real and complex element
   types, rank-deficient and zero matrices, and dimensions smaller than the
   requested rank. Use QR/orthogonalization at each power iteration.
3. Thread `randomize`, `random_mode`, `random_tol`, `random_niter`, and
   `random_rng` through both public overloads, `svd_std`, and the compatibility
   alias forwarding path.
4. Preserve the existing global candidate selection, accounting for the
   per-class sketch rank policy described above.
5. Document the API and approximation guarantees in `docs/src/tensor-operations.md`
   and the quick reference after behavior is implemented.

## Verification plan

- Exact mode regression: existing SVD reconstruction, singular-value,
  truncation, keyword-selector, and alias tests remain unchanged.
- Randomized dense/class helper: compare leading singular values and
  reconstructed low-rank approximation against exact SVD on matrices with a
  clear spectral gap; cover real and complex input, seeded RNG repeatability,
  zero/rank-deficient matrices, and dimensions below the sketch rank.
- TLArray integration: verify sector structure and tags, `U/S/Vd` bond
  conventions, approximate reconstruction error, global `Nkeep`, cutoff
  bookkeeping, `get_lists`, and input nonmutation.
- Invalid options: check missing `Nkeep`, nonpositive randomized `Nkeep` or
  `random_tol`, and negative `random_niter` produce clear errors.
- Performance check: compare exact and randomized paths on representative
  large, low-rank cases; do not promise speedups for small or full-rank blocks.

## Allocation limitation

The first release defaults to proportional allocation with no adaptive
candidate growth. Users can select `random_mode=:per_class` for the
accuracy-oriented candidate pool. Global top-`Nkeep` selection in proportional mode
is approximate: a class may have important singular values outside its
computed candidate pool. Confirm LowRankApprox's rank, iteration, complex
matrix, and RNG support before implementation; if any are unavailable, use
the small Telum helper described above.
