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
    random_tol=0.1, Nkeep=nothing, random_niter=2, ...)
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

When `randomize=false`, ignore all `random_*` options; they must not change
exact SVD behavior or trigger randomized-option validation. When
`randomize=true`, validate the randomized options and require `Nkeep > 0`.

In randomized mode, validate positive `Nkeep`, positive `random_tol`, and
nonnegative `random_niter`. Reject `randomize=true` with `Nkeep=nothing` using
an explanatory `ArgumentError`. Validate `random_mode` against `:proportional`
and `:per_class`. The keyword-only leg-selector overload must forward all
randomized options. For each class matrix `Aᵢ` with dimensions `mᵢ × nᵢ`,
define its available singular-value count as `dᵢ = min(mᵢ, nᵢ)`. These counts
must be derived from class metadata before constructing the dense class
matrices. In proportional mode, request
`ceil(Nkeep * (1 + random_tol) * dᵢ / sumⱼ(dⱼ))` candidates from class `i`, then
cap the request at `dᵢ`. Per-class rounding up can make the summed requests
slightly greater than the nominal `Nkeep * (1 + random_tol)` budget. In
per-class mode, request `min(dᵢ, Nkeep)` from every class. Use exact SVD for a
class when its requested candidate count reaches `dᵢ`.

## Matrix-level library choice

Prefer `LowRankApprox.jl`'s partial SVD for the matrix-level randomized
factorization rather than maintaining a new randomized SVD implementation in
Telum. It exposes a partial SVD API and supports sketching and power iterations.
Do not use `RandomizedLinAlg.jl`'s `rsvd` as the default: its documentation
warns that the routine may return large singular pairs without reliably
finding the largest ones. Before implementation, verify that the selected
LowRankApprox API supports the planned rank, power-iteration count, and
complex matrices.

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
using `Nkeep + oversampling` as the rank for every class can still be costly
and can over-allocate. The implementation should first establish a safe,
documented allocation policy. A practical first version can cap each class
sketch rank by `min(size(A)..., Nkeep + oversampling)` and retain the current
global selection across the approximate candidates. This is a valid candidate
pool but may miss globally important values if a class needs more candidates;
measure and document this limitation, and consider adaptive per-class growth
until the global top `Nkeep` is stable in a later iteration.

Per-class allocation requests at least `Nkeep` candidates from each class and
therefore gives every class enough candidates to contribute to the global top
`Nkeep`. Proportional allocation is the default and can be much faster, but
may allocate fewer than `Nkeep` candidates to a class and miss large singular
values from that class. Zero matrices and empty classes should follow existing
exact-path output conventions. `Nkeep=0` is invalid when `randomize=true`.

## Unsymmetric tensor path

The feature also applies to unsymmetric tensors. The implementation must locate
and route the unsymmetric tensor SVD's dense matrix factorization through the
same randomized matrix-SVD helper; changing only the symmetric CGT class SVD
call site is insufficient if the unsymmetric path factors its matrix
separately. Preserve the unsymmetric path's existing factor shapes, index and
bond conventions, cutoff behavior, `Nkeep` behavior, `get_lists` semantics,
and input nonmutation.

Use the unsymmetric path's actual factorization units for candidate allocation.
If it produces one dense matrix, treat that as one class with
`d = min(m, n)`: proportional mode requests up to
`ceil(Nkeep * (1 + random_tol))` candidates, capped at `d`, while per-class
mode requests up to `Nkeep`, also capped at `d`. If the path has multiple
independent blocks, apply the same per-block `dᵢ` allocation formula used for
symmetry classes. Add integration coverage for both symmetric and unsymmetric
inputs so enabling randomization does not leave one path on exact SVD by
accident.

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
- Explain that `oversampling`, `niter`, and `rng` affect approximation quality
  and reproducibility, while exact mode remains deterministic under the
  existing dense linear algebra backend.

## Implementation outline

1. Extract the dense class SVD into a helper that accepts either exact mode or
   randomized options, without changing assembly of symmetry-aware classes.
2. Implement/test a randomized matrix SVD helper for real and complex element
   types, rank-deficient and zero matrices, and dimensions smaller than the
   requested rank. Use QR/orthogonalization at each power iteration.
3. Thread `randomize`, `oversampling`, `niter`, and `rng` through both public
   overloads, `svd_std`, and the compatibility alias forwarding path.
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
  `oversampling`, and invalid `niter` produce clear errors.
- Performance check: compare exact and randomized paths on representative
  large, low-rank cases; do not promise speedups for small or full-rank blocks.

## Open design question

The first release defaults to proportional allocation with no adaptive
candidate growth. Users can select `allocation=:per_class` for a more
conservative candidate pool. Global top-`Nkeep` selection in proportional mode
is approximate: a class may have important singular values outside its
computed candidate pool.
