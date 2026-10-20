# Proposed `discard_eigen` keyword interface

Status: implemented. This document records the agreed interface and selection rules.

## Recommended interface

```julia
discard_eigen(result::EigenResult,
              kept_tag::AbstractString = "eigK",
              discarded_tag::AbstractString = "eigD";
              Nkeep::Union{Nothing, Integer} = nothing,
              keep_low::Bool = true,
              threshold::Union{Nothing, Real} = nothing,
              tol::Real = 0.1)
```

Return `(kept, discarded)`, both `EigenResult` objects. Selection is global
across symmetry sectors. The input is not mutated.

| Keyword | Meaning |
| --- | --- |
| `Nkeep=nothing` | Requested number of eigenvalue entries in count mode. An explicit threshold makes a count unnecessary. |
| `keep_low=true` | Keep the algebraically smallest eigenvalues by default; set `false` to keep the largest. |
| `threshold=nothing` | Explicit boundary in eigenvalue units, used in threshold mode. |
| `tol=0.1` | Existing relative count-window expansion for choosing a spectral gap; applies only in count mode. |

Keep the tag arguments positional, matching existing Telum usage. Move `Nkeep`
to a keyword in the primary interface and retain `tol` as a keyword.

`keep_low` expresses the two selection directions with a boolean, so users
do not need to remember symbolic choices. The default `true` preserves the
existing preference for low eigenvalues; `false` selects high eigenvalues.
Use `threshold` for the explicit boundary: `tol` already names the count-window
adjustment, and SVD's `cutoff` denotes a relative threshold with different
semantics.

## Eigenvalue ordering

Compare the eigenvalues by their signed numerical values. A negative value is
smaller than a positive value. For example, among `[-100, -2, 1, 4]`,
`keep_low=true` prefers `-100` and `keep_low=false` prefers `4`.
Use the same values for count selection, spectral gaps, and threshold comparisons.
There is no magnitude ranking, implicit energy shift, normalization, or relative
scaling.

The `hermitian` keyword no longer controls selection and is unnecessary in the
proposed interface. Remove it from `discard_eigen`; Hermitian and general
factor handling continues to follow the presence or absence of `result.V_inv`.
The `hermitian` keyword on `eigen` itself is unaffected.

This proposal covers real eigenvalues. Complex-spectrum behavior is deferred
until needed.

Each `eig_list` entry counts once, regardless of its irrep degeneracy. Selecting
an entry preserves its entire symmetry multiplet, as the current function does.
Separate entries with equal eigenvalues can still be split by count selection.

## Selection modes

Require exactly one of `Nkeep` and `threshold`. Supplying both or neither raises
`ArgumentError`. This keeps the boundary or count authoritative and avoids
silently ignoring a supplied count. Combining a threshold and count cap would
need a separate, explicitly defined contract; it is outside this proposal.

### Count mode: `Nkeep` supplied

- `keep_low=true` selects entries in ascending eigenvalue order.
- `keep_low=false` selects entries in descending eigenvalue order.
- `Nkeep` must be a nonnegative integer. Zero keeps nothing; a count at or above
  the number of available entries keeps everything.
- With `tol=0`, keep exactly `min(Nkeep, total)` entries.
- With `tol>0`, preserve the current count-window behavior: consider retaining
  from `Nkeep` through `min(total, ceil(Nkeep * (1 + tol)))` entries, and choose
  the cut with the largest adjacent eigenvalue gap. For descending selection, use
  `λ[i] - λ[i+1]` so gaps remain nonnegative. Resolve equal maximum gaps
  in favor of the smaller retained count.
- Preserve the existing special case: if the expanded window reaches the end
  of the spectrum, retain all entries. `Nkeep=0` keeps no entries, and
  `Nkeep >= total` already keeps all entries; neither requires window expansion.
  Eigenvalues equal to zero count as ordinary entries and participate in
  selection and gap detection. Thus `Nkeep` remains a target that `tol` can
  increase, not a hard cap.
- `tol` must be finite and nonnegative.

Use a stable selection order for equal eigenvalues; preserve their input order
instead of reversing an ascending list, which would also reverse ties.

### Threshold mode: `threshold` supplied

| `keep_low` | Kept entries |
| --- | --- |
| `true` | `λ <= threshold` |
| `false` | `λ >= threshold` |

The threshold must be a finite real number and may be negative, zero, or positive.

Use ordinary floating-point `<=` and `>=` directly, with no approximate
comparison or boundary tolerance. All entries whose eigenvalues equal the
threshold are kept, even across sectors. The retained
count follows entirely from the threshold and can be zero or the full spectrum.

`tol` has no effect in this mode, including its default value of `0.1`; do not
move an explicit threshold to a nearby spectral gap. Still validate `tol` as
finite and nonnegative so malformed options fail consistently in both modes.

`keep_low` must be a `Bool`. For an empty spectrum, any otherwise valid
selection returns two empty results.

## Examples

```julia
# Existing smallest-eigenvalue behavior, expressed with keyword Nkeep.
kept, discarded = discard_eigen(e, "K", "D"; Nkeep=50)

# Exactly the 20 algebraically largest eigenvalues, without gap adjustment.
kept, discarded = discard_eigen(e; Nkeep=20, keep_low=false, tol=0)

# Keep eigenvalues at or below an explicit energy boundary.
kept, discarded = discard_eigen(e, "low", "high"; threshold=-0.5)

# Keep eigenvalues at or above an explicit boundary, with no count argument.
kept, discarded = discard_eigen(e; keep_low=false, threshold=1e-8)
```

For a Hermitian spectrum `[-3, -1, 0, 2, 2, 5]`:

| Keywords | Kept eigenvalues, shown in ascending order |
| --- | --- |
| `Nkeep=2, tol=0` | `[-3, -1]` |
| `Nkeep=2, keep_low=false, tol=0` | `[2, 5]` |
| `threshold=2` | `[-3, -1, 0, 2, 2]` |
| `keep_low=false, threshold=2` | `[2, 2, 5]` |

## Output and storage invariants

Selection direction controls membership. Preserve ascending eigenvalue order in
both returned `eig_list` values, including when selecting the largest eigenvalues.
For general results with real eigenvalues, this deliberately replaces the old
magnitude ordering with signed ordering. Preserve stable
ties and renumber each entry's local index to match its selected tensor block.

Reuse the existing sector selection and retagging paths for `V`, `D`, and
`V_inv`. Preserve `V_inv === nothing` for Hermitian results. Keep defined RMT
types uniform, diagonal storage eligibility, and orthonormal w-matrix columns.
The kept and discarded entries must form a disjoint, exhaustive partition of
the original entries, with every factor using matching eigenvector indices.

## API migration and implementation scope

Remove the two existing positional-count signatures without compatibility
forwarding methods. This API break is accepted given the library's current user
base. Migrate repository callers directly to keyword `Nkeep` and keyword `tol`,
and remove any `hermitian` argument passed to `discard_eigen`.

For example:

```julia
# Previous call:
discard_eigen(e, 50, "K", "D"; tol=0.1)

# Proposed call:
discard_eigen(e, "K", "D"; Nkeep=50, tol=0.1)
```

Implement this through small changes to `discard_eigen`, `_split_eigen_result`,
and the existing keep-count helper. No new types or wrapper objects are needed.
Keep membership selection separate from factor slicing, and precompute selected
local indices before copying payloads. Independent per-sector copies should
remain possible without redesigning the selection logic.

Migrate the call in `benchmark/two_site_dmrg.jl` and examples in
`docs/src/dmrg-tutorial.md`, `docs/src/tensor-operations.md`,
`docs/example_line_by_line_tutorial.md`, and `TELUM_QUICK_REFERENCE.md`.

Add small regression cases to the existing `test/test_eigen.jl` for largest
selection, inclusive thresholds, count/threshold validation, gap adjustment in
both directions, stable ties, empty/full selections, signed ordering for
negative eigenvalues, and matching sector/eigenvector bookkeeping. Cover the
migration of existing positional calls. Run the relevant tests and inspect type
stability when implementing; a documentation build is unnecessary.
