# Refactor plan: propagate lazy `getsub` through contractions

## Goal

Make a lazy slice of a contraction slice its inputs before dense contraction
work is requested.  For example, for a lazy expression `A * B`,

```julia
getsub(A * B, leg, crit)
```

should identify which free output leg `leg` represents, apply the equivalent
selection to the corresponding leg of `A` or `B`, and construct a replacement
lazy contraction from those sliced inputs.  If an input is itself a lazy
contraction, repeat this process recursively until a concrete `TLArray` is
reached.

The immediate benefit is that a slice reduces source-sector and RMT-axis work
*before* the expensive contraction payload is evaluated.  In particular, a
sector removed by the slice must not cause an intermediate contraction RMT to
be allocated or filled.

The final goal of this refactor is to remove `SubTLArray` entirely.  It is a
temporary representation from the earlier lazy-`getsub` implementation, not a
permanent lazy tensor node.  After this work, lazy structural slicing should be
represented by recursively rebuilding the existing lazy expression with
concrete sliced leaves, rather than by wrapping an expression in a deferred
post-contraction slice.

This document supersedes the future direction of
`docs/completed/lazy_getsub_plan.md`; that document describes the existing
`SubTLArray` implementation and remains useful as historical context.

## Intended behavior

- At a concrete `TLArray` leaf, execute ordinary `getsub` immediately.
  Whole-sector selections (`Colon()`) retain references to the original RMT
  payloads; only genuine axis selection allocates a sliced RMT.
- At a `TLArrayContraction`, map selected free output legs to its two sources,
  recursively slice those sources, then make a replacement lazy contraction
  that recycles its CGT-side metadata.
- A nested contraction chain therefore receives selections at its concrete
  leaves before any lazy contraction RMT is requested.
- The replacement starts with no computed nonzero result sectors, even if the
  discarded parent contraction had some sectors materialized.  It has fresh
  RMT state and remapped work items, while retaining compatible CGT metadata.
- Public `getsub` semantics, including predicate validation, `preserve_space`,
  visible index state, and known-zero sector behavior, remain unchanged.

## Scope: recycle the CGT-side contraction recipe

`TLArrayContraction` stores several fields indexed by the sector slots of its
two inputs and of its output:

- `work_items` contains input-sector slots and an output-sector slot;
- `rmt_sizes`, `isdefined`, `iszero`, and `RMTs` are indexed by output slots;
- w-matrix storage and `factors` are also stored per output slot, but are
  CGT-side data rather than reduced-space data;

This refactor does **not** recompute CGT-side contraction work.  In particular,
it must recycle the parent contraction's output qlabels, w-matrices, QR work,
and contraction factors.  A `getsub` selection changes reduced-space axes and
may filter sector slots, but it does not change the qlabels or CGT fusion data
of a retained output sector.

The required transformation is therefore limited to sector IDs and RMT data:

- map old input-sector slots to the recursively sliced input slots;
- filter/remap output-sector slots;
- remap `work_items` to the new input and output slots;
- reuse the original qlabels, w-matrices, QR results, and factors for every
  retained output slot;
- update affected RMT-axis dimensions and reset lazy RMT state.

Propagation needs its source sector maps as a first-class internal result.  The
current public concrete `getsub` returns only a `TLArray`, and it may drop a
sector that becomes numerically zero after slicing.  The replacement
contraction cannot safely recover a complete old-to-new map by comparing output
qlabels: qlabels are not a unique sector identity.  Add a small internal
concrete helper which returns both the sliced `TLArray` and an
`old_source_to_new` vector (zero means dropped).  It is a function-level helper,
not a new tensor type.  The public `getsub` continues to return only the
tensor.

Calling `_lazy_contract` again would repeat valid CGT/QR/factor setup.  Do not
do that.  Use compact old-to-new index vectors local to the propagation helper
to build a replacement `TLArrayContraction` with recycled CGT-side metadata.

### Exact condition for recycling factors

The per-output `factors` are not independently reusable merely because the
output qlabels are unchanged.  They are made by QR over the complete ordered
list of source-sector pairs contributing to that output sector.  A retained
output sector may recycle its w-matrices, QR result, and factors **only when
its contributing source-pair sequence is identical to the parent's sequence,
apart from replacing each source slot with its old-to-new slot map**.

The normal qlabel-based propagation case has this property.  Every work item
for one output sector has the same output free qlabels.  A predicate which
retains that output qlabel consequently retains all of that output sector's
source pairs; a predicate which rejects it drops the entire output sector.

There is one important exception.  Concrete `getsub` may discover that an
actual selected RMT is numerically zero and drop that source sector.  This can
remove only some source pairs from an otherwise retained output sector.  The
old QR/factors are then invalid: retaining them while filtering the associated
work items produces the wrong contraction.

Use this policy for the exception:

1. The concrete propagation helper records zero for the dropped source slot;
   it does not store a zero RMT payload merely to retain its old pair
   structure.
2. Remove work items using that source sector.
3. If the removal changes a surviving output sector's source-pair sequence,
   recompute the CGT/QR/factors for that output sector only.  This is an
   exceptional fallback, not the normal propagation path.
4. If all work items for an output sector are removed, mark that result sector
   known-zero and leave its RMT slot undefined.

This fallback is necessary to retain the concrete-TLArray invariant that a
sector known to be zero before materialization has no stored RMT payload.

## Leg and view-state mapping

The request is expressed in visible result-leg coordinates.  A contraction may
also carry non-identity embedded `conj`, `scale`, and `perm` state.  The
implementation must therefore:

1. map every requested visible result leg through the contraction's stored
   permutation to a stored output leg;
2. map that stored free-output position to the source tensor:
   - positions `1:nfree1` map to `arr1` via `perm1`;
   - remaining free positions map to `arr2` via `perm2`;
3. recurse using the corresponding *visible* source legs, because the original
   contraction was constructed from the sources' public `inds` and `spaces`;
4. construct the replacement with recycled CGT metadata and remapped
   sector/RMT metadata;
5. transfer the old result `conj`, `scale`, and `perm` state directly.

The final step is essential: output permutation, conjugation, and scaling are
cheap expression state and must not be applied to concrete RMT storage merely
to propagate a structural operation.

## Implementation phases

### Phase 1: establish propagation helpers

Add small internal helpers in `src/tlarray/getsub.jl`; do not add a new struct,
trait system, or source-specializing type parameter.

- Normalize predicate picks once using the existing validation helpers.
- Add an internal recursive entry point that receives normalized selected legs
  and picks.
- Add the internal concrete endpoint described above, returning a sliced
  `TLArray` plus a complete old-to-new source-slot vector.  Do not infer that
  vector from qlabels after the fact.
- Keep the public `getsub` methods as thin dispatch/normalization wrappers.
- Make the concrete `TLArray` endpoint call the existing eager
  `_apply_getsub_picks` path.  Verify that its all-`Colon()` path aliases RMT
  payloads and does not copy them.

The orchestration layer may operate on `AbstractTLArray`; dense payload work
must remain behind existing concrete function barriers.

### Phase 2: propagate through `TLArrayContraction`

Implement the contraction specialization first.

1. Translate requested result legs from visible to stored coordinates.
2. Partition them into `arr1` and `arr2` leg lists using the free-leg portions
   of `perm1` and `perm2`.
3. Apply the existing predicate semantics recursively to each nonempty source
   group.  A group receives exactly the selections associated with its mapped
   output legs.
4. Build compact old-to-new sector-slot maps for both sliced sources, then one
   `old_output_to_new` map in old output-slot order.  This output map is the
   sole authority for every compacted output-indexed container.
5. Filter and remap `work_items`: retain an item only when its input and output
   sectors survive, replace its first three fields with their mapped slot IDs,
   and preserve its factor position unchanged on the normal path.  The
   exceptional factor-rebuild path also replaces factor positions with those
   of the newly prepared per-output factors.
6. Reuse the parent qlabels, w-matrices, QR results, and `factors` for retained
   output sectors in the same order only when the source-pair sequence is
   unchanged.  For every compacted output slot, reindex its qlabels,
   `wmatinfo`, factors, RMT size, and state through `old_output_to_new`.
   `wmatdata` may be shared with the parent because `wmatinfo` retains its
   offsets and the data is immutable.  Do not recompute CGT, QR, X-symbol, or
   factor data on the normal path.
7. Recalculate only affected `rmt_sizes` axes from the propagated source
   selections.  Copy all other physical axes and all non-Abelian/rank axes from
   the parent.  Create fresh RMT slots/state, retaining known-zero bits without
   allocating payloads.  Invoke the per-output exceptional fallback when a
   newly discovered zero source changes its contributing pair sequence.
8. Transfer the parent view state directly to the replacement.  Its lock is
   new because lazy RMT materialization state is independent.

### Phase 3: recursive expression coverage

Extend the recursive dispatcher to the remaining lazy structural nodes so no
`SubTLArray` needs to be created during normal propagation.

- `AddSingletonTLArray` and `DeleteSingletonTLArray`: translate legs through
  their existing source/result leg maps and recurse.  An inserted singleton
  leg has only its valid singleton coordinate; selecting it can be represented
  as whole-sector retention/removal without touching source RMT axes.
- Existing `SubTLArray` instances: provide a temporary compatibility path that
  composes their selections into the recursive request, then recurses into
  `arr`.  This permits mixed old/new expression trees while the type is being
  removed.

Composition for an old `SubTLArray` is coordinate-sensitive.  Its
`source_sectors` maps result slots to source slots, and its `saved_indices`
maps selected coordinates to the source RMT axes.  A subsequent vector pick
must be composed by indexing the saved vector; `Colon()` retains that saved
vector.  Sector filtering is composed through source qlabels, not by assuming
that result and source sector numbers agree.

### Phase 4: remove `SubTLArray`

Once every public lazy `getsub` path propagates to concrete leaves:

1. change lazy `getsub` dispatch so it always returns either a concrete
   `TLArray`, a rebuilt `TLArrayContraction`, or an existing singleton wrapper
   around one of those;
2. remove `_getsub_lazy`, `SubTLArray`, `GetSubSelector`, its accessor methods,
   materialization code, view-state rewrap method, and conversion support;
3. remove `SubTLArray` branches from contraction, summation, display, sparse
   conversion, and any other common-interface unions;
4. update tests to assert propagation behavior rather than deferred wrapper
   internals such as `source_sectors`;
5. update the AbstractTLArray subtype checklist/documentation to list only the
   remaining concrete and lazy tensor subtypes.

No compatibility constructor for `SubTLArray` is needed: it is internal and
not exported.  The removal should be a separate commit after propagation tests
pass, keeping the behavioral change easy to review.

## `preserve_space` policy

Keep the current rule: `preserve_space=true` permits only whole-sector
selection.  It can be propagated safely because it changes sector membership
without changing any selected-axis dimension.  An actual index slice continues
to throw before any source materialization or contraction rebuild occurs.

## Invariants

- Lazy wrappers preserve sector slot numbering only while they are wrappers.
  Propagation deliberately constructs a new expression, so it may have new
  input and output sector numbering.  Old sector IDs are never reused; parent
  CGT data for retained output sectors is compacted and reindexed through the
  explicit slot maps.
- A known-zero concrete sector remains undefined and zero; slicing must not
  allocate an RMT for it.
- Defined RMTs of every concrete result retain one concrete RMT type.  A real
  slice may densify a special storage format only according to the existing
  concrete `getsub` rules.
- CGT/TLArray leg ordering, w-matrices, QR results, and factors are fixed by
  the parent contraction.  Propagation reuses them and must not reconstruct
  CGR metadata or introduce a qlabel tie-breaker.
- A no-slice whole-sector leaf uses original RMT references.  Actual slicing
  allocates only the leaf RMT payload that needs slicing, never a discarded
  intermediate contraction payload.
- Recycled factor data is valid only when its contributing pair sequence is
  unchanged modulo sector-slot renumbering.  Changed pair sequences use the
  exceptional per-output CGT/QR/factor rebuild.

## Test and validation plan

Add focused cases to `test/test_lazy_getsub.jl` and `test/test_contraction.jl`:

- add the primary numerical regression in `test/test_contraction.jl`: construct
  the same operands twice, compare
  `getsub(contract(A, legsA, B, legsB), leg, crit)` against
  `TLArray(getsub(_lazy_contract(A, legsA, B, legsB), leg, crit))`, and compare
  their dense sparse-array representations with `≈`.  The criterion must make
  a genuine RMT-axis selection, not only `Colon()`, so the test protects the
  intended "slice then contract" result against the eager "contract then
  slice" reference;
- slicing a lazy contraction leaves the original contraction's RMT slots
  undefined and computes the replacement expression only on demand;
- whole-sector propagation aliases concrete leaf RMTs;
- an axis slice is numerically equal to eager `getsub(contract(A, B), ...)`;
- nested contractions propagate to concrete leaves without materializing their
  original intermediate contraction RMTs;
- a source slice that becomes numerically zero either removes the complete
  output sector or takes the exceptional per-output factor-rebuild path; it
  must never reuse factors after silently removing only some contributing
  pairs;
- selected legs map correctly through output permutations, scaling, and
  conjugation;
- multi-leg predicates map into both operands correctly;
- known-zero sectors remain undefined/zero;
- `preserve_space=true` has its existing success and failure behavior;
- chained selections, including noncontiguous and negative index selections,
  compose correctly during the temporary `SubTLArray` compatibility phase.

After removal, delete wrapper-specific assertions and retain only public
behavior and allocation/aliasing assertions.

Run the relevant test files, inspect the new mapping/rebuild helpers with
`@code_warntype`, and compare allocations for a representative slice-before-
contraction workload.  Also run the required DMRG command for the final
internal refactor validation:

```bash
julia --project=Telum Telum/test/dmrg_runtests.jl
```

The expected final energies are `-82.37493703992092` for `Nkeep=20` and
`-82.39703156828021` for `Nkeep=50`, up to numerical precision.

## Risks and controls

| Risk | Control |
| --- | --- |
| Source sector IDs change after leaf filtering. | Use explicit compact old-to-new source maps and one `old_output_to_new` map while rewriting every output-indexed field and `work_items`. |
| Output containers become misaligned after compaction. | Use `old_output_to_new` as the only mapping for qlabels, wmatinfo, factors, rmt_sizes, RMT state, and `work_items[:, 3]`; preserve a single retained-output order. |
| Flat w-matrix offsets become invalid. | Share immutable parent `wmatdata` and compact `wmatinfo` without changing its offsets; do not repack one without the other. |
| A free-axis slice changes reduced-space dimensions. | Update affected physical RMT dimensions only; copy parent non-Abelian/rank axes unchanged. |
| A newly sliced source sector is numerically zero. | Remove its work items; if a surviving output sector loses only some pairs, recompute CGT/QR/factors for that output sector instead of reusing invalid factors. |
| Visible result legs are permuted. | Map visible → stored output → source leg before recursion; reapply view state afterwards. |
| Chained old `SubTLArray` selectors use different coordinate systems. | Compose selector vectors explicitly and test vectors/ranges/negative indices. |
| A wrapper-only optimization becomes a payload copy. | Assert leaf RMT identity for all-`Colon()` selections. |
| Eager and old lazy getsub differ on known-zero slot retention. | Decide and document the new rule before removal; tests must not rely on the old `SubTLArray` slot layout when sector IDs are intentionally renumbered. |
| Parent RMTs are accidentally reused for an actual slice. | Start replacements with fresh undefined RMT state; alias a parent RMT only for all-Colon, identical-sector-layout cases. |
| Propagation setup outweighs gains for tiny fully consumed expressions. | Keep eager `getsub` unchanged; benchmark representative workloads before adopting broader rewrites. |

## Acceptance criteria

The refactor is complete when a lazy `getsub` reaches ordinary `TLArray`
leaves recursively, no discarded intermediate contraction RMT is materialized,
the parent CGT metadata (w-matrices, QR results, and factors) is recycled
rather than recomputed, whole-sector concrete leaves retain RMT references,
numerical results match the eager reference, and `SubTLArray` plus all of its
interface branches have been removed.
