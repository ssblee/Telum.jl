# Opt-in lazy evaluation design

## Decision

Make eager evaluation the public default.  Add an explicit lexical opt-in,
initially `Telum.@lazy`, which rewrites a supported expression so that Telum
operations construct lazy results.  Values produced outside `@lazy` remain
ordinary `TLArray`s.

This is a good direction for Telum.  Tensor-network code often has a clear
evaluation boundary (an update, sweep step, or DMRG iteration), and retaining
every intermediate contraction until such a boundary can use substantially
more memory than eagerly releasing each intermediate result.  An opt-in block
makes that cost visible in the calling code and permits selective use where
sector-level demand actually avoids work.

The feature must not promise that a lazy block always uses less memory or is
faster.  It is useful when a later operation requests only part of a result,
or when a lazy structural operation can eliminate work before RMT payloads are
computed.  A chain which eventually consumes every sector normally has extra
metadata, source references, and dispatch/setup overhead; eager execution is
then the preferred default.

## User-facing API

```julia
using Telum

# Eager by default.
c = contract(a, (2,), b, (1,))     # ::TLArray

# Lazy only within this expression/block.
c = @lazy contract(a, (2,), b, (1,))  # ::TLArray

result = @lazy begin
    env = contract(left, (2,), mpo, (1,))
    env = contract(env, (2,), right, (1,))
    cropped = getsub(env, 1, keep_sector)
    cropped
end                                  # materializes and returns ::TLArray
```

`@lazy` is lexical, not a global switch: it does not alter task-local state,
method dispatch elsewhere, or values already constructed before the block.  A
single expression form and a `begin ... end` form should both be supported.
`Telum.@lazy` is the canonical exported spelling; do not use a generic global
flag such as `set_lazy!(true)`.  Lazy nodes are internal implementation values:
assignments inside the block may hold them, but the macro always materializes
the final `AbstractTLArray` result to `TLArray`.  The only other permitted final
result is a scalar (`Number`), returned unchanged.  A tuple, container,
user-defined object, or any other result type is rejected with an explanatory
error in the first release; this prevents a lazy tensor from escaping through a
compound return value.

### Lazy-value lifetime contract

Intermediate lazy tensors are valid only inside the dynamic extent of an
`@lazy` block.  They must not escape through a global, closure, task, mutable
container, or user-defined object.  This is what permits the implementation to
omit `reference_depth`: after final `TLArray(q)` conversion, normal block-local
lazy nodes and their source chains become unreachable and can be collected.

Escaping a lazy intermediate is unsupported and may raise an error when the
escaped value is later used.  Telum cannot reliably raise an error at the
moment of storage: generic Julia code such as `push!(v, x)`, `holder.field = x`,
or `() -> x` can retain `x` without a syntactically visible Telum operation.
The macro must therefore reject final unsupported containers where practical,
but documentation and tests must treat all other escaping storage as a caller
contract rather than a guaranteed immediate diagnostic.

For a supported operation `f`, implementation exposes a deliberately named
internal lazy entry point such as `_lazy_f`.  The macro changes only calls on
an allowlist, e.g.

```julia
@lazy contract(a, la, b, lb; reduce_lock=false)
# expands conceptually to
Telum._lazy_contract(a, la, b, lb; reduce_lock=false)
```

The expansion must preserve positional arguments, keyword arguments, source
locations, and normal Julia evaluation order.  It must use `GlobalRef(Telum,
...)`/proper macro hygiene so a local variable named `contract` cannot silently
change the target.  Calls not on the allowlist are recursively left intact;
their arguments may nevertheless contain rewritten Telum calls.

Do **not** rewrite arbitrary named function calls based on the runtime type of
their arguments.  A Julia macro sees syntax, not values or later dispatch.
Such a scheme would be fragile around aliases, imports, qualified names,
closures, and user-defined overloads.  Start with direct unqualified Telum
names and `Telum.name`; add another syntactic spelling only with explicit
tests.  `*` is the intentional exception: rewrite it to `_lazy_contract(a, b)`.
Its generic two-argument fallback is ordinary `a * b`, while its
`_lazy_contract(::AbstractTLArray, ::AbstractTLArray)` specialization discovers
matching contraction legs and constructs a lazy contraction.  Thus scalar
multiplication and unrelated products inside a lazy block retain their normal
behavior.

## Eager/lazy operation contract

Every computation adopted by the feature has one public behavior and, where
needed, one internal lazy counterpart:

| Mode | Result | Payload behavior |
| --- | --- | --- |
| Default eager call | `TLArray` (or the existing concrete result type) | Computes all active output sectors before return. |
| Internal lazy entry point | operation-specific `AbstractTLArray` subtype, when justified | Allocates immutable result metadata and defers RMT payload work until `compute_sectors` or a consuming operation requests it. |
| `@lazy` expression/block result | `TLArray` | Materializes the final tensor value after all intermediate lazy operations have been evaluated. |

The two paths must share result-metadata construction and numerical payload
kernels.  The intended shape is:

```julia
function f(args...; kwargs...)
    lazy = _lazy_f(args...; kwargs...)
    compute_sectors(lazy, sector_slots(lazy))
    return TLArray(lazy)
end
```

The exact conversion helper may differ for multi-result operations such as SVD
or QR, but eager code must be a thin materializing wrapper, never a second
independent implementation.  This prevents metadata ordering, zero-sector,
and numerical behavior from drifting between modes.  `contract` already
largely follows this model (`_lazy_contract` plus its former `lazy` keyword); split it
into an eager `contract` and a separate internal `_lazy_contract` entry point.
Do not retain a runtime `lazy::Bool` switch because its two return types weaken
type stability.

The established `AbstractTLArray` invariants remain mandatory:

* Metadata and sector-slot numbering are fixed when the lazy result is built.
  Lazy wrappers must neither sort, compact, filter, nor renumber slots.
* A known-zero output sector has `iszero[sector] == true`, stays undefined, and
  has no stored zero RMT payload.  A concrete eager result has every active
  sector defined before it is returned.
* Each defined RMT in an array has one concrete RMT type.  Diagonal storage is
  used only under the existing two-identical-space-plus-singletons rule.
* Embedded `conj`, `scale`, and `perm` state stays in metadata.  Structural
  lazy operations act on stored-leg indices and must not force RMT canonicality.
* A lazy source retains its inputs until requested sectors have been computed.
  `@lazy` is deliberately restricted to small lexical blocks, so lazy nodes do
  not carry or maintain `reference_depth`; long-lived lazy graphs are out of
  scope for this feature.

Any new lazy array subtype must implement the complete common metadata,
w-matrix, RMT shape/payload, sector-computation, and materialization interface.
Before adding one,
document why an existing lazy representation or a function-level patch is
insufficient, the invariant it represents, its layout/dispatch benefit, and
the type-stability check.

## Scope and rollout

Do not attempt to make “every computation” lazy in the first change.  Classify
operations first:

| Class | Initial treatment | Reason |
| --- | --- | --- |
| Contraction | Make default eager; rewrite to existing lazy constructor | It already has per-sector work items and demand-driven materialization. |
| Metadata-only transforms (`conj`, scale, `permutedims`, index edits) | Already cheap/lazy view state; no new computation node | Their work is metadata-only and should preserve storage. |
| Lazy structural transforms (`getsub`, singleton add/delete) | Keep existing lazy behavior when their source is lazy | They can pass sector demand through to the source. |
| Sum, `oplus`, tensor product | Keep eager initially; design only after measuring demand propagation | Their output-sector matching and buffer lifetimes need dedicated lazy planning. |
| SVD, QR, eigen | Keep eager initially | They return multiple coupled concrete outputs and generally need all input sectors; a wrapper would retain large sources with little opportunity to save work. |
| I/O, sparse conversion, norms, scalar indexing | Materialization boundaries | Their specified result requires concrete payloads. |

This staged scope is important: creating a distinct lazy struct for every
operation merely to satisfy the macro would increase retained memory and
compile time, recreating the problem the change intends to solve.  Add a node
only after its consumer can request a strict subset of output sectors or it
can fuse/avoid a materialization boundary.

The initial lazy-operation graph is closed under contraction, `getsub`, and
singleton add/delete: these operations must compose in every valid order inside
an `@lazy` block.  The existing code does not yet establish every edge—for
example, `getsub(addSingleton(lazy_q), ...)` needs compatible lazy dispatch—so
the implementation must extend the existing lazy routes where required rather
than introduce an eager boundary.  Every composed result must preserve source
sector-slot numbering, embedded view state, and the established zero-sector
invariants.

## Macro rewrite rules

Implement a small recursive AST transformer in `src/lazy.jl`, included after
the callable implementations are available and exported from `Telum.jl`.
Keep it ordinary macro code, with no generated functions or runtime AST
interpretation in payload hot paths.

1. Validate that the macro receives one expression.
2. Recurse through `:block`, `:let`, `:if`, `:for`, `:while`, `:try`, lambda,
   assignment RHS, tuple, and call argument expressions while preserving scope.
   Do not rewrite the binding side of assignments, function names in a local
   function definition, quoted code, `macrocall` arguments, module imports, or
   declarations.
3. For a direct supported `:call`, first recursively transform arguments and
   keywords, then replace only its callee with the appropriate lazy entry point.
4. Preserve nested `@lazy` blocks (rewriting twice must be idempotent) and
   allow `@eager expr` later as an explicit escape hatch if evidence shows it
   is needed.  Do not implement `@eager` in the first patch without a concrete
   use case.
5. Emit a useful `ArgumentError` at macro expansion for syntactically invalid
   invocations.  Do not reject unsupported valid Julia syntax merely because
   it is not rewritten.

Initially the allowlist contains `contract` and infix `*`.  Factor the existing
matched-leg discovery out of `Base.:*`; the two-argument eager and lazy
contraction overloads call that helper, then dispatch to their explicit-leg
counterparts.  This keeps the ergonomic multiplication syntax and avoids
making users spell explicit leg lists solely to get lazy execution.

The initial contraction interface is:

```julia
# Public, eager-only API; both return TLArray.
contract(q1, q2; reduce_lock=true)
contract(q1, legs1, q2, legs2; reduce_lock=true, verify_legs=true)

# Internal API used by @lazy; both return TLArrayContraction.
_lazy_contract(q1, q2; reduce_lock=true)
_lazy_contract(q1, legs1, q2, legs2; reduce_lock=true, verify_legs=true)
```

`contract(q1, q2)` uses the exact matching rule currently used by `q1 * q2`:
nonempty, unlocked, direction-compatible indices with matching spaces must
have a unique match.  It throws for no match or ambiguity.  `q1 * q2` is an
eager alias of this two-argument overload.  Within `@lazy`, both
`contract(q1, q2)` and `q1 * q2` rewrite to `_lazy_contract(q1, q2)`; the
explicit-leg call form rewrites to its explicit-leg internal counterpart.

The generic fallback `_lazy_contract(a, b) = a * b` exists only because the
macro cannot know operand types.  It is not an exported API and must not carry
keywords: a rewritten scalar or unrelated product keeps Base's normal
two-argument multiplication semantics.

Remove the public `lazy` keyword from `contract` in the same breaking change.
The eager and lazy entry points have one concrete result type each, rather than
one keyword-selected method with an unstable union of `TLArray` and
`TLArrayContraction` results.

## Concrete conversion boundary

`@lazy` finalizes an internal lazy tensor with `TLArray(q)`.  This materializes
all active sectors and returns the required public `TLArray` type; it may alias
the lazy result's now-materialized RMT storage and preserve embedded `conj`,
`scale`, and `perm` state.  That is correct: these are valid `TLArray` view
states, and the lazy wrapper itself is not returned.

Keep public `to_concrete`, but restrict it to `to_concrete(::TLArray)`.  Its
meaning is an independent canonical copy of an already concrete user tensor.
Internal callers which currently accept `AbstractTLArray`—HDF5 output, sparse
conversion, display, `oplus`, and internal `copy` paths—must use the private
canonicalization helper instead.  That helper first calls `TLArray(q)` for a
lazy input, then makes a canonical copy only when required.  Do not expose lazy
node types or a public conversion method for them.  Remove `TLArrayContraction` from the
export list; it remains technically reachable as `Telum.TLArrayContraction`
under Julia's module rules, but is not part of Telum's supported public API.

## Implementation plan

1. **Baseline and memory characterization.** Record representative DMRG and
   small sector-selective examples under the current behavior: peak live bytes,
   allocations, runtime, number of materialized sectors, and maximum
   live lazy-node count.  Use these as acceptance measurements, not just timing.
2. **Split contraction entry points.** Make `contract` an eager-only thin
   materializing wrapper and retain `_lazy_contract` as its internal lazy
   counterpart.  Remove the `lazy` keyword and prove each entry point has one
   concrete return type.
3. **Add two-argument contraction and share leg matching.** Extract the
   matching-leg discovery from `Base.:*`; use it from eager/lazy two-argument
   `contract` and retain `*` as the eager ergonomic alias.
4. **Implement `@lazy`.** Put the hygienic recursive transformer and final
   `TLArray(q)` finalization boundary in a small new source file.  Export only the
   macro, and document exactly that `contract` and `*` are rewritten.
5. **Use the existing lazy structural operations.** `getsub` and
   `addSingleton`/`deleteSingleton` already select their lazy behavior when
   their input is an internal lazy tensor.  Do not rewrite these calls; extend
   their dispatch as necessary so all valid compositions of contraction,
   `getsub`, add-singleton, and delete-singleton remain lazy inside the block
   and materialize only at its end.
6. **Remove reference-depth plumbing.** Delete the fields, constructors,
   accessors, and tests for `reference_depth` from the existing internal lazy
   nodes.  Update the project subtype checklist accordingly.
7. **Narrow `to_concrete`.** Add the private materialization helper, restrict
   public `to_concrete` to `TLArray`, and move all internal lazy callers to the
   helper.
8. **Establish materialization boundaries.** Audit all exported consumers.
   Operations which cannot consume a lazy input sectorwise must call
   `TLArray(q)` deliberately and document that boundary.  `materialize(q)`
   alone is not a public-result boundary because it returns the original lazy
   wrapper.  Use the private canonicalization helper only where independent,
   identity-view storage is actually required.
9. **Add lazy operations selectively.** For each candidate, first write a
   minimal demand-propagation design: result metadata, source-sector map,
   source retention lifetime, per-sector computation method, zero-sector
   behavior, and expected memory win.  Obtain approval before adding a new
   lazy subtype.  Implement the lazy node, then make eager behavior its thin
   materializing wrapper.
10. **Optimize only measured paths.** If the work-item fill is heavy, use
   independent result-sector writes, precomputed sector converters/ranges, and
   thread-safe per-sector buffers.  Do not introduce shared loop counters or
   eager global caches just to make laziness work.
11. **Release and migration.** Announce the semantic default change prominently
   in release notes: code which expected a lazy default must use `@lazy`.
   Removing `contract(...; lazy=...)` is intentional because a keyword-selected
   return type defeats the type-stability goal.

## Testing and validation

Behavior-changing work needs tests in the existing relevant files, rather than
a new broad lazy test suite.

* `test_contraction.jl`: eager `contract` and eager `*` result/type; `@lazy`
  expression and block forms return `TLArray`; positional and keyword
  preservation; equality of dense sector payloads; known-zero slot behavior.
* `test_lazy_getsub.jl`: a macro-created lazy contraction followed by `getsub`
  must request only mapped source sectors and preserve slot indices.
* `test_tlarray_structure.jl`: test all valid pairwise orderings and at least
  one three-operation chain of `getsub`, `addSingleton`, and `deleteSingleton`
  on a macro-created lazy contraction.  Verify their final `TLArray` values
  against the corresponding eager sequence, including known-zero sector slots
  and non-identity embedded `conj`/`scale`/`perm` state.
* Existing factorization/eigen tests: eagerly consuming a macro-created lazy
  input yields current concrete result types and values.
* Macro unit tests: `*` and two-argument `contract` rewrite to lazy contraction
  for tensor operands but preserve scalar products; qualified/unqualified
  `contract`; final result materialization; local binding hygiene; and quoted
  syntax untouched.
* Run the relevant Telum test selection and the required DMRG runner:
  `julia --project=Telum Telum/test/dmrg_runtests.jl`.  Its final Nkeep 20 and
  50 energies must remain `-82.37493703992092` and `-82.39703156828021` to
  numerical precision.
* Inspect representative methods with `@code_warntype` (macro-expanded lazy
  contraction construction and `compute_sectors`) and report allocation/peak
  memory changes against the baseline.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Lazy DAG retains sources and their metadata | Peak memory exceeds eager execution | Default eager; restrict lazy evaluation to small lexical blocks; add only demand-propagating nodes; track peak live bytes. |
| Macro rewrites a call users did not intend | Semantic surprise or hygiene bug | Strict syntactic allowlist, `GlobalRef` targets, documented supported forms, and macro expansion tests. |
| Separate eager/lazy numerical implementations drift | Different sector ordering or values | One lazy metadata builder and one payload kernel; eager path materializes it. |
| Eager consumer forces full computation late | Poor locality / unexpected peak | Audit explicit materialization boundaries and benchmark complete-pipeline memory, not only construction. |
| Zero slots are compacted or reordered | Incorrect source-sector mapping | Preserve slot numbering in every lazy wrapper; assert metadata mappings in tests. |
| New wrapper types cause recompilation or unstable dispatch | Slower DMRG / excessive compilation | No source-concrete-type parameters in lazy fields; concrete metadata containers; function barriers around payload work; `@code_warntype`. |
| Lazy mutation or aliases observe changing input | Incorrect deferred result | Treat lazy inputs as immutable by contract, or materialize/copy at explicit mutation boundaries; document the rule before exposing mutable APIs. |
| Exceptions appear later than the source expression | Harder debugging | Validate structural compatibility eagerly at node construction; retain source locations in error wrapping only if measurement justifies the metadata. |
| Multithreaded sector requests race | Corrupt RMT slots | Keep locking/once-only sector computation; parallelize only independent result-sector writes with isolated buffers. |

### Important risk details

**Removing `reference_depth` does not itself bound memory.** A lexical macro
block limits the normal source spelling, but it cannot prevent an intermediate
lazy tensor from escaping through a closure, global variable, task, exception,
or a mutable container.  Such an escaped node retains its entire source chain.
Making the types unexported prevents this in supported user code but cannot
make it impossible in Julia.  Removing the field is therefore an accepted
trade-off only if Telum treats escaped lazy nodes as unsupported and tests the
normal non-escaping block case for prompt garbage collection.  If realistic
internal workflows retain lazy nodes beyond a block, keep a depth limit or add
an explicit internal materialization checkpoint instead.  This is an advanced
feature: users are responsible for keeping intermediate lazy values within the
block, just as they are responsible for other lifetime-sensitive optimizations.

**The final materialization boundary can erase the intended win.** In
`@lazy begin getsub(a * b, ...) end`, slicing can reduce the materialized work.
In `@lazy begin norm(a * b) end`, `norm` may request every sector before the
block ends, so the lazy node adds metadata and source retention without saving
payload work.  Benchmarks must compare complete user workflows and peak live
memory, not only the allocation count of expression construction.  A block
with only a small number of operations is often not worth making lazy; Telum
will document this as a performance trade-off rather than attempt automatic
selection.  A later `norm(::AbstractTLArray)` lazy consumer can instead request
one sector at a time and accumulate squared norms sequentially, avoiding full
result materialization without requiring a lazy norm result node.

**Macro rewriting has semantic rather than numerical failure modes.** It must
not evaluate operands twice, reorder keyword/default evaluation, rewrite a
quoted expression, or replace a locally bound function named `contract`.
`*` is particularly sensitive because it appears in scalar arithmetic,
broadcasted closures, and user overloads.  The generic `_lazy_contract(a, b) = a*b`
fallback preserves ordinary products, but must be tested for scalar–tensor
scaling and mixed non-Telum operands as well as tensor–tensor contraction.

**Canonicalization is separate from finalization.** Lazy nodes may carry
non-identity `conj`, `scale`, or `perm`, and returning `TLArray(q)` may retain
that state.  This is valid for the final result of `@lazy`; it satisfies the
public type and preserves tensor semantics without an unnecessary copy.  HDF5
serialization, sparse expansion, and the public independent-copy API instead
need canonical payloads.  The private helper must therefore have two explicitly
tested modes: `TLArray(q)` finalization, and independent canonical copying only
where needed.  Conflating them risks wrong leg order, missed conjugation, or
double application of scale.

**A public `to_concrete(::TLArray)` is an API break.** Existing downstream code
may use it on `TLArrayContraction` because that type is currently exported and
the method accepts `AbstractTLArray`.  Removing that method changes a runtime
conversion into a `MethodError`.  The release note must identify this break;
after the patch `TLArrayContraction` is an internal lazy-block value and
`to_concrete` is never called on it.  Internal consumers first finalize with
`TLArray(q)` and call `to_concrete` only if they require an independent
canonical copy.  If compatibility is required, retain the old method for one
deprecation cycle but mark it internal.

**Type stability depends on the whole boundary, not only `contract`.** Removing
the `lazy::Bool` keyword makes direct contraction type-stable, but a macro
finalizer may return either `TLArray` or `Number`, depending on the block's
last expression.  This is normal Julia control-flow behavior; the precise
promise is that a tensor crossing the boundary is always a `TLArray`, while a
scalar is returned unchanged.  Other final result types are rejected.  Verify
the eager and internal lazy entry points separately with `@code_warntype`, and
do not claim a universally concrete type for every possible `@lazy` block.

## Non-goals for the first release

* No global lazy-mode flag, task-local switch, or runtime tracing system.
* No automatic fusion engine or general-purpose expression graph.
* No promise that every Telum or Base operation is rewritten by `@lazy`.
* No new lazy wrappers for SVD, QR, eigen, I/O, or scalar reductions before a
  measured demand-propagation design proves a benefit.

This keeps the initial feature small: it reverses the current surprising
default, exposes laziness where it is already implemented, and creates a safe
path to extend it only when it saves real memory.
