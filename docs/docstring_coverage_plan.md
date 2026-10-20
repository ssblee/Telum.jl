# Telum and LurCGT docstring coverage plan

## Goal

Give both packages useful Julia help-mode documentation, starting with every
exported binding and its user-facing methods, then extending coverage to the
internal functions that users of the source tree must understand.  The work
must describe the current implementation accurately without changing tensor
semantics, storage layout, or performance-critical code.

The primary deliverable is source docstrings (`""" ... """`) that work with
Julia's `?name` help mode and can be collected by Documenter.  Documentation
pages are a secondary deliverable: use them for workflows and concepts, not as
a substitute for API docstrings.

This is a documentation-only effort.  It must not change source-code behavior,
public APIs, data layout, type parameters, evaluation strategy, or performance
paths; source edits are limited to adding or correcting docstrings and
documentation comments needed to describe existing behavior.

## Scope and priorities

### Tier 1 — public interface (required)

Document every exported binding in `Telum/src/Telum.jl` and
`LurCGT/src/LurCGT.jl`, including all distinct public method families.  This
includes:

- public types, their constructors, and important fields/invariants;
- exported functions, operators, macros, and mutating functions;
- public `Base`/`LinearAlgebra` extensions when they are part of normal use;
- keyword arguments, accepted input forms, return values, thrown errors, and
  mutation/materialization behavior where relevant.

An export alone is not enough to declare a name complete.  For example,
`contract`, `getsub`, `svd_leg`, and tag/index operations need one coherent
docstring covering their supported call forms; type constructors need their
own docstrings even when a surrounding type has one.

### Tier 2 — supported extension and advanced interface

Document non-exported functions/types that are intentionally used to extend or
inspect the library, especially abstract-array metadata accessors and lazy
evaluation operations in Telum, plus symmetry/CGT construction and persistence
operations in LurCGT.  Mark them as internal/advanced when they are not a
stability promise.

### Tier 3 — implementation coverage (best effort)

Add concise docstrings to remaining named functions, particularly functions
with non-obvious mathematical meaning, side effects, cache/SQLite behavior,
or assumptions about CGT/TLArray leg order.  Tiny single-purpose helpers do
not need prose when their names, types, and immediate context make them
self-evident; do document them when an invariant would otherwise be hidden.

Do not document generated methods, anonymous closures, or private mechanical
helpers merely to inflate a count.  Do not add docstrings to every overload
when a method-group docstring accurately covers the overload set.

## Documentation standard

Use Julia/Documenter-compatible docstrings immediately before the binding or
first canonical method.  Prefer this shape:

```julia
"""
    function_name(args...; keyword=default)

One-sentence summary.

Explain the tensor/symmetry meaning, supported argument forms, result, and any
important mutation, ownership, laziness, ordering, or error conditions.

# Examples

```julia
...
```
"""
```

Apply the following rules consistently:

- Start with a representative signature and an imperative or third-person
  summary; do not copy implementation details into the first sentence.
- State q-label and leg-direction conventions when an operation depends on
  them.  Preserve the CGT/TLArray stable ordering rule rather than implying a
  different tie-breaker.
- For `AbstractTLArray` operations, identify whether the result is eager or
  lazy, whether it preserves sector-slot numbering, and when `to_concrete` or
  `materialize` is required.
- For functions that can retain zero sectors or exploit `DiagRMT` storage,
  describe observable behavior but do not expose ephemeral buffer strategies.
- For mutating functions, use `!` in the documented signature and say which
  argument changes.  For I/O and SQLite functions, say which files/databases
  are read or modified and any cache/lock effects.
- Keep examples small, deterministic, and executable in the appropriate
  project environment.  Avoid examples that require downloading data or
  writing to a user's persistent cache.
- Reference related functions with `[`name`](@ref)` only after the relevant
  Documenter page/module setup makes the reference resolvable; otherwise use
  plain code formatting until that setup is added.

## Work breakdown

### 1. Establish an API inventory and coverage ledger

Create a checked-in or reviewable temporary ledger generated from source
exports, then manually classify every entry by package, source file, and
method family.  The ledger should record: exported name, owner file, public
call forms/constructors, docstring status, tier, reviewer, and any example
needed.

Use `Telum/src/Telum.jl` and `LurCGT/src/LurCGT.jl` as the authoritative public
lists.  Inspect included source files to distinguish real APIs from re-exported
LurCGT names.  Count documented *bindings/method families*, not raw occurrences
of `"""`, since one docstring may legitimately describe several overloads.

### 2. Document Telum's public surface

Work in source-file groups so that terms and examples remain consistent:

1. `TLArray.jl` and `tlarray/constructors.jl`: `AbstractTLArray`, `TLArray`,
   `TLIndex`, index/tag operations, tensor construction, metadata inspection,
   and conversion/materialization.
2. `contract.jl`, `permute.jl`, `sum_tlarray.jl`, `tlarray/algebra.jl`,
   `tlarray/getsub.jl`, and `tlarray/oplus.jl`: contraction, transformations,
   direct sum/subtensor operations, and their lazy/eager semantics.
3. `svd.jl`, `qr.jl`, and `eig.jl`: decomposition APIs, truncation/discard
   information, leg conventions, and return tuple structure.
4. `getLocalSpace.jl`, `getIdentity.jl`, `get1jtensor.jl`, and
   `localspaces/`: local-space option types, symmetry selection, operators,
   and physical meaning of returned tensors.
5. `DiagRMT.jl`, `sparse_array.jl`, `io_hdf5.jl`, `lazy.jl`, and `utils.jl`:
   diagonal storage, sparse conversion, HDF5 persistence, `@lazy`, and the
   remaining exported utility/index-matching APIs.

For every Telum name re-exported from LurCGT, give users a short Telum-side
cross-reference only if necessary; maintain the canonical API docstring in
LurCGT to prevent duplicated and diverging documentation.

### 3. Document LurCGT's public surface

Document in the same dependency order users encounter it:

1. `Symmetry/`, `irep.jl`, and `Base.jl`: symmetry types (`Z`, `U1`, `SU`,
   `SO`, `Sp`, `G2`), irreps, duals, dimensions, and representation
   decomposition.
2. `clebsch.jl`, `FTree.jl`, `omlist.jl`, `CGTperm.jl`, `CGTSVD.jl`,
   `CGTQR.jl`, and related symbols: CGT construction/manipulation,
   contraction, canonical bases, tree conventions, and factorization results.
3. `sqlite_io.jl`, `clebsch_io.jl`, and cache helpers: persistence, run modes,
   local/global data locations, merge/finalize operations, and destructive
   maintenance commands.  Clearly label administrative functions as advanced.
4. `decomp_space.jl`, `cgt_flip.jl`, `Conjperm.jl`, `Fsymbol.jl`, `Rsymbol.jl`,
   and `Xsymbol.jl`: advanced recoupling and basis-transformation APIs.

### 4. Fill high-value internals

After Tier 1 is complete, add Tier 2/3 docstrings by source subsystem.  Favor
functions at module boundaries, generic-function extension points, and any code
whose correctness relies on sector indexing, direction conventions, cache
ownership, or SQLite transaction/lock assumptions.  Keep hot inner loops and
mechanical conversion helpers minimally documented with nearby comments only
when a non-obvious invariant needs preservation.

### 5. Add navigable API pages

Expand `Telum/docs/src/` and `Telum/docs/make.jl` only after public docstrings
exist.  Add concise conceptual pages (tensor/index model, local spaces,
contraction/decompositions, lazy evaluation, and I/O) plus generated API pages
using `@autodocs`/`@docs` for Telum.  Add LurCGT API pages either by loading
LurCGT in the documentation environment or by documenting it in its own docs
project; choose the latter if dependency/version isolation makes a combined
build fragile.  The plan should not require inventing a new package interface.

### 6. Review, validate, and maintain coverage

For each batch:

- build documentation with `julia --project=Telum/docs Telum/docs/make.jl`;
- inspect warnings, unresolved references, duplicate bindings, and example
  failures;
- run the relevant package tests when a doctest or example exercises behavior;
- review source diffs separately with `git -C Telum diff` and
  `git -C LurCGT diff`.

Add a lightweight CI check or script only after the ledger format is proven
useful.  It should fail when a newly exported binding lacks a categorized
coverage entry, while allowing explicitly documented exceptions for aliases,
re-exports, and generated APIs.  Avoid a brittle check that treats every
private helper or overload as independently required.

## Completion criteria

- Every export in both module files has an accurate canonical docstring or a
  ledger-backed explanation of why it is a documented alias/re-export.
- Every user-facing constructor and distinct method family has a discoverable
  signature, behavior description, and necessary keyword/return information.
- Documentation correctly states Telum's lazy sector-slot and leg-order
  invariants, and LurCGT's symmetry/CGT conventions where exposed.
- The Telum Documenter build succeeds without new warnings caused by the work.
- Examples and doctests used by the docs succeed in clean project environments.
- No source behavior, type layout, allocation-sensitive path, or public API has
  changed; source changes are limited to docstrings and documentation comments.

## Suggested delivery sequence

Make small reviewable batches: first Telum core tensors and local spaces, then
Telum numerical/I/O APIs, then LurCGT symmetry/CGT APIs, then LurCGT SQLite and
advanced APIs, and finally the internal pass plus Documenter navigation.  Each
batch should update the coverage ledger and be reviewed for mathematical
terminology and example accuracy before the next batch starts.
