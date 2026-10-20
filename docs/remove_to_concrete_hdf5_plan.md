# Remove `to_concrete` and preserve TLArray view state in HDF5

## Goal

Remove `to_concrete` and `_canonical_tlarray`.  Printing and HDF5 storage of a
concrete `TLArray` must operate on its existing storage and embedded view
state, rather than construct an independent tensor with trivial `conj`,
`scale`, and `perm` fields.  HDF5 must persist those three fields together
with the existing tensor metadata.

`_canonical_tlarray` must be deleted rather than retained as an internal
fallback: it copies metadata and every active RMT while physically applying
view state, so callers such as sum alignment can incur a full-tensor memory
peak merely to consume or align a tensor.  Each such caller must instead use
the source state directly or allocate only the exact result storage it owns.

The refactor must retain the distinction between raw stored RMT data and its
logical value: `conj`, `scale`, and `perm` remain deferred state and are not
applied to all payloads merely for display or serialization.

## Current state

- `to_concrete(::TLArray)` and `_canonical_tlarray` currently provide a
  generic canonicalizing deep-copy boundary.  They materialize lazy inputs,
  apply pending permutation, scale, and conjugation to RMTs, copy metadata,
  and return a `TLArray` with identity view state.
- `save_tlarray(::TLArray)` currently calls `to_concrete` whenever the state
  is non-identity.  The on-disk schema therefore represents only canonical
  storage.
- `show(::SubTLArray)` and `printmeta(::AbstractTLArray)` also canonicalize
  before printing.  Direct `TLArray` display does not allocate a replacement,
  but it must use logical accessors consistently when a concrete tensor has
  embedded state.
- The HDF5 schema is currently `1.0` and stores symmetries, raw indices and
  spaces, sector metadata, w-matrices, sector flags, and RMT arenas, but not
  `conj`, `scale`, or `perm`.

## Implementation plan

### 1. Remove the public conversion API

1. Remove `to_concrete` from the export list in `src/Telum.jl`, then delete
   its method and docstring from `src/TLArray.jl`.
2. Delete `_canonical_tlarray` and all callers.  Do not replace it with another
   general materialization/canonicalization helper.
3. Define `copy(q::TLArray)` only, as a field-preserving deep copy. It must
   retain the source object's `conj`, `scale`, and `perm` state rather than
   turn that state into RMT mutations or identity fields. Do not define
   `copy(::AbstractTLArray)` or copy methods for lazy subtypes: those values
   are evaluation recipes, and copying them has no required ownership or
   materialization meaning.
4. At every former `_canonical_tlarray` call site, make the ownership and
   storage decision local to the operation.  If an operation requires a new
   owned RMT layout, it must explicitly allocate that layout, fill it using the
   required logical RMT transforms, and construct the result with the intended
   state.  Operations that only consume data should use the sector interface
   directly.  This rule applies to every former caller; no caller is allowed
   to retain a convenience canonical-copy path.
5. In particular, rewrite `random_similar`, sparse conversion, sum alignment,
   `oplus`, singleton conversion, and any lazy-finalization paths to state
   their output-storage requirements explicitly:

   - `random_similar` is defined only for concrete `TLArray`. It starts with
     `copy(q)` and replaces each active RMT with a random payload of the same
     raw RMT type and shape, retaining the copied embedded view state. Lazy
     evaluation objects intentionally have no `random_similar` method.
   - Sparse conversion must first expand the raw stored tensor into a sparse
     array, then apply the deferred conjugation, leg permutation, and scale to
     that sparse result.  These sparse-array transforms are local and avoid a
     full intermediate `TLArray` copy.
   - Sum alignment must use the common `AbstractTLArray` metadata, w-matrix,
     and RMT interfaces directly.  It must retain or compose the input
     permutation as part of the aligned lazy input/result processing; it must
     never canonicalize an aligned summand merely to give the sum code a
     uniform representation.
   - For every remaining former caller, first establish its precise ownership,
     view-state, and aliasing contract with a transformed-input regression
     test, then implement only the storage it genuinely owns.

   No path may use a hidden generic canonical-copy escape hatch.
6. Remove or replace all tests that directly invoke `to_concrete`. Tests for
   independent concrete data should use `copy(::TLArray)`; tests involving
   lazy tensors should compare the operation's logical result or test that
   operation's explicit storage contract, rather than copying the wrapper.
7. Update `docs/src/advanced.md`, `Telum_docs/5_advanced_tips.ipynb`, lazy
   documentation, and affected docstrings.  User-facing guidance should say
   that `copy(q)` creates independently owned storage; it must not promise a
   public canonicalization operation.

### 2. Print through the logical TLArray interface

1. Keep `printmeta` defined only for concrete `TLArray`. It prints the compact
   header from existing metadata only and must neither materialize sectors nor
   canonicalize a lazy evaluation object. Remove the broad
   `printmeta(::AbstractTLArray)` overload rather than making it an implicit
   lazy-evaluation boundary.
2. Generalize full text display where needed so it consumes the common logical
   metadata and sector accessors directly. Use `inds(q)`,
   `sector_qlabel(q, sector, leg)`, `sector_rmt(q, sector)`,
   and RMT-shape accessors instead of raw fields where the visible state may
   differ from stored state.
3. When display requests RMT data, obtain the logical RMT value: apply the
   embedded physical-leg permutation, conjugation, and scale before returning
   or formatting that data.  Reuse stored data only when the state makes this
   correct; a temporary permuted, conjugated, and/or scaled payload is allowed
   when the state is nontrivial.  Do not mutate stored RMTs and do not construct
   a replacement `TLArray`.
4. Preserve sector-slot numbering, including known-zero slots, while deriving
   the displayed active-sector list.
5. Remove canonical-copy calls from display-only paths.  Lazy payload
   evaluation should occur only if a displayed sector needs its RMT, following
   the normal lazy-sector contract.

### 3. Define HDF5 schema v2 with stored view state

1. Bump `_TLARRAY_HDF5_SCHEMA_VERSION` from `"1.0"` to `"2.0"`.
2. Write the raw `TLArray` state as part of group metadata:

   - `conj::Bool`;
   - `scale::T`, preserving the logical tensor element type; and
   - `perm::NTuple{QD,Int}`, encoded as a rank-length integer vector.

   V2 must also distinguish the logical tensor element type `T` from the raw
   RMT element type.  Keep the existing `eltype` tag for the logical tensor
   type and add an `rmt_eltype` tag for the raw RMT arena type.  This permits,
   for example, `ComplexF64` deferred scale on `Float32` RMT payloads without
   converting all payloads during save.  V1 is the equal-type identity-state
   subset: its existing `eltype` tag means both types, `scale = one(T)`,
   `conj = false`, and `perm = identity`.

   The current arena writer is a v1 assumption: it is parameterized by logical
   `T` and accumulates payloads in `T[]`. That would convert every raw RMT when
   a deferred scale promotes the logical tensor type, defeating raw-storage
   preservation and adding a full-arena conversion allocation. V2 must obtain
   the raw element type from the concrete RMT storage, allocate/read the arena
   with that type, and pass the independently decoded logical `T` and `scale`
   to the state-preserving `TLArray` constructor.

3. Keep serializing raw stored indices, spaces, q-labels, w-matrices, sector
   flags, and RMT payloads.  Do not reorder legs, transform RMTs, or compact
   sector slots in the serializer.  In particular, the writer must call
   `stored_inds(q)` and `stored_spaces(q)`, not the logical `q.inds`/`inds(q)`
   and `q.spaces`/`spaces(q)` accessors.

   The v2 file format is deliberately a stored-representation format.  For a
   tensor produced by `permutedims`, it saves the stored pre-permutation leg
   metadata and RMT axes together with the composed `perm` field.  It does not
   construct the visible reordered metadata first.  “Stored” means the current
   result object's `stored_*` fields, not necessarily those of its earliest
   source tensor: chained views and structural operations may have their own
   current stored representation.  On load, reconstruct those same fields and
   state; the ordinary logical accessors then derive the visible leg order.
4. Extend validation to check state type/shape, validate `scale` against the
   logical tensor type and the raw RMT arena against `rmt_eltype`, and validate
   that `perm` is a permutation of `1:QD`.
5. Rename `_validate_concrete_tlarray_for_hdf5` if useful to make clear that a
   concrete `TLArray` is valid with either identity or non-identity embedded
   state.

### 4. Load raw storage with its state intact

1. Add a v2 state reader alongside the other group metadata readers.
2. Construct the loaded tensor through the existing internal
   `TLArray(Val(:alias_storage_view_state), ...)` path, supplying the decoded
   `conj`, `scale`, and `perm` fields.  Do not use the default constructor,
   which initializes identity state.
3. Retain v1 read compatibility: v1 is the equal-type trivial-state subset of
   v2 because its writer canonicalized transformed tensors before serializing
   them.  Dispatch by `schema_version`; for v1 read the existing `eltype` as
   both the tensor and raw-RMT type, then supply `(false, one(T), identity
   permutation)`.  For v2, decode `eltype` and `rmt_eltype` independently.
   New files are written only as v2.
4. Keep strict rejection for unknown schema versions and malformed state
   metadata.

### 5. Simplify save boundaries

1. Make `save_tlarray(h5, name, q::TLArray)` call the writer directly.  It
   must not branch on identity state and must not allocate a replacement
   `TLArray`.
2. Define the path and handle overloads only for explicit `TLArray` values.
   Lazy `AbstractTLArray` subtypes are internal evaluation objects and are not
   serializable through this API; do not materialize or canonicalize them as an
   implicit save side effect.

## Test plan

Add focused coverage in `test/test_hdf5_io.jl` and adjust affected structure,
contraction, and lazy-getsub tests.

1. Round-trip a concrete tensor with nontrivial `perm`, complex `scale`, and
   `conj`.  Assert that raw state, stored metadata, sector flags, raw RMT
   element type, logical tensor element type, and logical numerical values are
   preserved.  Include real `Float32` RMT payloads with a complex scale so the
   two element types differ.
2. Verify direct save leaves the source object and its state unchanged.  This
   guards against accidental canonicalization in the writer.
3. Add a minimal v1 compatibility fixture/helper which omits view state and
   confirms that load assigns identity state.
4. Check text display for transformed concrete and lazy tensors: displayed
   index directions/order, q-label order, and scalar values must reflect the
   logical tensor without creating a canonical tensor.  Include a tensor with
   multiple sectors and nontrivial RMTs, and verify that only the small RMT
   payloads selected by the formatter are evaluated/transformed.
   Separately, check that `printmeta(::TLArray)` prints only its header and
   requires no RMT access; lazy evaluation objects must have no `printmeta`
   overload.
5. Replace `to_concrete` tests with `copy(::TLArray)` ownership tests where
   applicable, assert that `to_concrete` is neither exported nor defined, and
   assert that no broad `copy(::AbstractTLArray)` method is introduced.
6. Add focused transformed-input tests for each former canonicalization caller:
   sparse conversion must agree with a dense logical reference; sum with
   permuted matching legs must cancel/merge without materializing its input;
   `oplus`, singleton conversion, and lazy finalization must preserve logical
   values, view state where applicable, sector slots, and their documented
   storage ownership.
7. Assert that `random_similar` and `save_tlarray` accept concrete `TLArray`
   values only. For `random_similar`, verify independent RMT payload ownership
   and preservation of embedded `conj`, `scale`, and `perm` state.

## Validation

Run the focused HDF5, TLArray-structure, lazy-getsub, and contraction tests,
then run:

```bash
julia --project=Telum Telum/test/runtests.jl
```

Inspect `@code_warntype` for the new HDF5 state read/write helpers.  Compare
allocations for saving a transformed concrete `TLArray` before and after the
change; the new direct path should avoid the full canonical tensor and RMT
copies.  Report each operation that deliberately allocates owned storage and
its explicit reason.
