# Tensors without symmetry: design and status

This note records the dense-tensor constructor and specialization decisions.
The constructor is implemented in `src/tlarray/constructors.jl`; the selected
all-Abelian `legflip` optimization is implemented in `src/get1jtensor.jl`.

## Current representation and opportunity

`TLArray` represents a tensor with a symmetry tuple, per-leg q-label spaces,
sector slots, optional non-Abelian w-matrix storage, and reduced matrix tensor
(RMT) payloads. It also carries `TLIndex` metadata for tags, arrow direction,
prime level, lock, and dual-space state. The array-import constructor fills the
gap between ordinary Julia arrays and this representation.

The type parameters already admit a zero-length symmetry tuple in principle:
there are no q-label components, no non-Abelian w-matrices, and `RD == QD`.
That makes a dense tensor a natural degenerate case of `TLArray`, but it should
be made an explicit, documented use case and audited across algorithms rather
than relying on accidental support. In this case a leg's space can be encoded
as one trivial q-label `()` with multiplicity equal to the corresponding array
dimension, and the entire dense array is one RMT block. Although this tensor
has one sector, its w-matrix metadata still contains one empty descriptor for
that sector; Telum validates descriptor and sector counts equally.

## Ways to define a tensor without symmetry

### Chosen interface: import an array with leg metadata (implemented)

The direct constructor is:

```julia
TLArray(A::AbstractArray, inds::NTuple{QD, TLIndex})
```

It infers rank and scalar type from `A`, creates a no-symmetry tensor, and
stores a copy of `A` as the sole RMT block. The tensor's leg order and
dimensions match Julia's array axes. It accepts numeric element types and
requires one-based axes.

This form is explicit, composes with existing metadata tools, and lets callers
choose any combination of tags, directions, prime levels, locks, and dual
flags. A convenience keyword form can cover the common case:

```julia
TLArray(A; itags=("left", "right"), dirs=('+', '-'))
```

`itags` defaults to empty tags and `dirs` to `'+'`; both accept scalar or
per-leg tuple/vector values. A tuple of `TLIndex` values remains the path for
advanced metadata such as prime level, lock, or dual state. The constructor
copies the input data, so subsequent changes to `A` do not modify the tensor.
The normal `TLArray` index invariants apply: for example, tags must not produce
duplicate tagged indices, and empty tags cannot be combined with a nonzero
lock.

This is the only proposed import interface selected for implementation. It
keeps construction direct: callers already have either the array and its leg
metadata or can provide tags and directions inline.

Other constructor proposals considered earlier are not part of the selected
scope: constructing from `leginfo` values and adding a separate dense
conversion/import API. Existing `to_sparse_array` remains the current explicit
expansion path for symmetry tensors.

## Specialization scope

The all-Abelian `legflip` fast path is implemented for concrete `TLArray`
values, including deferred conjugation, scaling, and permutation views. Lazy
tensor wrappers continue through the established general path. This is the
only operation specialization selected for now.

### Leg flip

The generic path builds a one-leg identity-like tensor using `get1jtensor`,
contracts it into the source, and permutes the result. For all-Abelian concrete
tensors, the fast path maps the selected logical leg to its stored leg using
the deferred permutation, flips that stored `TLIndex.dir`, toggles
`TLIndex.dual`, dualizes the stored leg's q-labels and space list with
`get_dualq`, and reuses the RMT payloads. It preserves deferred conjugation,
scale, and permutation state. There are no non-Abelian basis changes or stored
w-matrices to transform. The no-symmetry case only changes arrow and dual
metadata because its q-label is always `()`. Lazy wrappers remain on the
existing path to avoid changing lazy evaluation behavior.

Contraction specialization is not selected: a second contraction path could
conflict with Telum's lazy evaluation and would need to duplicate or carefully
reproduce its semantics. Other candidate specializations, including
permutation, decompositions, `oplus`, and sums, are also deferred because the
expected performance gain is small relative to the added complexity.
