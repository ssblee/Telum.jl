# Fuse and split tensor legs

## Goal

`fuselegs` combines two or more logical legs of one tensor into one leg and
returns both the resulting tensor and, when applicable, the fusion isometry.
The implementation has separate paths for unsymmetric and symmetric tensors.

This document describes the public API and behavior of `fuselegs` and
`splitleg`.

## Public interface

```julia
fuselegs(q, legs; itag="", dir='-', plev=0, lock=0)
```

The leg metadata keywords are scalar values: `itag` is a string, `dir` is a
character, and `plev` and `lock` are integers. The default fused-leg direction
is outgoing (`dir='-'`). Empty `itag` values are allowed. Metadata must follow
the normal `TLIndex` construction and normalization rules; this proposal adds
no special empty-tag/lock restriction.

`legs` is an ordered tuple or vector containing at least two distinct,
in-range logical leg numbers. Its order is significant: do not sort it. The
order determines the selected legs' order in the fused basis, with the first
selected leg varying fastest according to Julia's column-major storage order.
Reject invalid leg selections with clear argument or bounds errors.

Unselected legs retain their original relative order. The fused leg occupies
the output position of the first selected leg in the input's logical order.
There is no separate `position` argument.

The return value is always an explicit pair:

```julia
(fused_tensor, isometry)
```

For a symmetric tensor, `isometry` is the isometry returned by the
`getIdentity` call used to fuse the selected legs. For an unsymmetric tensor,
return `(fused_tensor, nothing)`.

## Unsymmetric tensors

For a tensor without symmetries, fusion is a permutation followed by a
reshape. Resolve/evaluate all deferred view states so the returned tensor has
the default `perm`, `conj`, and `scale` state. Permute the logical axes into
this order:

1. unselected legs that precede the first selected leg, in their original
   relative order;
2. selected legs in the caller-provided `legs` order;
3. all remaining unselected legs, in their original relative order.

The selected axes are now contiguous and can be reshaped into the fused leg.
This ordering places the fused leg at the position of the first selected leg
while retaining the requested selected-leg basis order. Set the fused
direction from `dir` (default `'-'`). Return `(fused_tensor, nothing)`.

Allocate new RMT storage by default and do not mutate the input. The only
case where the unsymmetric path may reuse the input's RMT storage is when both
conditions hold: (1) `conj`, `perm`, and `scale` are all trivial, and (2) the
selected legs are adjacent in the input's logical order and occur in the same
order as in `legs`. In that case, fusion is a reshape of the already-contiguous
axes and the result may share the RMT storage. This path does not need to
allocate or return an isometry.

## Symmetric tensors

For symmetric tensors, including tensors with only Abelian symmetries, use the
conventional symmetry-aware construction: obtain the fusion isometry from
`getIdentity` and contract the selected legs against it, accounting for the
input tensor's view state. The resulting tensor must have trivial `conj` and
`scale` state. If the requested fused direction is incoming (`dir='+'`), apply
`legflip` to the fused result after contraction. Return the exact isometry
produced by `getIdentity` as the second item of the result pair.

The final placement of the fused leg is a logical permutation represented
through the result's `perm` field only. Do not reorder or copy RMT payload just
to place the fused leg. The isometry and contraction must respect the selected
legs' caller-specified order and the resulting fused basis convention.

A reshape alone is not sufficient for symmetric tensors: it does not encode
symmetry labels, Clebsch–Gordan basis changes, or fusion multiplicities.

## Risks and recommendations

The main correctness risk is a mismatch between the selected-leg order used to
construct the fusion basis and the order used to permute/contract the tensor.
Such a mismatch can produce plausible numerical values in an incorrect basis.
Test reversed selected-leg order explicitly, as well as noncontiguous selected
legs, mixed directions, output placement, and unchanged input data. For
symmetric tensors, include Abelian and non-Abelian cases and verify that the
returned isometry corresponds to the actual contraction. A useful non-Abelian
check is to reconstruct the original tensor from the fused tensor and returned
isometry, then verify that the norm of their difference is small. If the fused
leg has incoming direction, first contract a 1j-tensor with the `getIdentity`
result as needed to make the reconstruction's isometry orientation match the
fused leg. For unsymmetric tensors, test nontrivial deferred `perm`, `conj`,
and `scale` states and confirm the output is in default view state.

Preserve all unselected leg metadata and relative order. Do not impose an
additional empty-tag/lock rule beyond the existing `TLIndex` rules. Keep the
two implementation paths explicit so that the unsymmetric reshape path does
not accidentally discard symmetric basis information.

## `splitleg` interface

```julia
splitleg(q, leg, dims; itags="", dirs=nothing, plevs=0, locks=0)
```

The plural metadata keywords describe the replacement legs. Each accepts a
scalar broadcast across all replacement legs or a tuple/vector with one value
per replacement leg. `dirs` defaults to the selected leg's logical direction;
the other defaults are empty tags and zero prime/lock levels. `dims` contains
at least two positive dimensions and may contain one `-1` for inference. The
function returns the tensor only, preserves `conj`, `scale`, and `perm`, and
shares RMT storage for unsymmetric tensors. Symmetric input currently raises an
error.
