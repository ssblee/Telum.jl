# QSpaces Functions Used in `example/`

This document is a shorter, editable tutorial for the QSpaces examples.  
Instead of explaining every single line, it explains the main QSpace functions and tensor operations that appear in:

1. `example/MPO.jl`
2. `example/DMRG_GS.jl`
3. `example/DMRG_central.jl`
4. `example/NRG.jl`

## How to Read the Examples

The examples mostly follow the same pattern:

1. Build a local symmetry-adapted space.
2. Create local operators as `QSpace` tensors.
3. Add or rename legs so those tensors can be contracted into MPOs or MPSs.
4. Use tensor contractions, decompositions, and truncation helpers to run DMRG or NRG.

Before the function list, here are the three most important ideas.

## Core Ideas

### 1. `QSpace`

`QSpace` is the symmetry-aware tensor type used everywhere in the examples.  
It stores:

- tensor blocks
- symmetry labels on each leg
- leg directions
- leg tags such as `"S,3"` or `"SB,7"`

You will often see:

- `QSpace(...)` to retag or rebuild a tensor
- `A * B` for tensor contraction
- `A'` for conjugation / dagger
- `A ⊗ B` for tensor-product construction

### 2. Legs and `itag`

A QSpace tensor is understood through its legs.  
The examples identify legs by string tags such as:

- `"site"`
- `"op"`
- `"SLeft"`, `"SRight"`
- `"OB,3"`, `"SB,5"`
- `"K,4"`, `"D,4"`

Many helper functions work by finding, renaming, adding, or removing these tagged legs.

### 3. Neutral Symmetry Sector

Many example constructions need the symmetry-neutral block at the boundary.  
This is why the examples repeatedly use helpers such as `zero_qlabels`, `getvac`, `getsub`, and `deleteSingleton`.

## Function Guide

## Local-Space Builders

### `SpinOptions(...)`

Used in `example/MPO.jl`.

Purpose:

- chooses the symmetry and local spin representation for a spin site

Examples:

```julia
option = SpinOptions(SU{2}, 1//2)
option = SpinOptions(U1, 1//2)
```

Interpretation:

- `SU{2}` keeps full spin symmetry
- `U1` keeps only conserved spin projection
- `1//2` means spin-1/2 local degrees of freedom

### `FermionSOptions(...)`

Used in `example/MPO.jl` and `example/NRG.jl`.

Purpose:

- chooses the local fermionic site with charge and spin symmetries

Example:

```julia
opt = FermionSOptions(1, :U1, :SU2, nothing)
```

Interpretation:

- `1` is the number of channels
- `:U1` is the charge symmetry over all channels
- `:SU2` is the spin symmetry over all channels
- `nothing` disables channel/flavor symmetry

For multiple channel groups, pass explicit `(symmetry, channels)` tuples:

```julia
opt = FermionSOptions(3, :U1, [(:SU2, [1, 2]), (:U1, [3])], :SU3)
```

### `getLocalSpace(...)`

Used in `example/MPO.jl` and `example/NRG.jl`.

Purpose:

- constructs the standard local operators for the chosen site type

Examples:

```julia
q = getLocalSpace(option, ("site", "site", "op"))
q0 = getLocalSpace(option, ("s,0", "s,0", "op"))
```

What it returns:

- a container of local QSpace operators such as `I`, `S`, `Sp`, `Sm`, `Sz`, `F`, `Z`

Typical fields used in the examples:

- `q.I`: identity
- `q.S`: spin operator multiplet
- `q.Sp`, `q.Sm`, `q.Sz`: spin components in `U1` examples
- `q.F`: fermion operator
- `q.Z`: fermion parity operator

## Boundary and Sector Helpers

### `zero_qlabels(...)`

Used in `example/MPO.jl` and `example/DMRG_central.jl`.

Purpose:

- returns the symmetry-neutral qlabel needed to select boundary blocks

Examples:

```julia
zq = zero_qlabels(t)
zq = zero_qlabels(MPO[1])
```

Why it matters:

- MPO boundaries are usually selected in the neutral sector
- the initialization code also uses it to extract a physical matrix from a QSpace tensor

### `getvac(...)`

Used in `example/DMRG_central.jl` and `example/NRG.jl`.

Purpose:

- creates a vacuum-like trivial state in a given space

Examples:

```julia
Aprev = getvac(MPO[1], ("SLeft", "SLeft"))
v = getvac(q0.I, ("K,vac", "K,vac"))
```

Typical role:

- start an MPS or NRG iteration from a trivial boundary state

### `getsub(...)`

Used in `example/MPO.jl` and `example/DMRG_central.jl`.

Purpose:

- extracts a selected symmetry block or subspace from a QSpace tensor

Examples:

```julia
getsub(t, 3, [(zq, -1)])
getsub(Hnow, bli, [(zq, 1)])
```

Typical role:

- cut an MPO bulk tensor into a left or right boundary tensor
- isolate the neutral outgoing block before calling a dense eigensolver

### `deleteSingleton(...)`

Used in `example/DMRG_central.jl`.

Purpose:

- removes a leg that has become trivial or one-dimensional

Example:

```julia
Hmat = deleteSingleton(getsub(Hnow, bli, [(zq, 1)]), bli)
```

Typical role:

- turn a QSpace object with a dummy boundary leg into an ordinary matrix-like object

## Leg and Tag Manipulation

### `addSingleton(...)`

Used in all example groups except the plotting section.

Purpose:

- adds a new trivial leg to a tensor

Examples:

```julia
i4d = addSingleton(q.I, (3, 4); itag=("left", "right"), dir=('+', '-'))
Hprev = addSingleton(Aprev, 3; itag="OLeft", dir='-')
```

Typical role:

- convert local operators into MPO-compatible tensors
- attach explicit left or right boundary legs

### `setitag(...)`

Used heavily in `example/MPO.jl`.

Purpose:

- renames the tag of a tensor leg

Examples:

```julia
s4d = setitag(s4d, 4, "right")
sc4d = permutedims(setitag(sc4d, 3, "left"), (2, 1, 3, 4))
```

Typical role:

- make temporary tensor legs match the naming convention expected by later contractions

### `removeitag(...)`

Used in `example/DMRG_GS.jl`.

Purpose:

- removes a suffix or temporary part of a leg tag

Examples:

```julia
MPS[i] = removeitag(U * S, "right")
MPS[i+1] = removeitag(S * Vd, "left")
```

Typical role:

- clean up temporary tags produced during two-site SVD splits

### `findleg(...)`

Used in `example/DMRG_GS.jl`, `example/DMRG_central.jl`, and `example/NRG.jl`.

Purpose:

- locates a leg by tag and optionally direction

Examples:

```julia
li = findleg(MPS[1]; itag="SLeft")
bli = findleg(Hnow; itag=i==N ? "ORight" : "OB,$i")
l = findleg(Hprev; dir='-')
```

Typical role:

- identify exactly which leg should be used in `getIdentity`, `getsub`, or an SVD split

## Tensor Construction Helpers

### `getIdentity(...)`

Used in all four examples.

Purpose:

- builds an identity tensor between compatible spaces

Examples:

```julia
opid = QSpace(getIdentity((q.S, 3)), ("left", "right"))
Anow = getIdentity((Aprev, li), (MPO[i], 2); itag="SL,$i")
left_id = getIdentity((MPS[1]', li); itag="SLeft")
```

Typical role:

- create a basis-expansion map
- create an identity operator on a boundary space
- connect an existing kept space to a newly added local space

### `oplus(...)`

Used in `example/MPO.jl`.

Purpose:

- forms a direct sum of QSpace objects along selected legs

Example:

```julia
t = oplus(qss, (3, 4))
```

Typical role:

- combine a symbolic MPO transfer matrix into one bulk MPO tensor template

### `legflip(...)`

Used in `example/MPO.jl`.

Purpose:

- flips the direction or interpretation of a leg

Examples:

```julia
ZF_flip = setitag(legflip(lock(q.Z, 1) * q.F, 3), 3, "left")
fc_flip = addSingleton(legflip(q.F', 3), 3; itag="left", dir='+')
```

Typical role:

- build fermionic operators with the correct leg orientation for MPO assembly

### `permutedims(...)`

Used in `example/MPO.jl` and `example/NRG.jl`.

Purpose:

- reorders tensor legs

Examples:

```julia
sc4d = permutedims(setitag(sc4d, 3, "left"), (2, 1, 3, 4))
A0 = permutedims(getIdentity((v, 2), (H0, 2); itag="K,0"), (1, 3, 2))
```

Typical role:

- rearrange physical and virtual legs into the order expected by later contractions

## Contraction and Stabilization Helpers

### `lock(...)`

Used everywhere in the tensor-network parts of the examples.

Purpose:

- freezes or stabilizes the leg/tag structure of an intermediate contraction

Examples:

```julia
Hnow = Anow' * lock(Anow * Hprev * MPO[i]; itag="SL,$i")
Hlr[i+1] = MPS[i]' * lock(Hlr[i] * MPO[i] * MPS[i]; itag="SB,$i")
n0 = lock(q0.F', 2) * q0.F
```

Why it appears so often:

- long contractions can leave temporary leg metadata that should be fixed before the next operation
- many examples use `lock` just before an adjoint or before reusing an intermediate tensor

### `*` on `QSpace`

Used in every example.

Purpose:

- contracts compatible legs between QSpace tensors

Examples:

```julia
Hprev * MPO[i]
MPS[i] * MPO[i]
Fprev * Anow * Fnow
```

Interpretation:

- this is tensor contraction, not plain scalar multiplication
- scalar prefactors such as `J * s4d` or `-t * f4d` are also supported

### `'` on `QSpace`

Used in all algorithmic examples.

Purpose:

- returns the adjoint / conjugate tensor

Examples:

```julia
Anew'
MPS[i]'
q.F'
```

Typical role:

- build expectation values
- form Hermitian operators
- project tensors back into a basis

### `⊗`

Used in `example/MPO.jl`.

Purpose:

- takes the tensor product of two QSpace objects

Example:

```julia
qss[3, 2] = q.I ⊗ opid
```

Typical role:

- combine local identity action with an operator-space identity channel

## Decomposition and Truncation Helpers

### `svd(...)`

Used in `example/DMRG_GS.jl`.

Purpose:

- splits a tensor into left factor, singular values, and right factor

Examples:

```julia
U, S, Vd = svd(M, "temp", target_tag; itag=target_tag)
U, S, Vd = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=Nkeep)
```

Typical role:

- move the canonical center in one-site DMRG
- split a two-site tensor back into two MPS tensors
- truncate bond dimension with `Nkeep`

### `eigen(...)`

Used in `example/DMRG_central.jl`, `example/DMRG_GS.jl`, and `example/NRG.jl`.

Purpose:

- diagonalizes a matrix-like QSpace object or dense projected matrix

Examples:

```julia
e = eigen((Hmat + Hmat') / 2)
_, V = eigen(Hkrylov)
e = eigen(Hnow)
```

Typical role:

- find local low-energy states
- solve effective DMRG eigenproblems
- diagonalize NRG effective Hamiltonians

### `discard_eigen(...)`

Used in `example/DMRG_central.jl` and `example/NRG.jl`.

Purpose:

- splits an eigendecomposition into kept and discarded sectors

Examples:

```julia
ek, _ = discard_eigen(e, i==N ? 1 : Nkeep, i==N ? "SRight" : "SB,$i", "SD,$i")
ek, ed = discard_eigen(e, Nkeep, "K,$si", "D,$si")
```

Typical role:

- keep only the low-energy states
- label the kept space and discarded space for later tensor construction

Outputs commonly used:

- `ek.V`: kept eigenvectors
- `ek.D`: kept-space effective Hamiltonian
- `ek.eig_list`: metadata for eigenvalues and symmetry sectors

## QSpace Patterns in Each Example

## `example/MPO.jl`

Main QSpace functions:

- `getLocalSpace`
- `addSingleton`
- `setitag`
- `permutedims`
- `getIdentity`
- `legflip`
- `oplus`
- `zero_qlabels`
- `getsub`
- `QSpace`

Main idea:

- build local operators first, then reshape and retag them until they fit the MPO layout

## `example/DMRG_central.jl`

Main QSpace functions:

- `zero_qlabels`
- `getvac`
- `addSingleton`
- `findleg`
- `getIdentity`
- `lock`
- `getsub`
- `deleteSingleton`
- `eigen`
- `discard_eigen`

Main idea:

- create an initial MPS by repeatedly enlarging the kept space, projecting the Hamiltonian, and truncating back down

## `example/DMRG_GS.jl`

Main QSpace functions:

- `findleg`
- `getIdentity`
- `addSingleton`
- `lock`
- `svd`
- `removeitag`

Main idea:

- contract left and right environments around one site or two sites, optimize locally, then split the result back into MPS form

## `example/NRG.jl`

Main QSpace functions:

- `getLocalSpace`
- `getvac`
- `getIdentity`
- `permutedims`
- `lock`
- `findleg`
- `discard_eigen`
- `QSpace`

Main idea:

- build a symmetry-adapted impurity basis, enlarge it site by site, diagonalize, then keep only the low-energy sectors

## Minimal Reading Path

If you want to understand the examples quickly, this order works well:

1. Read `getLocalSpace`, `QSpace`, `addSingleton`, `setitag`, and `permutedims`.
2. Then read `getIdentity`, `findleg`, `lock`, and `*`.
3. Then read `svd`, `eigen`, and `discard_eigen`.
4. After that, the example files become much easier to follow.

## Short Summary

If you only remember one sentence for each group:

- construction: `getLocalSpace`, `QSpace`, `addSingleton`, `setitag`, `permutedims`
- navigation: `findleg`, `getsub`, `deleteSingleton`, `zero_qlabels`
- contraction: `*`, `'`, `⊗`, `lock`
- truncation: `svd`, `eigen`, `discard_eigen`

That is most of the QSpaces vocabulary used by the examples.
