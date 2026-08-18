# Migrating from QSpace

Telum preserves several familiar ideas from QSpace while using Julia and a more
ITensor-like interface for tensor construction and contraction.

## Contraction

For explicit control of contracted leg positions, use `contract`:

```julia
C = contract(A, (1, 2), B, (2, 3))
```

For most tensor-network code, prefer tagged contraction:

```julia
C = A * B
```

`A * B` contracts legs with the same nonempty tags, opposite directions, zero
lock levels, and compatible prime levels, dual flags, and space lists.

## Legs and symmetry sectors

Telum records a direction on every leg: `+` for incoming and `-` for outgoing.
It also stores each leg’s symmetry space list explicitly. This information is
used to validate contractions and to retain zero symmetry sectors in operations
such as eigendecomposition.

Telum’s q-labels describe the symmetry sector of a space. For example, a
one-channel spinful fermion with U(1) charge and SU(2) spin has q-labels for
charge relative to half filling and `2S`.

## Local operators

`getLocalSpace` returns a named collection of local symmetry-adapted operators:
`q.I` is the identity, `q.Z` is fermionic parity, `q.F` is the fermion
annihilation IROP, and `q.S` is the spin IROP. Use `keys(q)` to inspect the
available operators.

## Decompositions

For `svd` and `qr`, select the legs that belong to the left factor. Telum
creates the required internal bond leg and preserves the symmetry structure of
the factors. Tags and leg-selection keywords can be used instead of explicit
leg numbers when that makes the algorithm clearer.
