# Channel SO(3) symmetry for spinful fermions

Status: implemented in `src/localspaces/FermionS.jl`, with regression coverage
in `test/test_localspace.jl`.

## Goal and scope

Support `:SO3` and `:SO` as channel symmetries in `FermionSOptions` for exactly three
ordered channels. The three channels form the spin-1 vector representation of
SO(3), with channel 1, 2, and 3 carrying `J_z` weights `+2`, `0`,
and `-2`. Each channel has both spin components. The first target is the
existing 64-state (`4^3`) local Fock space and its symmetry-adapted operators.
An explicit three-channel group may also be selected within a larger local
space, subject to the existing channel-group rules.

The numbered spelling should be `FermionSOptions(3, :U1, :SU2, :SO3)` or
`(:SO3, [a, b, c])` in a channel-group option. Match the existing SU(N)
syntax by also accepting `:SO` for a group of three channels, including
`FermionSOptions(3, :U1, :SU2, :SO)` and `(:SO, [a, b, c])`.
`:SO4` and other numbered SO symbols must be rejected: this plan supports
SO(3) only. The order `[a, b, c]` assigns the `+2`, `0`, and `-2` weights;
it must not be sorted implicitly.

## Operator convention

Let `N_i = sum(FF[σ, i]' * FF[σ, i] for σ in 1:2)` and
`E_ij = sum(FF[σ, i]' * FF[σ, j] for σ in 1:2)`. For a group `[a, b, c]`,
use the physical generators

```text
J_z = 2 * (N_a - N_c)
J_+ = sqrt(2) * (E_ab + E_bc)
J_- = J_+' = sqrt(2) * (E_ba + E_cb)
```

Thus `J_+` moves a particle from channel 3 to 2 or from 2 to 1 when the
group is `[1, 2, 3]`. With this normalization,
`[J_z, J_+] = 2J_+`, `[J_z, J_-] = -2J_-`, and
`[J_+, J_-] = J_z`. Build them from the existing
`FF` matrices so the fermionic signs follow the established mode ordering.

`getSymmetryInfo` supplies **lowering** matrices to `decompose_space`, so the
new channel entry must provide `J_-` with `sqrt(2)` transfer coefficients. LurCGT
stores SO(3) z weights doubled relative to the usual magnetic quantum number:
provide each Fock state's weight as `(2 * (N_a - N_c),)`. Start with the
`sqrt(2)` lowering operator above.

For the same ordered channel group, `J_z` is the sum of the two existing SU(3)
channel Cartan operators and `J_-` is `sqrt(2)` times the sum of its two
adjacent lowering operators. The existing `getLocalSpace` cross-symmetry
validation still runs; no new compatibility logic or separate compatibility
test is planned.

## Implementation steps

1. Update `src/localspaces/options.jl` documentation to list `:SO3` and `:SO` and their
   exactly-three-channel requirement. Keep the existing `FermionSOptions`
   storage and option normalization.
2. Extend `chan_symmops` in `src/localspaces/FermionS.jl` with a specific
   SO(3) branch for `:SO3` and `:SO`. Validate the three selected channel indices, append
   `SO{3}`, the doubled weights, and one `J_-` matrix. Leave the SU(N) branch
   and its Cartan convention intact. Reject unrecognized channel symbols
   explicitly.
3. Widen the two channel-symbol checks in `add_spin_irop!` and
   `_split_annihilation_blocks_by_channel_symmetry` to accept both SO(3)
   spellings. Reuse their existing grouping, overlap checks, and operator
   selection without changing `_add_annihilation_block!`. Its last-channel
   annihilation operator has channel `J_z` weight `+2`, so it is the SO(3)
   maximal-weight component; channel-summed spin operators remain scalars.
   Verify both claims by reconstructing the operators in the tests.
4. Keep the general `getLocalSpace` pipeline and tensor storage interfaces
   unchanged. Add no new option type, tensor subtype, or separate channel
   representation layer unless a concrete failure demonstrates a need.
5. Extend the existing option documentation and local-space examples with
   both SO(3) spellings and the ordering convention.

## Validation

Add focused cases to `test/test_localspace.jl`:

- Verify the three commutators above on the full Fock basis, and that `J_+`
  acts on one-particle states in the requested 3-to-2-to-1 direction with
  `sqrt(2)` coefficients.
- Check the one-particle channel multiplet has weights `(2,)`, `(0,)`,
  `(-2,)` in LurCGT's convention and that the full 64-state space decomposes
  and reconstructs its identity, parity, spin, and fermion operators.
- Exercise `:SO3` alone and with total `:U1` charge and `:SU2` spin. Compare
  expanded operators to the original sparse Fock-basis matrices, accounting
  for the symmetry-adapted basis transformation.
- Check that `:SO` and `:SO3` select identical symmetry data for a three-channel
  group, and reject `:SO4` and other unsupported numbered SO symbols.
- Reject groups with fewer or more than three channels, repeated or invalid
  indices, and overlapping channel groups. Test a nonconsecutive ordered group
  if groups within larger local spaces are supported.

Run the relevant local-space tests, then relevant factorization and
contraction tests using the new option. Inspect type stability with
`@code_warntype` or JET where practical. Report allocation impact if existing
benchmark tools cover local-space construction or subsequent tensor operations.
Do not run the documentation build unless explicitly requested.

If decomposition or operator reconstruction fails with the specified
`sqrt(2)` operator, inspect LurCGT's internal basis normalization at that
point. The two channel-symbol checks in implementation step 3 remain required
so the existing spin and fermion operator selection rules handle `:SO3` and
`:SO`.
