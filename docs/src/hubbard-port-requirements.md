# Requirements for porting the Hubbard setup from QSMPSLib

This note scopes a Julia conversion of the **Hamiltonian MPO construction and
initial MPS setup** in `Run_DMRG_Hubbard.m`. The first implementation targets
`Acharge,SU2spin`, with the two-site DMRG path as its eventual consumer. Keep
the model and symmetry dispatch organized so additional settings can be added
later, while implementing only this setting in the first version. It does not
specify the DMRG sweep, observables, disk storage, or parallel workers.

The purpose is to compare the performance of the MATLAB and Julia codes.
Treat the MATLAB implementation as the reference: preserve its algorithms,
data flow, defaults, ordering, and numerical choices wherever this scope does
not explicitly narrow them. Avoid Julia-only redesigns that change the work
being measured or the resulting tensors.

## Implementation location

Implement the conversion in a **new benchmark directory** at
`C:\Users\lurlu\.julia\dev\Telum\benchmark`. Put the converted code and
benchmark-specific files there, rather than in the existing package `src/`
tree. The benchmark may depend on the local `Telum` package and its `LurCGT`
backend. Keep this requirements document under `docs/` as its design
reference.

## Source path and boundary

The MATLAB run script calls:

1. `IniHamilMPO_HubbardSq(Para)` to build the local operators, lattice-ordered
   hopping map and Hubbard Hamiltonian MPO.
2. `PrepObsOp_HubbardSqLatt(Op, Para)` to add measurement operators. This is
   outside the requested scope.
3. `IniMPS_FermionS_unihole(Op.Z, Para.L, Para.Ntot, Para.Stot)` when no saved
   MPS is supplied. This creates an initial state in a chosen total charge and
   spin sector.
4. `mainDMRG` for ordinary two-site updates (`Para.CBE == 0`). This is not a
   MATLAB solver port in this change; the existing Telum two-site DMRG
   implementation is made available under `benchmark/` for the converted
   Hamiltonian and initial MPS to use.

`IniHamilMPO_HubbardSq` delegates lattice ordering to
`util/IntrcMapSqLatt.m`, operator construction to `util/PrepSpaceFermionS.m`,
and the Hamiltonian MPO assembly to helpers including `GenPseudoMPO`,
`pMPO2MPO`, `GenMPO_OnsiteHam`, `DrctSumA`, and `ComprsMPO`.

## Model and sign conventions to preserve

For the default `Acharge,SU2spin` branch, the Hamiltonian is the ordinary
spinful Hubbard model on the requested graph:

```math
H = -t \sum_{(i,j)\in E_1,\sigma}
    (c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.})
  -t' \sum_{(i,j)\in E_2,\sigma}
    (c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.})
  +U\sum_i n_{i\uparrow}n_{i\downarrow}
  -\mu\sum_i n_i.
```

The MATLAB map is called with `J1=-Para.t` and `J2=-Para.tp`; keep this sign
convention explicit when building the Julia MPO. In a local-operator
formulation, `n_up*n_down` is equivalently `n*(n-1)/2`. The MATLAB run sets
`mu=0` for `Acharge,SU2spin`, but the model builder accepts a general `mu`.
The optional magnetic field is fixed to zero in the run script and breaks
spin-rotation symmetry, so it is not part of this symmetry setting.

The default geometry is `Lx=8`, `Ly=4`, `BCX=OBC`, `BCY=PBC`, path `ZigZag`,
`t=1`, `tp=0`, `U=8`, `Ntot=24`, and `Stot=0`. The site order determines the
MPS chain and the interaction endpoints. In general, the port needs a clearly
defined contract for `Lx`, `Ly`, both boundary conditions, the site-order path,
and whether next-nearest-neighbor hopping `tp` is included.

## Default-value parity

The Julia benchmark's default configuration must match `Run_DMRG_Hubbard.m`
exactly. The conversion-relevant defaults are:

| Parameter | MATLAB default |
| --- | --- |
| `Lx`, `Ly` | `8`, `4` |
| `BCX`, `BCY`, `path` | `OBC`, `PBC`, `ZigZag` |
| `Symm` | `Acharge,SU2spin` |
| `t`, `tp`, `U` | `1`, `0`, `8` |
| `Ntot`, `Stot` | `24`, `0` |
| `mu`, `h` | `0`, `0` |
| `CBE` | `0` (ordinary two-site DMRG path) |
| `MPSSource` | empty (initialize a new MPS) |

Preserve these values as defaults in the Julia configuration API; do not choose
different defaults for convenience. Preserve both boundary-condition inputs
(`OBC`/`PBC`) and their behavior. Use only the `ZigZag` path in this first
version; other MATLAB site-order paths are outside the current scope. The
source DMRG settings are `K=5`, `Ktol=1e-12`,
`DStep=[200,500,1000,2000,3000,4000,6000,8000,10000]`, `Nrep=5`, and
`Nsw1=0`. For the current benchmark run, use the explicitly shortened
`DStep=[200,500,1000,2000,4000,10000]`, keeping `Nrep=5`, `K=5`,
`Ktol=1e-12`, and `Nsw1=0`. `benchmark/two_site_dmrg.jl` accepts a per-sweep
`Nkeep` schedule. Set its Lanczos tolerance and Krylov count from `Ktol` and
`K`.

The first performance comparison uses the default geometry. Support variable
`Lx`/`Ly` and both boundary-condition settings with `path=ZigZag`; do not add
`Snake`, `Slash`, or `Diagonal` ordering until requested. It is not necessary
to test every lattice size and boundary-condition combination before the
default-case conversion can be checked.

## Symmetry contract

- One local site is a spinful fermion with states `|0>`, `|up>`, `|down>`,
  `|up down>`; local dimension is four.
- The conserved symmetry is `U(1)` particle number × `SU(2)` spin. In Telum's
  current local-space API this is represented by
  `FermionSOptions(1, :U1, :SU2, nothing)` and
  `getLocalSpace(options, ("site", "site", "op"))`.
- Telum's local `U1` charge is measured relative to half filling for one
  channel: empty and double states have charges `-1` and `+1`; singly occupied
  states have charge `0`. This matches the MATLAB initializer's final total
  charge target `Ntot-L`.
- Telum generates spin weights from twice `S_z`; the MATLAB parameter
  `Stot` is also documented as twice the total spin. Confirm that the
  irreducible-representation label used for the global MPS boundary is `Stot`
  in the relevant Telum APIs before using it as a target.
- Hopping operators must preserve fermionic signs, including the Jordan–Wigner
  parity strings. The MATLAB `PrepSpaceFermionS` makes distinct left/right
  operator tensors and inserts parity `Z`; a Julia port must not replace these
  with naive local bosonic products.

## MPO construction requirements

The resulting object should be a chain of rank-four `TLArray` MPO tensors with
physical input/output legs and adjacent virtual bond legs. Required behavior:

1. Build the site-order permutation and graph edge list from lattice size,
   boundary conditions, and path. Translate each physical edge to ordered MPS
   sites while preserving its coupling.
2. Build the identity, number, doublon, annihilation/creation, and parity
   tensors in the selected mixed symmetry. Make the orientation and tags
   explicit at every virtual and physical leg.
3. Encode each hopping edge and its Hermitian conjugate once with the correct
   `-t` or `-tp` coefficient, then add onsite `U` and `-mu` terms.
4. Close the left and right virtual bonds to their vacuum sectors. Verify the
   returned MPO contracts to a Hermitian operator and respects total charge
   and spin.
5. Return only the Hamiltonian MPO. Do not create separate MPOs for total
   filling/particle number, its square, correlations, or other observables.
   Local number operators may still be used internally to form onsite terms
   such as `U*n*(n-1)/2 - mu*n`.

Preserve the source's Hamiltonian construction sequence and compression
settings where they apply. In particular, MATLAB compresses the Hamiltonian
MPO with SVD using `stol=1e-14` and maximum bond dimension `1000`. Do not
substitute a different algorithm or threshold in the primary performance
comparison; record any alternate settings as separate variants.

Telum's `oplus` accepts vector and matrix inputs for direct sums. Use it when
combining operator-valued MPO channels or assembling the local MPO tensor if
that makes the construction clearer. A matrix of local operator blocks is a
natural representation of an MPO automaton; the existing DMRG tutorial and
`test/support/dmrg.jl` show matrix-based assembly and endpoint restriction to
the vacuum sector. A vector input can be simpler for combining channels with
the same layout. Choose the representation that keeps virtual-sector meaning,
leg direction, and endpoint handling easiest to follow.

## Preserve the MATLAB organization

Keep the Julia layout recognizable relative to the MATLAB run script and
Hamiltonian initializer:

- Keep run parameters together: lattice size, boundary conditions, path,
  couplings, symmetry setting, and target `Ntot`/`Stot`.
- Keep symmetry dispatch in the Hamiltonian initializer, analogous to the
  `switch Para.Symm` in `IniHamilMPO_HubbardSq`. Implement
  `Acharge,SU2spin` now and return a clear unsupported-setting error for
  branches that have not been ported yet.
- Keep local operator setup, lattice interaction mapping, Hamiltonian
  construction, and MPS initialization in distinct functions/modules. Avoid
  flattening the port into one monolithic builder.
- Preserve extension points for additional symmetry settings and the original
  geometry options without implying that unimplemented symmetry branches are
  supported.

The current Telum DMRG validation fixture already has a `HubbardMPO` builder
in `test/support/dmrg.jl`. It is a useful reference for spinful local
operators, parity strings, `oplus`, and virtual-bond closure, but it builds a
**nearest-neighbor one-dimensional chain**. It is test support rather than a
public model API and does not replace the square-lattice interaction-map
logic in this port.

The generic two-site DMRG routines (`eigs_GS`, `getHrl`,
`DMRG_GS_2site!`, and generic `init_MPS`) have been extracted to
`benchmark/two_site_dmrg.jl` under the `BenchmarkDMRG` module. Test support
includes that file so existing integration checks can keep using those
routine names. The Hubbard MPO and sector-specific initial MPS conversion
remain separate inputs to this reusable sweep code.

## Initial MPS requirements

The MATLAB `IniMPS_FermionS_unihole` initializer accepts `L`, `Ntot`, and
`Stot`, supports the hole-doped/half-filled range `0 ≤ Ntot ≤ L`, places
`L-Ntot` empty sites approximately uniformly in the MPS order, uses singly
occupied local states elsewhere, and grows a symmetry-resolved state from one
boundary. It prunes intermediate spin sectors, then selects the requested
global `(Ntot-L, Stot)` sector at the final boundary and orthogonalizes the
MPS. Supporting electron-doped inputs `Ntot > L` requires a new initializer
that places doublons; it is not a literal translation of this MATLAB routine.

The converted initializer must preserve the requested target sector. The
current Julia implementation uses Telum's iterative symmetry-resolved
initializer with an explicit global charge and spin target. It does not yet
reproduce MATLAB's evenly spaced-hole product seed and pruning sequence. Since
the target sector is preserved, this is suitable for correctness checking;
record the initialization difference in performance comparisons.

A Julia initializer for equivalent behavior needs to:

- For a literal first port of `IniMPS_FermionS_unihole`, validate
  `0 ≤ Ntot ≤ L`, and that the requested charge/spin sector is allowed for
  the site count and particle number. Supporting `Ntot > L` can be added
  later with a doublon-aware initialization strategy.
- Keep the requested global charge and total-spin irrep at the MPS boundary;
  do not let the initialization select an arbitrary lowest sector.
- Produce a nonzero, normalized MPS with compatible adjacent bond sectors and
  canonical/gauge conditions suitable for a two-site sweep, following the
  MATLAB initialization sequence.
- Preserve the MATLAB evenly spaced-hole configuration and `Stot`-dependent
  sector pruning for the primary benchmark. Other initializers are separate
  benchmark variants.

`benchmark/hubbard_mps.jl` calls the generic `init_MPS` in
`benchmark/two_site_dmrg.jl` with an explicit target sector
`((Ntot-L,), (Stot,))`. This keeps charge and spin fixed while using a
different seed construction from MATLAB's evenly spaced holes.

## Suggested public-facing interface

Keep geometry, model parameters, and symmetry explicit rather than copying the
MATLAB `Para` struct wholesale. For example, a future interface could follow
this shape (names are illustrative, not committed API):

```julia
params = HubbardDMRGParameters(Lx=8, Ly=4, BCX=:OBC, BCY=:PBC,
                           path=:ZigZag,
                           space_options=FermionSOptions(1, :U1, :SU2, nothing),
                           t=1, tp=0, U=8, mu=0, Ntot=24, Stot=0)
ops = prepare_hubbard_operators(params)
H = initialize_hubbard_hamiltonian(params, ops)
psi0 = initialize_hubbard_mps(params, H)
```

These names illustrate the desired separation and are not a finalized API.
Do not add constructors for out-of-scope observable MPOs to this interface.

## Acceptance checks for the later code change

- The four local basis states have the expected relative charge and spin
  sectors, and creation/annihilation tensors carry the correct symmetry
  quantum numbers.
- On a small open square, materializing the Hamiltonian MPO agrees with an
  independently assembled exact Fock-space Hamiltonian, including the `-t`,
  `U`, and `-mu` signs.
- For a small square lattice, generated edge endpoints match the selected
  site-order path and boundary conditions, including periodic edges.
- The MPO is Hermitian and commutes with total charge and total `SU(2)` spin.
- The initial MPS has the requested boundary sector, is normalized/nonzero,
  and can be passed to the existing two-site update without gauge or leg-tag
  mismatches.
- On a small system whose bond dimension can be saturated, the two-site DMRG
  energy agrees with exact diagonalization in the same `(Ntot, Stot)` sector.
- Also run the default 8×4 benchmark at `Nkeep=10000` and compare its energy
  per site with the author's reference below.

### Exact-diagonalization fixture

Use a 2×2 open-boundary Hubbard lattice as a small end-to-end fixture:
`Lx=2`, `Ly=2`, `BCX=OBC`, `BCY=OBC`, path `ZigZag`, `t=1`, `tp=0`,
`U=8`, `mu=0`, `Ntot=4`, and `Stot=0`. Construct the reference Hamiltonian
directly in the fermionic occupation-number basis (or its fixed-charge block),
using creation/annihilation signs and the physical lattice edge list. Do not
obtain the reference by densifying the MPO or by reusing the MPO builder's
operator/channel assembly, so the comparison can detect MPO construction
errors.

Check that the MPO's fixed-sector matrix and low-energy spectrum agree with
this reference. Then run two-site DMRG with a bond cap large enough to represent
the exact ground state and compare the variational energy to the exact
`(Ntot=4, Stot=0)` ground-state energy. Confirm the returned state has the
target particle number and spin and is normalized. A separate small lattice
with a periodic boundary should check that periodic edges are mapped and
counted according to `IntrcMapSqLatt`; compare it with an independently built
Fock-space reference as well.

## Benchmark comparison protocol

- Use matching parameters and the same symmetry, lattice site order, local
  basis, and MPO compression settings in both languages.
- The initial MPS must be in the same requested `(Ntot, Stot)` sector. Its
  individual random tensor values need not match MATLAB's; use the same
  initialization algorithm and record the seed for reproducibility.
- Measure Hamiltonian MPO construction/compression and initial MPS creation
  as separate phases; report end-to-end setup time separately if useful.
- Keep package compilation/JIT warm-up separate from steady-state timings, and
  record runtime environment, thread count, and Julia/MATLAB versions.
- Compare resulting MPO/MPS sector structure and numerical content before
  interpreting timing differences. Record any detail that prevents exact
  workload parity.
- Do not include observable MPO construction or correlation/filling
  calculations in this benchmark scope.

## Ground-state energy reference

The original-code author reports a ground-state energy per site of
`-0.9210993644419946` for the default 8×4 Hubbard run at `Nkeep=10000`. For
`L=32`, this corresponds to total energy `-29.475179662143827`. Use this as
an end-to-end validation target after the Julia Hamiltonian MPO and
sector-matched initial MPS are connected to the user's two-site DMRG routine
in `benchmark/two_site_dmrg.jl`.

For the first MPO check, run 4–5 two-site sweeps at `Nkeep=10000`, using
`K=5` and `Ktol=1e-12`, and record the energy after each sweep and the final
energy per site. Treat the reported value as the convergence reference: a
4–5-sweep result is an initial check, so compare its distance to the reference
and its sweep-to-sweep trend. The initial MPS may have different random tensor
entries as long as it is in the target sector.

The moved routine accepts a scalar keep dimension or a per-sweep schedule.
The current shortened schedule can be called like this:

```julia
include(joinpath(@__DIR__, "two_site_dmrg.jl"))
using .BenchmarkDMRG

DStep = [200, 500, 1000, 2000, 4000, 10000]
DList = repeat(DStep; inner=5)
psi, energy = DMRG_GS_2site!(psi, H, DList, length(DList);
                            tol=1e-12, nKrylov=5)
energy_per_site = energy / 32
```

Here `psi` is the output of the converted target-sector initializer and `H` is
the Hamiltonian MPO. This direct call avoids the generic benchmark
`init_MPS`, whose sector choice does not match the MATLAB Hubbard initializer.
For the initial 4–5-sweep check, pass `10000` and `4` or `5` directly instead
of `DList`.

## Conversion status

The conversion now lives in `benchmark/`: parameter defaults and symmetry
dispatch, ZigZag geometry, local operator setup, hopping-channel planning,
Hamiltonian MPO materialization, target-sector MPS initialization, the
reusable two-site DMRG routine, a standalone Fock-space ED reference, and a
benchmark runner. It builds only the Hamiltonian MPO. Correlation, filling,
and other observable MPOs are omitted as requested.

Validation performed:

- `julia benchmark/validate_hubbard_ed.jl` checks the independent 2×2 OBC ED
  fixture (`E0 = -1.320234958271926` for the lowest fixed-`Ntot=4` sector).
- `julia benchmark/validate_default_mpo.jl` builds the default 8×4 OBC/PBC
  ZigZag MPO; it has 32 sites and maximum virtual dimension 10 before SVD
  compression.
- `julia benchmark/validate_hubbard_mpo.jl` runs four two-site sweeps on 2×2
  OBC in the target `(Ntot=4, Stot=0)` sector and agrees with ED to numerical
  precision (`-1.320234958271928` versus `-1.3202349582719277`).

The raw automaton MPO is validated end to end. MATLAB's `ComprsMPO` settings
(`stol=1e-14`, `maxD=1000`) have a corresponding experimental Julia SVD helper,
but the compressed MPO currently fails inside the DMRG environment
contractions. Keep `compress=false` for validated runs until that integration
is corrected; this means MPO compression timing and parity are not yet
available. The default 8×4 run also exposed an initializer limitation:
`Nkeep_init=200` prunes away the requested `((-8,), (0,))` sector before the
last site, so it cannot construct the target MPS. Raising the cap to 1,000
made initialization too slow to complete in the current check, and the
original 8×4 DMRG was therefore not run. The author's
`-0.9210993644419946` per-site reference remains outstanding. Porting the
MATLAB sector-aware unihole seed (or implementing an equivalently efficient
target-preserving initializer) is required before the default benchmark can be
considered complete. The runner otherwise uses the shortened six-value
`DStep` schedule with `Nrep=5` and the MATLAB `K=5`, `Ktol=1e-12` settings.
