# ── Option types for getLocalSpace ──────────────────────────────────────────
#
# Each concrete option struct encodes the valid configuration for one class
# of physical local space.  The three classes mirror the three branches in
# the original QSpace Matlab code:
#
#   SpinOptions        – pure spin models (no charge degree of freedom)
#   FermionOptions     – spinless (single-component) fermions
#   FermionSOptions    – spinful (multi-component) fermions
#
# Keeping the options in separate types means that
#   • every (option-type, function) pair can be dispatched independently,
#   • invalid combinations are caught at construction time or via the type
#     system rather than as run-time errors, and
#   • users can add a new space type by defining a new option struct and the
#     two generic hooks getSymmetryOps / getIROPs for it (see getLocalSpace.jl).
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Abstract base ────────────────────────────────────────────────────────────

"""
    LocalSpaceOptions

Abstract supertype for all option structs passed to `getLocalSpace`.
"""
abstract type LocalSpaceOptions end

# ── Spin ─────────────────────────────────────────────────────────────────────

"""
    SpinOptions(; symmetry, spin)

Options for a pure spin-`spin` local space with `symmetry` ∈ {`SU{2}`, `U1`, …}.

# Fields
- `symmetry :: Type{<:Symmetry}` – symmetry group for the spin degree of freedom.
- `spin     :: Rational{Int}`    – spin quantum number (e.g. `1//2`, `1`, `3//2`).
"""
struct SpinOptions <: LocalSpaceOptions
    symmetry::Type{<:Symmetry}
    spin::Rational{Int}
end

# ── Spinless fermion ──────────────────────────────────────────────────────────

"""
    FermionOptions(; symmetry)

Options for a spinless (single-component) fermionic site.

# Fields
- `symmetry :: Type{<:Symmetry}` – symmetry group (typically `U1` for particle number).
"""
struct FermionOptions <: LocalSpaceOptions
    symmetry::Type{<:Symmetry}
end

# ── Spinful fermion ───────────────────────────────────────────────────────────

"""
    FermionSOptions(; charge_symm, spin_symm, channel_symm, nchannels)

Options for a spinful (multi-component) fermionic site.

# Fields
- `charge_symm  :: Type{<:Symmetry}` – symmetry for charge (typically `U1`).
- `spin_symm    :: Type{<:Symmetry}` – symmetry for spin   (typically `SU{2}`).
- `channel_symm :: Union{Type{<:Symmetry}, Nothing}` – symmetry for channel /
  flavor degree of freedom (e.g. `SU{3}` for 3-channel Kondo), or `nothing`.
- `nchannels    :: Int`              – number of channels / flavours (≥ 1).
"""
struct FermionSOptions <: LocalSpaceOptions
    charge_symm ::Union{Type{<:Symmetry}, Nothing}
    spin_symm   ::Union{Type{<:Symmetry}, Nothing}
    channel_symm::Union{Type{<:Symmetry}, Nothing}
    nchannels   ::Int
end
