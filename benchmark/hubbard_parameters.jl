using Telum

"""Parameters matching the defaults in `Run_DMRG_Hubbard.m`."""
Base.@kwdef struct HubbardDMRGParameters
    Lx::Int = 8
    Ly::Int = 4
    BCX::Symbol = :OBC
    BCY::Symbol = :PBC
    path::Symbol = :ZigZag
    space_options::FermionSOptions = FermionSOptions(1, :U1, :SU2, nothing)
    t::Float64 = 1.0
    tp::Float64 = 0.0
    U::Float64 = 8.0
    Ntot::Int = 24
    Stot::Int = 0
    mu::Float64 = 0.0
    h::Float64 = 0.0
    CBE::Int = 0
    MPSSource::String = ""
    Nkeep::Vector{Int} = [200, 500, 1000, 2000, 4000, 10000]
    Nrep::Int = 5
    K::Int = 5
    Ktol::Float64 = 1e-12
    Nsw1::Int = 0
    Nsw::Int = 5
end

"""Validate the currently supported ZigZag lattice and symmetry sector."""
function validate(p::HubbardDMRGParameters)
    p.path === :ZigZag || throw(ArgumentError("only path=:ZigZag is currently ported"))
    p.BCX in (:OBC, :PBC) || throw(ArgumentError("BCX must be :OBC or :PBC"))
    p.BCY in (:OBC, :PBC) || throw(ArgumentError("BCY must be :OBC or :PBC"))
    p.Lx > 0 && p.Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    0 <= p.Ntot <= 2p.Lx*p.Ly || throw(ArgumentError("Ntot is outside the lattice capacity"))
    p.Nkeep[end] > 0 || throw(ArgumentError("Nkeep entries must be positive"))
    return p
end
