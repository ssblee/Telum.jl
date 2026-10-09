# Standalone reference check; run with `julia benchmark/validate_hubbard_ed.jl`.
using LinearAlgebra
include("hubbard_parameters.jl")
include("hubbard_geometry.jl")
include("hubbard_ed.jl")
include("hubbard_automaton.jl")

p = validate(HubbardDMRGParameters(Lx=2, Ly=2, BCX=:OBC, BCY=:OBC, Ntot=4))
graph = hubbard_interaction_map(p.Lx, p.Ly; BCX=p.BCX, BCY=p.BCY, t=p.t, tp=p.tp)
@assert graph.order == [1 3; 2 4]
@assert length(graph.bonds) == 4
channels = plan_hubbard_mpo(graph.bonds, p.Lx*p.Ly)
@assert length(channels) == 4
@assert channels[1].starts == [1]
@assert channels[1].active_after == [1]
@assert channels[2].ends == [(1,1.0)]
@assert channels[2].passes == [1]
@assert channels[3].ends == [(1,1.0)]
@assert channels[3].passes == [2]
@assert channels[4].ends == [(2,1.0), (3,1.0)]
@assert channels[4].active_after == Int[]
defaults = validate(HubbardDMRGParameters())
default_graph = hubbard_interaction_map(defaults.Lx, defaults.Ly;
    BCX=defaults.BCX, BCY=defaults.BCY, t=defaults.t, tp=defaults.tp)
@assert defaults.Nkeep == [200,500,1000,2000,4000,10000]
@assert length(default_graph.bonds) == 60
E0, basis = hubbard_ed_ground_energy(p.Lx*p.Ly, graph.bonds, p.U, p.Ntot)
@assert length(basis) == binomial(2p.Lx*p.Ly, p.Ntot)
@assert isapprox(E0, -1.320234958271926; atol=1e-12)
println("2×2 OBC, Ntot=4 Hubbard ED reference: E0 = $E0")
