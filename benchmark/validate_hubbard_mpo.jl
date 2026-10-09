using LinearAlgebra
using Telum

include("hubbard_parameters.jl")
include("hubbard_geometry.jl")
include("hubbard_operators.jl")
include("hubbard_automaton.jl")
include("hubbard_mpo.jl")
include("hubbard_ed.jl")
include("two_site_dmrg.jl")
include("hubbard_mps.jl")

p = HubbardDMRGParameters(Lx=2,Ly=2,BCX=:OBC,BCY=:OBC,Ntot=4,Nsw=4)
graph = hubbard_interaction_map(p.Lx,p.Ly; BCX=p.BCX,BCY=p.BCY,t=p.t,tp=p.tp)
H = initialize_hubbard_hamiltonian(p;compress=false)
Eed, _ = hubbard_ed_ground_energy(p.Lx*p.Ly,graph.bonds,p.U,p.Ntot;mu=p.mu)
psi, _, target = initialize_hubbard_mps(p,H;Nkeep=100)
@assert target == ((0,),(0,))
psi, Edmrg = BenchmarkDMRG.DMRG_GS_2site!(psi,H,100,4;tol=p.Ktol,nKrylov=p.K)
println("2×2 OBC ED energy:   $Eed")
println("2×2 OBC DMRG energy: $Edmrg")
@assert isapprox(Edmrg,Eed;atol=1e-9)
