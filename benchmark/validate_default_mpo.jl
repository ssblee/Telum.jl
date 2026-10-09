using Telum

include("hubbard_parameters.jl")
include("hubbard_geometry.jl")
include("hubbard_operators.jl")
include("hubbard_automaton.jl")
include("hubbard_mpo.jl")

p = HubbardDMRGParameters()
H = initialize_hubbard_hamiltonian(p;compress=false)
@assert length(H) == p.Lx*p.Ly == 32
maxbond = maximum(size(A,4) for A in H[1:end-1])
@assert maxbond <= 1000
println("Default 8×4 OBC/PBC ZigZag MPO built; max virtual dimension = $maxbond")
