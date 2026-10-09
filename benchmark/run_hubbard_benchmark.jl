using Telum

include("hubbard_parameters.jl")
include("hubbard_geometry.jl")
include("hubbard_operators.jl")
include("hubbard_automaton.jl")
include("hubbard_mpo.jl")
include("hubbard_mps.jl")
include("two_site_dmrg.jl")

"""Run the MATLAB-default ZigZag Hubbard schedule using the Telum DMRG code."""
function run_hubbard_benchmark(p::HubbardDMRGParameters=HubbardDMRGParameters();
                               Nkeep_init::Int=first(p.Nkeep),
                               Nsw::Int=length(p.Nkeep),
                               Nrep::Int=p.Nrep, compress::Bool=false,
                               time_blocks::Bool=false)
    validate(p)
    1 <= Nsw <= length(p.Nkeep) || throw(ArgumentError(
        "Nsw must select between 1 and $(length(p.Nkeep)) DStep values"))
    Nrep > 0 || throw(ArgumentError("Nrep must be positive"))

    ops = prepare_hubbard_operators(p)
    println("Prepared Hubbard operators")
    H = initialize_hubbard_hamiltonian(p,ops;compress)
    println("Prepared Hubbard Hamiltonian")
    psi, Einit, target = initialize_hubbard_mps(p,H;Nkeep=Nkeep_init)
    println("Prepared initial MPS with energy $Einit and target $target")

    # Run each selected DStep Nrep times, matching MATLAB's repelem(DStep,Nrep).
    schedule = repeat(p.Nkeep[1:Nsw]; inner=Nrep)
    psi, E = BenchmarkDMRG.DMRG_GS_2site!(psi,H,schedule,length(schedule);
        tol=p.Ktol,nKrylov=p.K,time_blocks)
    return (;energy=E, energy_per_site=E/(p.Lx*p.Ly), initial_energy=Einit,
            target, MPS=psi, MPO=H, parameters=p, schedule)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_hubbard_benchmark()
    println("Final total energy: $(result.energy)")
    println("Final energy per site: $(result.energy_per_site)")
end
