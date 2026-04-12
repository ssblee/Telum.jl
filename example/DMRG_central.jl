#using LurCGT
#using QSpaces
#using LinearAlgebra
include("MPO.jl")
include("DMRG_GS.jl")

# Find approximate ground state by using iterative diagonalization
# Currently cannot choose the final quantum number, 
# just return the resulting lowest energy state
function init_MPS(MPO::Vector{<:QSpace}, Nkeep::Int, Nkeep_last::Int=1; tol=0.0)
    N = length(MPO); MPS = Vector{QSpace{Float64, 3}}(undef, N)
    zq = zero_qlabels(MPO[1])
    Aprev = getvac(MPO[1], ("SLeft", "SLeft"))
    Hprev = addSingleton(Aprev, 3; itag="OLeft", dir='-')
    E, sp = nothing, nothing

    for i=1:N
        li = findleg(Aprev; itag=i==1 ? "SLeft" : "SB", dir='-')
        Anow = getIdentity((Aprev, li), (MPO[i], 2); itag="SL,$i")
        Hnow = Anow' * lock(Anow * Hprev * MPO[i]; itag="SL,$i")

        bli = findleg(Hnow; itag=i==N ? "ORight" : "OB,$i")
        Hmat = deleteSingleton(getsub(Hnow, bli, [(zq, 1)]), bli)
        e = eigen((Hmat + Hmat') / 2)

        ek, _ = discard_eigen(e, i==N ? Nkeep_last : Nkeep, i==N ? "SRight" : "SB,$i", "SD,$i"; tol)
        MPS[i] = Aprev = Anow * ek.V
        if i < N Hprev = lock(ek.V * Hnow; itag="SB") * ek.V'
        else E, sp = ek.eig_list[1][1], ek.eig_list[1][3] end
    end
    return MPS, E, sp
end

#MPO = MajumdarGhoshMPO(1.0, 40)
MPO = HubbardMPO(4.0, 1.5, 1.0, 40)
#MPO = XYMPO(1.0, 40)
#MPO = XXZMPO(0.3, 0.5, 40)

function do_dmrg(MPO, Nkeep_init=50, Nkeep_DMRG=50, Nsweep=4)
    MPS, E, sp = init_MPS(MPO, Nkeep_init; tol=0.0)
    println("Initial energy: $E")
    DMRG_GS_2site!(MPS, MPO, Nkeep_DMRG, Nsweep)
    return MPS
end

for Nkeep_DMRG = 20:10:70
    println("Nkeep_DMRG = $Nkeep_DMRG")
    @time do_dmrg(MPO, 50, 50, 4)
    GC.gc()
end