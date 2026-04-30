using Revise
using LurCGT
using QSpaces
using LinearAlgebra

include("DMRG_central.jl")

function benchmark_smallRMT()
    option = FermionSOptions(U1, SU{2}, SU{3}, 3)
    q = getLocalSpace(option);
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    qf = QSpace(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2));
    return qf, a
end

function benchmark_DMRGres(MPS, MPO, Hrl)
    Hl = Hrl[21]
    Hr = Hrl[23]'
    M = MPS[21] * MPS[22]
    H1, H2 = MPO[21], MPO[22]

    return Hl, Hr, M, H1, H2
end

function get_DMRGres(Nkeep=50)
    MPO = HubbardMPO(8.0, 1.5, 1.0, 40)
    MPS, E, sp = init_MPS(MPO, Nkeep; tol=0.0)
    DMRG_GS_2site!(MPS, MPO, Nkeep, 1)
    Hrl = getHrl(MPO, MPS)

    return MPS, MPO, Hrl
end
