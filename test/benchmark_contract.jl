using Revise
using LurCGT
using QSpaces
using LinearAlgebra


function benchmark_smallRMT()
    option = FermionSOptions(U1, SU{2}, SU{3}, 3)
    q = getLocalSpace(option);
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    qf = QSpace(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2));
    return qf, a
end

