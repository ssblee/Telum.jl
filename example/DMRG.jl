using LurCGT
using QSpaces

function build_MPO(qss::Matrix{<:QSpace}, N::Int)
    qss_ = complete_oplus_matrix(qss, (3, 4))
    MPO = Vector{QSpace}(undef, N)
    for i in 1:N
        tags = ("S,$i", "S,$i", i>1 ? "B,$(i-1)" : "Dummy,left", 
        i<N ? "B,$i" : "Dummy,right")

        if i == 1 MPO[i] = QSpace(oplus(qss_[4, :], 4), tags)
        elseif i == N MPO[i] = QSpace(oplus(qss_[:, 1], 3), tags)
        else MPO[i] = QSpace(oplus(qss_, (3, 4)), tags) end
    end
    return MPO
end

option = SpinOptions(SU{2}, 1//2)
q = getLocalSpace(option, ("site", "site", "op"))
J = 1

qss = Matrix{QSpace{Float64, 4, 1, 5}}(undef, 4, 4)

i4d = addSingleton(q.I, (3, 4); itags=("left", "right"), dirs=('+', '-'))
s4d = addSingleton(q.S, 3; itags="left", dirs='+')
s4d = settags(s4d, 4, "right")
sc4d = addSingleton(q.S', 4; itags="right", dirs='-')
sc4d = permutedims(settags(sc4d, 3, "left"), (2, 1, 3, 4))
opid = QSpace(getIdentity((q.S, 3)), ("left", "right"))

qss[1, 1] = qss[4, 4] = i4d
qss[2, 1] = sc4d
qss[3, 2] = q.I ⊗ opid
qss[4, 2] = J * s4d
qss[4, 3] = J/2 * s4d

MPO = build_MPO(qss, 20)