"""Build the spin-adapted Hamiltonian MPO for the converted Hubbard setup."""

function _hubbard_hopping_blocks(q)
    f4d = addSingleton(q.F, 3; itag="left", dir='+')
    f4d = setitag(f4d, 4, "right")

    fc_flip = addSingleton(legflip(q.F', 3), 3; itag="left", dir='+')
    fc_flip = setitag(permutedims(fc_flip, (2,1,3,4)), 4, "right")

    zf_flip = setitag(legflip(lock(q.Z, 1) * q.F, 3), 3, "left")
    fcz = permutedims(q.F' * lock(q.Z, 2), (1,3,2))
    fcz_end = setitag(addSingleton(fcz; itag="right", dir='-'),3,"left")
    zf_end = addSingleton(zf_flip,3; itag="left", dir='+')
    zf_end = permutedims(zf_end,(1,2,4,3))
    zf_end = setitag(setitag(zf_end,3,"left"),4,"right")
    zf_end = legflip(zf_end,4)
    return (; f4d, fc_flip, fcz=fcz_end, zf_flip=zf_end)
end

function _hubbard_identity4(q)
    return addSingleton(q.I; nlegs=2, itag=("left","right"), dir=('+','-'))
end

function _hubbard_pass4(q)
    fop_identity = TLArray(getIdentity((q.F,3)), ("left","right"))
    return q.Z ⊗ fop_identity
end

function _hubbard_pass4dag(q)
    fdop_identity = TLArray(getIdentity((q.F',3)), ("left","right"))
    fdop_identity = legflip(fdop_identity,1)
    return q.Z ⊗ fdop_identity
end

function _hubbard_onsite4(op)
    return addSingleton(op; nlegs=2, itag=("left","right"), dir=('+','-'))
end

"""Materialize scalar automaton channels as a chain of rank-four TLArray MPOs."""
function _materialize_hubbard_automaton(p, ops, graph)
    L = p.Lx*p.Ly
    plans = plan_hubbard_mpo(graph.bonds, L)
    # The two scalar states are followed by a pair of SU(2)-spinor channels
    # for each still-active start. Left and right channel lists can differ.
    state_f(active,i) = 2 + 2*(findfirst(==(i),active)-1)
    state_fdag(active,i) = state_f(active,i) + 1
    id4 = _hubbard_identity4(ops.q)
    pass4, pass4dag = _hubbard_pass4(ops.q), _hubbard_pass4dag(ops.q)
    hop = _hubbard_hopping_blocks(ops.q)
    onsite4 = _hubbard_onsite4(p.U*ops.doublon - p.mu*ops.number)

    mpo = Vector{TLArray{Float64,4}}(undef,L)
    zq = nothing
    for site in 1:L
        left_active, right_active = plans[site].active_before, plans[site].active_after
        blocks = Matrix{Any}(undef,2+2length(left_active),2+2length(right_active))
        fill!(blocks,nothing)
        blocks[1,1] = id4
        left_initial, right_initial = size(blocks,1), size(blocks,2)
        blocks[left_initial,right_initial] = id4
        # Onsite terms share the same initial-to-completed path.
        (!iszero(p.U) || !iszero(p.mu)) && (blocks[left_initial,1] = onsite4)

        # Start the two Hermitian-conjugate reduced hopping paths.
        for start in plans[site].starts
            blocks[left_initial,state_f(right_active,start)] = hop.f4d
            blocks[left_initial,state_fdag(right_active,start)] = hop.fc_flip
        end

        # Carry still-open starts across this site with the parity/irrep pass.
        for start in plans[site].passes
            blocks[state_f(left_active,start),state_f(right_active,start)] = pass4
            blocks[state_fdag(left_active,start),state_fdag(right_active,start)] = pass4dag
        end

        # Close each start at this endpoint. The interaction map already
        # stores +t/+tp; the Hamiltonian convention is -t/-tp.
        for (start,weight) in plans[site].ends
            coefficient = -weight
            blocks[state_f(left_active,start),1] = coefficient*hop.fcz
            blocks[state_fdag(left_active,start),1] = coefficient*hop.zf_flip
        end

        t = oplus(blocks,(3,4))
        zq === nothing && (zq = zero_qlabels(t))
        tags = ("S,$site", "S,$site", site > 1 ? "OB,$(site-1)" : "OLeft",
                site < L ? "OB,$site" : "ORight")
        if site == 1
            mpo[site] = TLArray(getsub(t,3,x -> x == zq ? -1 : nothing),tags)
        elseif site == L
            mpo[site] = TLArray(getsub(t,4,x -> x == zq ? 1 : nothing),tags)
        else
            mpo[site] = TLArray(t,tags)
        end
    end
    return mpo
end

"""Left-to-right SVD compression matching the MATLAB `stol`, `maxD` settings."""
function compress_hubbard_mpo!(MPO; stol::Real=1e-14, maxD::Int=1000)
    length(MPO) >= 2 || return MPO
    standardize(A,site,lefttag,righttag) = permutedims(A,(
        findleg(A;itag="S,$site",dir='+'),
        findleg(A;itag="S,$site",dir='-'),
        findleg(A;itag=lefttag),findleg(A;itag=righttag)))
    for site in 1:length(MPO)-1
        left_tag = site == 1 ? "OLeft" : "OB,$(site-1)"
        leftlegs = (findleg(MPO[site]; itag="S,$site",dir='+'),
                    findleg(MPO[site]; itag="S,$site",dir='-'),
                    findleg(MPO[site]; itag=left_tag))
        factors = svd(MPO[site],leftlegs,"OB,$site,left","OB,$site,right";
                      Nkeep=maxD,cutoff=stol)
        MPO[site] = standardize(removeitag(factors.U,"left"),site,left_tag,"OB,$site")
        carry = removeitag(factors.S*factors.Vd,"left")
        MPO[site+1] = carry*MPO[site+1]
        right_tag = site+1 == length(MPO) ? "ORight" : "OB,$(site+1)"
        MPO[site+1] = standardize(MPO[site+1],site+1,"OB,$site",right_tag)
    end
    return MPO
end

"""MATLAB `IniHamilMPO_HubbardSq` port for `Acharge,SU2spin`.

Returns only the Hamiltonian MPO; number, correlation, and measurement MPOs
remain out of scope. The uncompressed automaton is used by default; the
SVD utility can be enabled explicitly while its compatibility with Telum's
DMRG environment contractions is being finalized.
"""
function initialize_hubbard_hamiltonian(p::HubbardDMRGParameters,
                                         ops=prepare_hubbard_operators(p);
                                         compress::Bool=false, stol::Real=1e-14,
                                         maxD::Int=1000)
    validate(p)
    _hubbard_local_space(ops.options)
    graph = hubbard_interaction_map(p.Lx,p.Ly; BCX=p.BCX,BCY=p.BCY,t=p.t,tp=p.tp)
    mpo = _materialize_hubbard_automaton(p,ops,graph)
    return compress ? compress_hubbard_mpo!(mpo;stol,maxD) : mpo
end
