using Random

"""Select a single local charge/spin sector from the spinful identity."""
_unihole_local_identity(I, sector) =
    getsub(I, (1,2), q -> q == sector ? Colon() : nothing; preserve_space=true)

function _randomize_unihole_fusion(T, maxspin, rng)
    # MATLAB `FewData` keeps one multiplicity state per fused sector and
    # randomizes its reduced tensor data. First trim the newly fused bond.
    T = getsub(T, 1, _ -> 1)
    T = getsub(T, 1, q -> abs(q[2][1]) <= maxspin ? Colon() : nothing)
    T = Telum.materialize(T)
    for block in T.RMTs
        randn!(rng, block)
    end
    return T
end

function _seed_site_tags(T, site, L)
    lefttag = site == 1 ? "SLeft" : "SB,$(site-1)"
    righttag = site == L ? "SRight" : "SB,$site"
    T = setitag(T, 1, lefttag)
    T = setitag(T, 2, righttag)
    T = setitag(T, 3, "S,$site")
    return permutedims(T, (1,3,2))
end

function _seed_set_direction(T, axis, direction)
    indices = collect(Telum.inds(T))
    old = indices[axis]
    indices[axis] = Telum.TLIndex(old.itags, direction, old.plev, old.lock, old.dual)
    return TLArray(Telum.symm(T), T.qlabels, T.wmatdata, T.wmatinfo,
                   T.RMTs, Tuple(indices), Telum.spaces(T))
end

"""Port MATLAB `IniMPS_FermionS_unihole` for one-channel U(1)×SU(2).

Holes are placed at the same approximately uniform positions as MATLAB. Each
local site is restricted to empty or singly occupied states, the partial
fusion is spin-pruned, and the requested total `(Ntot-L, Stot)` sector is
selected at the end. Reduced tensor blocks are randomized as in MATLAB's
`FewData` helper.
"""
function initialize_hubbard_mps(p::HubbardDMRGParameters, H;
                                Nkeep::Int=100, tol::Real=0,
                                rng::AbstractRNG=Random.default_rng())
    validate(p)
    L = p.Lx*p.Ly
    0 <= p.Ntot <= L || throw(ArgumentError(
        "the MATLAB unihole initializer supports only 0 ≤ Ntot ≤ L"))
    0 <= p.Stot <= 2 || throw(ArgumentError(
        "the MATLAB SU(2) unihole initializer supports Stot ≤ 2"))
    length(H) == L || throw(DimensionMismatch("Hamiltonian MPO length does not match lattice"))

    # Keep the original function signature; initialization itself follows the
    # MATLAB sector construction and does not use an energy-based Nkeep cut.
    Nkeep > 0 || throw(ArgumentError("Nkeep must be positive"))
    tol >= 0 || throw(ArgumentError("tol must be nonnegative"))
    I = prepare_hubbard_operators(p).identity
    empty = ((-1,), (0,))
    single = ((0,), (1,))
    empty_op = _unihole_local_identity(I, empty)
    single_op = _unihole_local_identity(I, single)

    nhole = L - p.Ntot
    holes = if nhole == 0
        Set{Int}()
    else
        d = L / nhole
        Set(floor(Int, k*d - d/2 + 0.5) + 1 for k in 1:nhole)
    end
    maxspin = max(p.Stot, 1)
    MPS = Vector{TLArray{Float64,3}}(undef, L)
    T = getvac(I, ("seed", "seed"))

    for site in L:-1:2
        local_op = site in holes ? empty_op : single_op
        T = getIdentity((T,1), (local_op,2); itag="seed_fused")
        T = permutedims(T, (3,1,2))
        T = _randomize_unihole_fusion(T, maxspin, rng)
        MPS[site] = _seed_site_tags(T, site, L)
    end

    local_op = 1 in holes ? empty_op : single_op
    T = getIdentity((T,1), (local_op,2); itag="seed_fused")
    T = permutedims(T, (3,1,2))
    T = _randomize_unihole_fusion(T, maxspin, rng)
    target = ((p.Ntot-L,), (p.Stot,))
    T = getsub(T, 1, q -> q == target ? Colon() : nothing)
    isempty(Telum.spaces(T)[1]) && throw(ArgumentError(
        "unihole fusion did not generate requested sector $target"))
    boundary = getIdentity(T, 1; itag="SLeft")
    T = contract(boundary, 1, T, 1)
    T = _seed_set_direction(T, 1, '+')
    T = setitag(T, 1, "SLeft")
    T = setitag(T, 2, "SB,1")
    T = setitag(T, 3, "S,1")
    MPS[1] = permutedims(T, (1,3,2))

    return MPS, nothing, target
end

