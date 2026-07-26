"""
    empty_tlarray(symm::NTuple{N, Any}, inds::NTuple{QD, TLIndex}; T::Type=Float64) where {N, QD}

Create an empty rank-`QD` (zero-sector) TLArray over the given symmetries.

`symm` is an `N`-tuple of symmetry types (e.g. `(SU{2}, U1)`); `inds` is a
`QD`-tuple of `TLIndex` objects describing the leg directions, tags, and prime
levels.  All `TLIndex` entries with non-empty tags must be pairwise distinct.

The element type of future sector data defaults to `Float64`; pass `T=ComplexF64`
(or another concrete `<:Number` type) to use a different element type.
"""
function empty_tlarray(symm::NTuple{N, Any}, inds::NTuple{QD, TLIndex};
                      T::Type=Float64) where {N, QD}
    RD = QD + N
    QT = qlabeltype(symm)
    spaces = ntuple(_ -> Vector{Tuple{QT, Int}}(), QD)
    qlabels = NTuple{QD, QT}[]
    M = n_nonabelian_symmetries(productsymm(symm))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, RD}[]
    return TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs, inds, spaces)
end

function empty_tlarray(q::AbstractTLArray; T::Type=Float64)
    return empty_tlarray(symm(q), inds(q); T=T)
end

function Base.zero(q::AbstractTLArray{T, QD, N, RD}) where {T, QD, N, RD}
    QT = qlabeltype(q)
    qlabels = NTuple{QD, QT}[]
    M = n_nonabelian_symmetries(productsymm(q))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, RD}[]
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds(q), _copy_spaces_tuple(spaces(q)))
end

"""
    qlabeltype(symm::NTuple{N, Any}) where {N}
    qlabeltype(q::TLArray)

Return the qlabel type for one leg sector over the symmetries in `symm` or `q`.

For example, `(U1, SU{3})` returns `Tuple{Tuple{Int}, NTuple{2, Int}}`.
"""
function qlabeltype(symm::NTuple{N, Any}) where {N}
    return Tuple{ntuple(n -> NTuple{nzops(symm[n]), Int}, N)...}
end

qlabeltype(::Type{<:ProductSymm{Syms}}) where {Syms} =
    Tuple{(NTuple{nzops(S), Int} for S in Syms.parameters)...}

qlabeltype(::AbstractTLArray{T, QD, N, RD, QT}) where {T, QD, N, RD, QT} = QT

"""
    zero_qlabels(symm::NTuple{N, Any}) where {N}
    zero_qlabels(q::TLArray)

Return the trivial qlabel for each symmetry in `symm` or `q`.

For example, `(SU{2}, SU{3})` returns `((0,), (0, 0))`.
"""
function zero_qlabels(symm::NTuple{N, Any}) where {N}
    return ntuple(n -> Tuple(0 for _ in 1:nzops(symm[n])), N)
end

zero_qlabels(q::TLArray) = zero_qlabels(symm(q))

function _is_singleton_leg(q::TLArray{T, QD, N}, leg::Int) where {T, QD, N}
    1 <= leg <= QD || throw(ArgumentError("leg must lie in 1:$QD, got $leg"))
    return length(q.spaces[leg]) == 1 && only(q.spaces[leg]) == (zero_qlabels(q), 1)
end

_singleton_legs(q::TLArray{T, QD}) where {T, QD} = [leg for leg in 1:QD if _is_singleton_leg(q, leg)]

function _normalize_delete_singleton_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one deletion leg must be specified"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "singleton deletion legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "singleton deletion legs must be unique, got $positions"))
    sort!(positions)
    return positions
end


function _delete_singleton_rmt(rmt::AbstractArray{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    new_dims = ntuple(axis -> begin
        if axis <= qd - length(positions)
            old_axis = axis
            for pos in positions
                pos <= old_axis && (old_axis += 1)
            end
            size(rmt, old_axis)
        else
            size(rmt, axis + length(positions))
        end
    end, qd + n_symm - length(positions))
    return reshape(rmt, new_dims)
end

function _delete_singleton_rmt(rmt::DiagRMT{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    any(pos -> pos == rmt.axis[1] || pos == rmt.axis[2], positions) && throw(ArgumentError(
        "cannot preserve DiagRMT when deleting a diagonal axis"))
    shift_axis(axis) = axis - count(pos -> pos < axis, positions)
    return DiagRMT{T,RD - length(positions)}(rmt.diag, (shift_axis(rmt.axis[1]), shift_axis(rmt.axis[2])))
end

function _delete_singleton_impl(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, positions) where {T, QD, N, RD, QT, PS, M, RMT}
    new_qd = QD - length(positions)
    new_rd = RD - length(positions)

    keep_inds = [q.inds[leg] for leg in 1:QD if leg ∉ positions]
    keep_spaces = Vector{Tuple{QT, Int}}[q.spaces[leg] for leg in 1:QD if leg ∉ positions]

    qlabels = Matrix{QT}(undef, new_qd, sector_count(q))
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    RMTs = RMT <: DiagRMT ? Vector{DiagRMT{T, new_rd}}(undef, sector_count(q)) :
                             Vector{Array{T, new_rd}}(undef, sector_count(q))
    keep_legs = [leg for leg in 1:QD if leg ∉ positions]
    for sector_index in sector_slots(q)
        for (new_leg, old_leg) in enumerate(keep_legs)
            qlabels[new_leg, sector_index] = sector_qlabel(q, sector_index, old_leg)
        end
        q.iszero[sector_index] && continue
        RMTs[sector_index] = _delete_singleton_rmt(q.RMTs[sector_index], positions, QD, N)
    end

    return TLArray(product_symms(PS), qlabels, wmatdata, wmatinfo, RMTs, Tuple(keep_inds), Tuple(keep_spaces))
end

"""
    deleteSingleton(q::TLArray; dir=nothing, itag=nothing, plev=nothing) -> TLArray

Delete singleton legs matching the supplied criteria.

With no keyword arguments, all singleton legs are deleted.
Only singleton legs are eligible for deletion. If none match, a warning is
emitted and `q` is returned unchanged.
"""
function deleteSingleton(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing) where {T, QD}
    singleton_legs = if isnothing(dir) && isnothing(itag) && isnothing(plev)
        _singleton_legs(q)
    else
        candidate_legs = findlegs(q; dir=dir, itag=itag, plev=plev)
        [leg for leg in candidate_legs if _is_singleton_leg(q, leg)]
    end

    if isempty(singleton_legs)
        if isnothing(dir) && isnothing(itag) && isnothing(plev)
            @warn "deleteSingleton: no singleton legs found"
        else
            @warn "deleteSingleton: no singleton legs matched the requested criteria"
        end
        return q
    end
    return _delete_singleton_impl(q, singleton_legs)
end

"""
    deleteSingleton(q::TLArray, leg::Integer) -> TLArray
    deleteSingleton(q::TLArray, legs::LegList) -> TLArray

Delete the specified singleton legs from `q`.

Every selected leg must be singleton. Otherwise an `ArgumentError` is thrown.
"""
function deleteSingleton(q::TLArray, leg::Integer)
    return deleteSingleton(q, (leg,))
end

function deleteSingleton(q::TLArray{T, QD}, legs::LegList) where {T, QD}
    positions = _normalize_delete_singleton_legs(q, legs)
    bad = [leg for leg in positions if !_is_singleton_leg(q, leg)]
    isempty(bad) || throw(ArgumentError(
        "deleteSingleton requires singleton legs, but legs $bad are not singleton"))
    return _delete_singleton_impl(q, positions)
end

function _expand_singleton_kw(values, count::Int, name::AbstractString)
    if values isa AbstractString || values isa Char || values isa Integer
        return fill(values, count)
    end
    collected = collect(values)
    length(collected) == count || throw(ArgumentError(
        "$name must have length $count, got $(length(collected))"))
    return collected
end

function _singleton_insert_spec(q::AbstractTLArray{T, QD}, legs;
                                itag="", plev=0, lock=0, dir='+') where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one insertion leg must be specified"))

    count = length(positions)
    itag_vec = _expand_singleton_kw(itag, count, "itag")
    plev_vec = _expand_singleton_kw(plev, count, "plev")
    lock_vec = _expand_singleton_kw(lock, count, "lock")
    dir_vec  = _expand_singleton_kw(dir,  count, "dir")

    perm = sortperm(positions)
    positions = positions[perm]
    itag_vec = itag_vec[perm]
    plev_vec = plev_vec[perm]
    lock_vec = lock_vec[perm]
    dir_vec  = dir_vec[perm]

    final_qd = QD + count
    all(p -> 1 <= p <= final_qd, positions) || throw(ArgumentError(
        "singleton insertion legs must lie in 1:$final_qd, got $positions"))
    length(unique(positions)) == count || throw(ArgumentError(
        "singleton insertion legs must be unique, got $positions"))

    for direction in dir_vec
        direction in ('+', '-') || throw(ArgumentError(
            "added leg directions must be '+' or '-', got '$direction'"))
    end

    return positions, itag_vec, plev_vec, lock_vec, dir_vec
end

function _insert_singleton_rmt(rmt::AbstractArray{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    old_phys = size(rmt)[1:qd]
    om_dims = size(rmt)[qd+1:qd+n_symm]

    new_phys = Int[]
    sizehint!(new_phys, qd + length(positions))

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:(qd + length(positions))
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            push!(new_phys, 1)
            insert_idx += 1
        else
            push!(new_phys, old_phys[old_leg])
            old_leg += 1
        end
    end

    return reshape(rmt, Tuple(vcat(new_phys, collect(om_dims))))
end

function _insert_singleton_rmt(rmt::DiagRMT{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    shift_axis(axis) = axis + count(pos -> pos <= axis, positions)
    return DiagRMT{T,RD + length(positions)}(rmt.diag, (shift_axis(rmt.axis[1]), shift_axis(rmt.axis[2])))
end

"""
    addSingleton(q::TLArray; nlegs=1, itag="", plev=0, lock=0, dir='+')

Append one or more singleton trivial legs to `q`.

The `nlegs` keyword controls how many singleton legs are created. New legs are
added to the end of the leg list. `itag`, `plev`, `lock`, and `dir` may each be
either a scalar applied to every added leg or an iterable with one value per
inserted leg.
"""
function addSingleton(q::TLArray{T, QD}; nlegs::Integer=1,
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    count = Int(nlegs)
    count >= 1 || throw(ArgumentError("nlegs must be at least 1, got $nlegs"))
    positions = (QD + 1):(QD + count)
    return addSingleton(q, positions; itag=itag, plev=plev, lock=lock, dir=dir)
end

"""
    addSingleton(q::TLArray, legs; itag="", plev=0, lock=0, dir='+')

Insert one or more singleton trivial legs into `q`.

`legs` may be a single integer or any iterable of integers. Each value is the
position of an added leg in the output tensor, whose rank is `ndims(q) +
length(legs)`. The original legs keep their relative order.

`itag`, `plev`, `lock`, and `dir` may each be either a scalar applied to
every added leg or an iterable with one value per inserted leg.
"""
function addSingleton(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD, N, RD, QT, PS, M, RMT}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)

    new_qd = QD + length(positions)
    new_rd = RD + length(positions)
    trivial_qlabels = zero_qlabels(q)

    new_inds = Vector{TLIndex}(undef, new_qd)
    new_spaces = Vector{Vector{Tuple{QT, Int}}}(undef, new_qd)
    singleton_space = [(trivial_qlabels, 1)]

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            new_inds[new_leg] = TLIndex(String(itag_vec[insert_idx]), dir_vec[insert_idx],
                                       plev_vec[insert_idx], lock_vec[insert_idx])
            new_spaces[new_leg] = copy(singleton_space)
            insert_idx += 1
        else
            new_inds[new_leg] = q.inds[old_leg]
            new_spaces[new_leg] = q.spaces[old_leg]
            old_leg += 1
        end
    end

    qlabels = Matrix{QT}(undef, new_qd, sector_count(q))
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    RMTs = RMT <: DiagRMT ? Vector{DiagRMT{T, new_rd}}(undef, sector_count(q)) :
                             Vector{Array{T, new_rd}}(undef, sector_count(q))
    for sector_index in sector_slots(q)
        old_leg = 1
        insert_idx = 1
        for new_leg in 1:new_qd
            if insert_idx <= length(positions) && positions[insert_idx] == new_leg
                qlabels[new_leg, sector_index] = trivial_qlabels
                insert_idx += 1
            else
                qlabels[new_leg, sector_index] = sector_qlabel(q, sector_index, old_leg)
                old_leg += 1
            end
        end
        q.iszero[sector_index] && continue
        RMTs[sector_index] = _insert_singleton_rmt(q.RMTs[sector_index], positions, QD, N)
    end

    return TLArray(product_symms(PS), qlabels, wmatdata, wmatinfo, RMTs, Tuple(new_inds), Tuple(new_spaces))
end

function addSingleton(q::TLArrayView{T, QD}; nlegs::Integer=1,
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    count = Int(nlegs)
    count >= 1 || throw(ArgumentError("nlegs must be at least 1, got $nlegs"))
    positions = (QD + 1):(QD + count)
    return addSingleton(q, positions; itag=itag, plev=plev, lock=lock, dir=dir)
end

function addSingleton(q::TLArrayView{T, QD}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)
    internal_dir = q.conj ? [d == '+' ? '-' : '+' for d in dir_vec] : dir_vec
    new_arr = addSingleton(q.arr; nlegs=length(positions),
                           itag=itag_vec, plev=plev_vec, lock=lock_vec, dir=internal_dir)

    new_qd = QD + length(positions)
    new_perm = Vector{Int}(undef, new_qd)
    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            new_perm[new_leg] = QD + insert_idx
            insert_idx += 1
        else
            new_perm[new_leg] = q.perm[old_leg]
            old_leg += 1
        end
    end
    return _normalize_tlarray_view(new_arr, q.conj, q.scale, Tuple(new_perm))
end

addSingleton(q::TLArrayContraction; nlegs::Integer=1,
             itag="", plev=0, lock=0, dir='+') =
    addSingleton(to_concrete(q); nlegs=nlegs, itag=itag, plev=plev, lock=lock, dir=dir)

addSingleton(q::TLArrayContraction, legs; itag="", plev=0, lock=0, dir='+') =
    addSingleton(to_concrete(q), legs; itag=itag, plev=plev, lock=lock, dir=dir)

"""
    getvac(q::TLArray, itags::Tuple{Vararg{AbstractString, 2}}=("", "")) -> TLArray

Build the rank-2 vacuum TLArray associated with `q`.

The result keeps the same symmetry tuple as `q`, has one incoming leg and one
outgoing leg, and contains exactly one trivial sector with RMT dimension 1 on
each leg. If `itags` is provided, it is used as the tags of the two legs.
"""
function getvac(q::TLArray{T, QD, N, RD},
                itags::Tuple{Vararg{AbstractString, 2}}=("", "")) where {T, QD, N, RD}
    trivial_qlabels = zero_qlabels(q)
    space_entry = (trivial_qlabels, 1)

    qlabels = Matrix{qlabeltype(q)}(undef, 2, 1)
    qlabels[1, 1] = trivial_qlabels
    qlabels[2, 1] = trivial_qlabels
    wmatdata, wmatinfo = _unit_wmat_storage(productsymm(q), 1)
    rmt_data = fill(one(T), ntuple(_ -> 1, N + 2))
    RMTs = Array{T, N + 2}[rmt_data]
    inds = (TLIndex(itags[1], '+'), TLIndex(itags[2], '-'))
    QT = qlabeltype(q)
    space_template = Vector{Tuple{QT, Int}}([space_entry])
    spaces = (copy(space_template), copy(space_template))

    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, inds, spaces)
end

function ⊗(q1::TLArray{T1, QD1, N, RD1},
           q2::TLArray{T2, QD2, N, RD2}) where {T1, T2, QD1, QD2, N, RD1, RD2}
    @assert symm(q1) == symm(q2) "TLArray objects must share the same symmetry tuple"

    q1_ext = addSingleton(q1, QD1 + 1; dir='-')
    q2_ext = addSingleton(q2, 1; dir='+')
    return contract(q1_ext, (QD1 + 1,), q2_ext, (1,))
end

LinearAlgebra.kron(q1::TLArray, q2::TLArray) = q1 ⊗ q2
