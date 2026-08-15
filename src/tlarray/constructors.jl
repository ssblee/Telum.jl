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

"""
    empty_tlarray(q::AbstractTLArray; T::Type=Float64) -> TLArray

Create an empty concrete `TLArray` with the same symmetry tuple and visible
indices as `q`.

`q` supplies only metadata: `symm(q)` determines the product symmetry and
`inds(q)` determines the output leg tags, directions, prime levels, and locks.
No sectors, w-matrices, or RMT payloads are copied. `T` chooses the element type
for future dense sector blocks.
"""
function empty_tlarray(q::AbstractTLArray; T::Type=Float64)
    return empty_tlarray(symm(q), inds(q); T=T)
end

"""
    zero(q::AbstractTLArray) -> TLArray

Return an empty concrete tensor with the same symmetry, indices, and cached
space lists as `q`.

The result has zero sectors and therefore no w-matrix or RMT payload storage.
Unlike `empty_tlarray(q)`, this preserves `spaces(q)` so the zero object can be
used in algorithms that need a compatible block-space template.
"""
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
    random_similar(a::TLArray) -> TLArray

Return a concrete tensor with the same logical indices, spaces, sector
q-labels, and w-matrices as `a`, with independently sampled RMT payloads.

Known-zero sectors remain zero and omitted, preserving the eager `TLArray`
sector-state invariant. The result owns independent RMT payloads while
retaining the source's embedded view state.
"""
function random_similar(a::TLArray)
    result = copy(a)
    for sector in sector_slots(result)
        is_sector_zero(result, sector) && continue
        rmt = result.RMTs[sector]
        result.RMTs[sector] = rmt isa DiagRMT ?
            DiagRMT(rand(eltype(rmt), length(rmt.diag)), rmt.axis) :
            rand(eltype(rmt), size(rmt))
    end
    return result
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

zero_qlabels(q::AbstractTLArray) = zero_qlabels(symm(q))

"""
    _is_singleton_leg(q::AbstractTLArray, leg::Int) -> Bool

Check whether visible leg `leg` is a trivial one-dimensional leg.

`q` supplies the space list and trivial qlabel for the product symmetry.
`leg` is validated against the visible tensor rank. A singleton leg is exactly
one cached space entry, and that entry must be `(zero_qlabels(q), 1)`.
"""
function _is_singleton_leg(q::AbstractTLArray{T, QD, N}, leg::Int) where {T, QD, N}
    1 <= leg <= QD || throw(ArgumentError("leg must lie in 1:$QD, got $leg"))
    leg_spaces = spaces(q)[leg]
    return length(leg_spaces) == 1 && only(leg_spaces) == (zero_qlabels(q), 1)
end

"""
    _singleton_legs(q::AbstractTLArray) -> Vector{Int}

Return all visible leg positions of `q` that are trivial one-dimensional
singleton legs.

The result is in ascending visible-leg order. Each candidate is checked through
`_is_singleton_leg`, so the definition is exactly "one space entry equal to
`(zero_qlabels(q), 1)`" rather than merely a leg whose dense dimension happens
to be one.
"""
_singleton_legs(q::AbstractTLArray{T, QD}) where {T, QD} = [leg for leg in 1:QD if _is_singleton_leg(q, leg)]

"""
    _normalize_delete_singleton_legs(q::AbstractTLArray, legs) -> Vector{Int}

Normalize user-selected deletion legs for `deleteSingleton`.

`legs` may be one integer or an iterable of integers. The result is sorted,
unique, non-empty, and bounds-checked against `ndims(q)`. This helper only
validates positions; the caller separately checks that each selected leg is
actually singleton.
"""
function _normalize_delete_singleton_legs(q::AbstractTLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one deletion leg must be specified"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "singleton deletion legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "singleton deletion legs must be unique, got $positions"))
    sort!(positions)
    return positions
end


"""
    _delete_singleton_rmt(rmt::AbstractArray, positions, qd::Int, n_symm::Int)

Remove singleton physical axes from one dense RMT block by reshaping.

`rmt` is stored with `qd` physical axes followed by `n_symm` w-matrix axes.
`positions` lists sorted physical axes to delete. Because each deleted axis is
known to have length one, the payload order is unchanged and a reshape is
sufficient; no numerical data is copied.
"""
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

"""
    _delete_singleton_rmt(rmt::DiagRMT, positions, qd::Int, n_symm::Int)

Remove singleton physical axes from a diagonal RMT block.

If `positions` intersects either diagonal physical axis, diagonal storage is no
longer valid and the block is densified before reshaping. Otherwise only the
stored diagonal axis numbers are shifted down by the number of deleted earlier
axes.
"""
function _delete_singleton_rmt(rmt::DiagRMT{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    if any(pos -> pos == rmt.axis[1] || pos == rmt.axis[2], positions)
        return _delete_singleton_rmt(Array{T, RD}(rmt), positions, qd, n_symm)
    end
    shift_axis(axis) = axis - count(pos -> pos < axis, positions)
    return DiagRMT{T,RD - length(positions)}(rmt.diag, (shift_axis(rmt.axis[1]), shift_axis(rmt.axis[2])))
end

"""
    _delete_singleton_preserves_diag(q::AbstractTLArray, positions) -> Bool

Return whether deleting `positions` can preserve diagonal RMT storage for `q`.

The check recomputes the diagonal-storage eligibility from `spaces(q)` and then
verifies that none of the two diagonal physical axes are removed. It is used
only to choose the output RMT container type; it does not inspect payloads.
"""
function _delete_singleton_preserves_diag(q::AbstractTLArray, positions)
    axes = _diag_rmt_axes_if_valid(spaces(q))
    isnothing(axes) && return false
    return !any(pos -> pos == axes[1] || pos == axes[2], positions)
end

"""
    _delete_singleton_rmt_type(::Type{RMT}, new_rd, q, positions) -> Type

Choose the concrete RMT container type after deleting singleton legs.

`RMT` is the source payload type, `new_rd` is the output RMT rank, `q` supplies
leg-space metadata used to decide diagonal eligibility, and `positions` are the
deleted physical axes. Dense array inputs remain dense. Diagonal inputs remain
diagonal only when the deletion preserves the two diagonal axes; otherwise the
result type is a dense `Array` because the diagonal representation would no
longer describe the reshaped payload.
"""
@inline _delete_singleton_rmt_type(::Type{Array{T, RD}}, new_rd::Int, q, positions) where {T, RD} =
    Array{T, new_rd}
@inline _delete_singleton_rmt_type(::Type{<:DiagRMT{T, RD}}, new_rd::Int, q, positions) where {T, RD} =
    _delete_singleton_preserves_diag(q, positions) ? DiagRMT{T, new_rd} : Array{T, new_rd}
@inline _delete_singleton_rmt_type(::Type{<:AbstractArray{T, RD}}, new_rd::Int, q, positions) where {T, RD} =
    Array{T, new_rd}

"""
    _insert_singleton_rmt_type(::Type{RMT}, new_rd) -> Type

Choose the concrete RMT container type after inserting singleton legs.

`RMT` is the source payload type and `new_rd` is the output RMT rank. Inserting
length-one physical axes preserves diagonal storage, so `DiagRMT` inputs remain
diagonal with shifted axes. Dense and generic abstract-array inputs use dense
`Array` storage.
"""
@inline _insert_singleton_rmt_type(::Type{Array{T, RD}}, new_rd::Int) where {T, RD} =
    Array{T, new_rd}
@inline _insert_singleton_rmt_type(::Type{<:DiagRMT{T, RD}}, new_rd::Int) where {T, RD} =
    DiagRMT{T, new_rd}
@inline _insert_singleton_rmt_type(::Type{<:AbstractArray{T, RD}}, new_rd::Int) where {T, RD} =
    Array{T, new_rd}

"""
    _delete_singleton_impl(q::TLArray, positions) -> TLArray

Materialize singleton-leg deletion for an identity-state concrete `TLArray`.

`positions` are sorted visible/source leg positions already validated as
singleton. The function keeps sector slot order, copies qlabels for surviving
legs, deep-copies w-matrix storage, and reshapes each nonzero RMT payload. Zero
sectors remain omitted from payload work, following the eager TLArray
sector-state invariant.
"""
function _delete_singleton_impl(q::TLArray{T, QD, N, RD, QT, PS, M, RMT}, positions) where {T, QD, N, RD, QT, PS, M, RMT}
    new_qd = QD - length(positions)
    new_rd = RD - length(positions)

    keep_inds = [q.inds[leg] for leg in 1:QD if leg ∉ positions]
    keep_spaces = Vector{Tuple{QT, Int}}[q.spaces[leg] for leg in 1:QD if leg ∉ positions]

    qlabels = Matrix{QT}(undef, new_qd, sector_count(q))
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    RMTout = _delete_singleton_rmt_type(RMT, new_rd, q, positions)
    RMTs = Vector{RMTout}(undef, sector_count(q))
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
    if !_is_identity_view_state(stored_conj(q), stored_scale(q), stored_perm(q))
        source_positions = sort!(Int[stored_perm(q)[leg] for leg in positions])
        new_arr = deleteSingleton(_storage_identity_tlarray(q), source_positions)
        deleted_visible = Set(positions)

        new_perm = Int[]
        sizehint!(new_perm, QD - length(positions))
        for old_visible_leg in 1:QD
            old_visible_leg in deleted_visible && continue
            old_source_leg = stored_perm(q)[old_visible_leg]
            push!(new_perm, old_source_leg - count(pos -> pos < old_source_leg, source_positions))
        end
        return _with_view_state(new_arr, stored_conj(q), stored_scale(q), Tuple(new_perm))
    end
    return _delete_singleton_impl(q, positions)
end

"""
    deleteSingleton(q::AbstractTLArray; dir=nothing, itag=nothing, plev=nothing)

Delete singleton legs from a lazy or abstract tensor without forcing unrelated
sector payloads.

Keyword selectors are interpreted on visible indices. With no selectors, all
singleton legs are selected. The result is a lazy singleton wrapper for
non-`TLArray` sources, preserving source sector slot numbering and computing
RMTs only when requested.
"""
function deleteSingleton(q::AbstractTLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing) where {T, QD}
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
    return deleteSingleton(q, singleton_legs)
end

"""
    deleteSingleton(q::AbstractTLArray, leg::Integer)
    deleteSingleton(q::AbstractTLArray, legs::LegList)

Delete explicitly selected singleton visible legs from `q`.

`leg` or `legs` select output-visible positions, not stored payload axes when
`q` carries permutation state. Every selected leg must have only the trivial
one-dimensional space. For lazy sources, the returned wrapper records
source/result leg maps and applies the reshape when a sector is computed.
"""
function deleteSingleton(q::AbstractTLArray, leg::Integer)
    return deleteSingleton(q, (leg,))
end

function deleteSingleton(q::AbstractTLArray{T, QD}, legs::LegList) where {T, QD}
    positions = _normalize_delete_singleton_legs(q, legs)
    bad = [leg for leg in positions if !_is_singleton_leg(q, leg)]
    isempty(bad) || throw(ArgumentError(
        "deleteSingleton requires singleton legs, but legs $bad are not singleton"))
    return _delete_singleton_lazy(q, positions)
end

"""
    _expand_singleton_kw(values, count::Int, name::AbstractString)

Expand one singleton-insertion keyword to one value per inserted leg.

`values` may be a scalar string, character, or integer, in which case it is
repeated `count` times. Otherwise it is collected and must already have length
`count`. `name` is used only to produce a precise `ArgumentError`.
"""
function _expand_singleton_kw(values, count::Int, name::AbstractString)
    if values isa AbstractString || values isa Char || values isa Integer
        return fill(values, count)
    end
    collected = collect(values)
    length(collected) == count || throw(ArgumentError(
        "$name must have length $count, got $(length(collected))"))
    return collected
end

"""
    _singleton_insert_spec(q::AbstractTLArray, legs; itag="", plev=0, lock=0, dir='+')

Normalize the insertion specification for `addSingleton`.

`legs` gives final output positions for the new singleton legs. `itag`, `plev`,
`lock`, and `dir` may be scalars or one value per inserted leg; all are sorted
to match the increasing output positions. Returned values are
`positions, itag_vec, plev_vec, lock_vec, dir_vec`. The direction values are
validated as `'+'` or `'-'`.
"""
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

"""
    _insert_singleton_rmt(rmt::AbstractArray, positions, qd::Int, n_symm::Int)

Insert singleton physical axes into one dense RMT block by reshaping.

`positions` are final physical-axis positions in the output RMT. The original
`qd` physical axes keep their relative order, and the trailing `n_symm`
w-matrix axes stay after all physical axes. Inserted axes have dimension one,
so no payload values are duplicated.
"""
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

"""
    _insert_singleton_rmt(rmt::DiagRMT, positions, qd::Int, n_symm::Int)

Insert singleton physical axes into a diagonal RMT descriptor.

The diagonal payload vector is reused, while each stored diagonal axis is
shifted up by the number of inserted physical axes at or before it. This keeps
diagonal storage valid because singleton axes do not change the paired
nontrivial dimensions.
"""
function _insert_singleton_rmt(rmt::DiagRMT{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    shift_axis(axis) = axis + count(pos -> pos <= axis, positions)
    return DiagRMT{T,RD + length(positions)}(rmt.diag, (shift_axis(rmt.axis[1]), shift_axis(rmt.axis[2])))
end

"""
    _storage_identity_tlarray(q::TLArray) -> TLArray

Return a concrete TLArray view of `q`'s owned storage with identity state.

The result aliases qlabel, w-matrix, RMT, `isdefined`, and `iszero` storage and
uses `stored_inds(q)`/`stored_spaces(q)`. It is a helper for structural
operations that must edit stored-leg metadata first and then reattach the
original lazy view state.
"""
function _storage_identity_tlarray(q::TLArray)
    return TLArray(Val(:alias_storage), symm(q), stored_qlabels(q), q.wmatdata, q.wmatinfo,
                   q.RMTs, q.isdefined, q.iszero, stored_inds(q), stored_spaces(q))
end

"""
    _add_singleton_leg_maps(qd::Int, positions)

Build source/result leg maps for singleton insertion.

`qd` is the source visible rank and `positions` are final positions of inserted
legs. Returns `(source_to_result, result_to_source)`. Inserted legs are encoded
as `0` in `result_to_source`; all original legs keep their relative order.
"""
function _add_singleton_leg_maps(qd::Int, positions)
    new_qd = qd + length(positions)
    source_to_result = Vector{Int}(undef, qd)
    result_to_source = zeros(Int, new_qd)
    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            insert_idx += 1
        else
            source_to_result[old_leg] = new_leg
            result_to_source[new_leg] = old_leg
            old_leg += 1
        end
    end
    return source_to_result, result_to_source
end

"""
    _delete_singleton_leg_maps(qd::Int, positions)

Build source/result leg maps for singleton deletion.

`qd` is the source visible rank and `positions` are source positions being
deleted. Returns `(source_to_result, result_to_source)`. Deleted legs are
encoded as `0` in `source_to_result`; surviving legs keep their relative order.
"""
function _delete_singleton_leg_maps(qd::Int, positions)
    new_qd = qd - length(positions)
    source_to_result = zeros(Int, qd)
    result_to_source = Vector{Int}(undef, new_qd)
    new_leg = 1
    delete_idx = 1
    for old_leg in 1:qd
        if delete_idx <= length(positions) && positions[delete_idx] == old_leg
            delete_idx += 1
        else
            source_to_result[old_leg] = new_leg
            result_to_source[new_leg] = old_leg
            new_leg += 1
        end
    end
    return source_to_result, result_to_source
end

"""
    _add_singleton_lazy(q, positions, itag_vec, plev_vec, lock_vec, dir_vec)

Construct a lazy singleton-insertion wrapper around `q`.

`positions` and the per-leg metadata vectors are normalized by
`_singleton_insert_spec`. The wrapper precomputes output indices, spaces, sector
qlabels, and source/result leg maps, but it does not reshape RMT payloads until
`compute_sectors` requests a sector.
"""
function _add_singleton_lazy(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}, positions,
                             itag_vec, plev_vec, lock_vec, dir_vec) where {T, QD, N, RD, QT, PS, M, RMT}
    new_qd = QD + length(positions)
    new_rd = RD + length(positions)
    trivial_qlabels = zero_qlabels(q)
    source_to_result, result_to_source = _add_singleton_leg_maps(QD, positions)

    source_inds = inds(q)
    source_spaces = spaces(q)
    new_inds = Vector{TLIndex}(undef, new_qd)
    new_spaces = Vector{Vector{Tuple{QT, Int}}}(undef, new_qd)
    singleton_space = [(trivial_qlabels, 1)]
    insert_idx = 1
    for new_leg in 1:new_qd
        source_leg = result_to_source[new_leg]
        if source_leg == 0
            new_inds[new_leg] = TLIndex(String(itag_vec[insert_idx]), dir_vec[insert_idx],
                                       plev_vec[insert_idx], lock_vec[insert_idx])
            new_spaces[new_leg] = copy(singleton_space)
            insert_idx += 1
        else
            new_inds[new_leg] = source_inds[source_leg]
            new_spaces[new_leg] = source_spaces[source_leg]
        end
    end

    qlabels = Vector{NTuple{new_qd, QT}}(undef, sector_count(q))
    for sector in sector_slots(q)
        qlabels[sector] = ntuple(Val(new_qd)) do new_leg
            source_leg = result_to_source[new_leg]
            source_leg == 0 ? trivial_qlabels : sector_qlabel(QT, q, sector, source_leg)
        end
    end

    RMTout = _insert_singleton_rmt_type(RMT, new_rd)
    return AddSingletonTLArray{T, new_qd, N, new_rd, QT, PS, M, RMTout}(
        qlabels, Tuple(new_inds), Tuple(new_spaces), q, positions,
        source_to_result, result_to_source)
end

"""
    _delete_singleton_lazy(q, positions) -> DeleteSingletonTLArray

Construct a lazy singleton-deletion wrapper around `q`.

`positions` are visible source legs already validated as singleton. When `q`
has non-identity stored view state, deletion is translated to stored-leg
positions, the state is detached from the source, and the adjusted state is
stored on the wrapper. This preserves source sector slot numbering and avoids
canonicalizing unrelated payloads.
"""
function _delete_singleton_lazy(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}, positions) where {T, QD, N, RD, QT, PS, M, RMT}
    if !_is_identity_view_state(stored_conj(q), stored_scale(q), stored_perm(q))
        source_positions = sort!(Int[stored_perm(q)[leg] for leg in positions])
        new_qd = QD - length(positions)
        new_rd = RD - length(positions)
        source_to_result, result_to_source = _delete_singleton_leg_maps(QD, source_positions)

        new_inds = [stored_inds(q)[old_leg] for old_leg in result_to_source]
        new_spaces = [stored_spaces(q)[old_leg] for old_leg in result_to_source]
        qlabels = Vector{NTuple{new_qd, QT}}(undef, sector_count(q))
        for sector in sector_slots(q)
            qlabels[sector] = ntuple(new_leg -> stored_sector_qlabel(QT, q, sector, result_to_source[new_leg]),
                                     Val(new_qd))
        end

        deleted_visible = Set(positions)
        new_perm = Int[]
        sizehint!(new_perm, new_qd)
        for old_visible_leg in 1:QD
            old_visible_leg in deleted_visible && continue
            old_source_leg = stored_perm(q)[old_visible_leg]
            push!(new_perm, old_source_leg - count(pos -> pos < old_source_leg, source_positions))
        end

        TR = eltype(RMT)
        source_arr = _with_view_state(q, false, one(T), _identity_phys_perm(Val(QD)))
        return DeleteSingletonTLArray{T, new_qd, N, new_rd, QT, PS, M, Array{TR, new_rd}}(
            qlabels, Tuple(new_inds), Tuple(new_spaces),
            stored_conj(q), stored_scale(q), Tuple(new_perm),
            source_arr, source_positions, source_to_result, result_to_source)
    end

    new_qd = QD - length(positions)
    new_rd = RD - length(positions)
    source_to_result, result_to_source = _delete_singleton_leg_maps(QD, positions)

    source_inds = inds(q)
    source_spaces = spaces(q)
    new_inds = [source_inds[old_leg] for old_leg in result_to_source]
    new_spaces = [source_spaces[old_leg] for old_leg in result_to_source]

    qlabels = Vector{NTuple{new_qd, QT}}(undef, sector_count(q))
    for sector in sector_slots(q)
        qlabels[sector] = ntuple(new_leg -> sector_qlabel(QT, q, sector, result_to_source[new_leg]), Val(new_qd))
    end

    RMTout = _delete_singleton_rmt_type(RMT, new_rd, q, positions)
    return DeleteSingletonTLArray{T, new_qd, N, new_rd, QT, PS, M, RMTout}(
        qlabels, Tuple(new_inds), Tuple(new_spaces), q, positions,
        source_to_result, result_to_source)
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
    if !_is_identity_view_state(stored_conj(q), stored_scale(q), stored_perm(q))
        positions, itag_vec, plev_vec, lock_vec, dir_vec =
            _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)
        internal_dir = stored_conj(q) ? [d == '+' ? '-' : '+' for d in dir_vec] : dir_vec
        new_arr = addSingleton(_storage_identity_tlarray(q); nlegs=length(positions),
                               itag=itag_vec, plev=plev_vec, lock=lock_vec,
                               dir=internal_dir)
        new_qd = QD + length(positions)
        new_perm = Vector{Int}(undef, new_qd)
        old_leg = 1
        insert_idx = 1
        for new_leg in 1:new_qd
            if insert_idx <= length(positions) && positions[insert_idx] == new_leg
                new_perm[new_leg] = QD + insert_idx
                insert_idx += 1
            else
                new_perm[new_leg] = stored_perm(q)[old_leg]
                old_leg += 1
            end
        end
        return _with_view_state(new_arr, stored_conj(q), stored_scale(q), Tuple(new_perm))
    end

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
    RMTout = _insert_singleton_rmt_type(RMT, new_rd)
    RMTs = Vector{RMTout}(undef, sector_count(q))
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

"""
    addSingleton(q::AbstractTLArray; nlegs=1, itag="", plev=0, lock=0, dir='+')

Append singleton trivial legs to a lazy or abstract tensor.

`nlegs` is the number of legs appended after the current visible rank.
`itag`, `plev`, `lock`, and `dir` may be scalars or iterables with one entry per
new leg. The returned lazy wrapper records the new leg metadata and reshapes
RMT sectors only when materialized.
"""
function addSingleton(q::AbstractTLArray{T, QD}; nlegs::Integer=1,
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    count = Int(nlegs)
    count >= 1 || throw(ArgumentError("nlegs must be at least 1, got $nlegs"))
    positions = (QD + 1):(QD + count)
    return addSingleton(q, positions; itag=itag, plev=plev, lock=lock, dir=dir)
end

"""
    addSingleton(q::AbstractTLArray, legs; itag="", plev=0, lock=0, dir='+')

Insert singleton trivial legs into explicit output positions of `q`.

`legs` may be one integer or an iterable. Each value names a leg position in
the result rank `ndims(q) + length(legs)`. Existing legs retain relative order.
Metadata keywords follow the same scalar-or-per-leg expansion as the concrete
`TLArray` method.
"""
function addSingleton(q::AbstractTLArray{T, QD, N, RD, QT, PS, M, RMT}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD, N, RD, QT, PS, M, RMT}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)
    return _add_singleton_lazy(q, positions, itag_vec, plev_vec, lock_vec, dir_vec)
end

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

"""
    ⊗(q1::TLArray, q2::TLArray) -> TLArray

Tensor product of two `TLArray` values with the same symmetry tuple.

`q1` and `q2` supply the two factors. The implementation inserts a temporary
outgoing singleton leg on `q1` and incoming singleton leg on `q2`, then
contracts those trivial legs. This reuses the ordinary contraction path so
qlabel ordering, CGT ordering, and RMT assembly follow the same invariants as
other tensor contractions.
"""
function ⊗(q1::TLArray{T1, QD1, N, RD1},
           q2::TLArray{T2, QD2, N, RD2}) where {T1, T2, QD1, QD2, N, RD1, RD2}
    @assert symm(q1) == symm(q2) "TLArray objects must share the same symmetry tuple"

    q1_ext = addSingleton(q1, QD1 + 1; dir='-')
    q2_ext = addSingleton(q2, 1; dir='+')
    return contract(q1_ext, (QD1 + 1,), q2_ext, (1,))
end

LinearAlgebra.kron(q1::TLArray, q2::TLArray) = q1 ⊗ q2
