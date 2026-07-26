getsub(q::TLArray, selector) = TLArray(q, selector)
getsub(q::TLArrayContraction, args...; kwargs...) =
    getsub(to_concrete(q), args...; kwargs...)

function _normalize_getsub_index(i::Int, dim::Int, sector, leg::Int)
    i == 0 && throw(ArgumentError(
        "selector for sector $sector on leg $leg cannot contain 0"))
    return i < 0 ? dim + i + 1 : i
end

function _normalize_getsub_indices(raw, dim::Int, sector, leg::Int)
    inds = if raw isa Integer
        [Int(raw)]
    elseif raw isa Tuple && all(i -> i isa Integer, raw)
        Int[Int(i) for i in raw]
    elseif raw isa AbstractRange{<:Integer}
        Int[i for i in raw]
    elseif raw isa AbstractVector{<:Integer}
        Int[i for i in raw]
    else
        throw(ArgumentError(
            "selector for sector $sector on leg $leg must be :, Int, integer tuple, AbstractRange, or AbstractVector{<:Integer}"))
    end

    isempty(inds) && throw(ArgumentError(
        "selector for sector $sector on leg $leg must not be empty"))
    inds = [_normalize_getsub_index(i, dim, sector, leg) for i in inds]
    length(unique(inds)) == length(inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains duplicate indices"))
    all(1 <= i <= dim for i in inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains out-of-bounds indices for dimension $dim"))
    return inds
end
function _normalize_getsub_predicate_pick(raw, dim::Int, sector, leg::Int)
    raw isa Bool && throw(ArgumentError("getsub predicate for sector $sector on leg $leg must not return Bool; use Colon() or nothing explicitly"))
    raw === nothing && return nothing
    raw isa Colon && return Colon()
    return _normalize_getsub_indices(raw, dim, sector, leg)
end

function _collect_getsub_predicate_picks(q::TLArray, positions, pred::Function)
    selected_picks = Dict{Int, Dict{Any, Any}}()
    for leg in positions
        picks = Dict{Any, Any}()
        for (sector, dim) in q.spaces[leg]
            pick = _normalize_getsub_predicate_pick(pred(sector), dim, sector, leg)
            isnothing(pick) && continue
            picks[sector] = pick
        end
        selected_picks[leg] = picks
    end
    return selected_picks
end

function _apply_getsub_picks(q::TLArray{T, QD, N, RD, QT},
                             positions,
                             selected_picks::Dict{Int, Dict{Any, Any}};
                             preserve_space::Bool=false) where {T, QD, N, RD, QT}
    if preserve_space
        for leg in positions
            for (sector, pick) in selected_picks[leg]
                pick isa Colon && continue
                throw(ArgumentError(
                    "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
            end
        end
    end

    selected_leg_set = Set(positions)
    qlabels_out = NTuple{QD, QT}[]
    source_wmat_sectors = Int[]
    RMTs_out = Array{T, RD}[]
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        picks_by_leg = Dict{Int, Any}()
        keep = true
        for leg in positions
            sector = sector_qlabel(q, sector_index, leg)
            picks = selected_picks[leg]
            haskey(picks, sector) || (keep = false; break)
            picks_by_leg[leg] = picks[sector]
        end
        keep || continue
        selectors = ntuple(d -> get(picks_by_leg, d, Colon()), RD)
        selected_rmt = q.RMTs[sector_index][selectors...]
        iszero(sum(abs2, selected_rmt)) && continue
        push!(qlabels_out, ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD)))
        push!(source_wmat_sectors, sector_index)
        push!(RMTs_out, selected_rmt)
    end

    spaces_out = if preserve_space
        _copy_spaces_tuple(q.spaces)
    else
        ntuple(l -> begin
            if l in selected_leg_set
                out = eltype(q.spaces[l])[]
                picks = selected_picks[l]
                for (sector, dim) in q.spaces[l]
                    haskey(picks, sector) || continue
                    pick = picks[sector]
                    push!(out, (sector, pick isa Colon ? dim : length(pick)))
                end
                out
            else
                copy(q.spaces[l])
            end
        end, QD)
    end

    wmatdata, wmatinfo = _copy_wmat_storage(q, source_wmat_sectors; deep=true)
    return TLArray(symm(q), qlabels_out, wmatdata, wmatinfo, RMTs_out, q.inds, spaces_out)
end

function _normalize_getsub_predicate_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("getsub requires at least one leg"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "getsub legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "getsub legs must be unique, got $positions"))
    return positions
end

"""
    getsub(q::TLArray, leg::Integer, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sector on `leg` satisfies
`pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), only `q.spaces[leg]` is truncated to
the retained sectors and all other leg-space lists are copied unchanged. If
`preserve_space=true`, all cached leg-space lists are preserved exactly and only
the sectors are filtered. This requires `pred` to keep whole sectors, so any
index-selection return value is rejected when `preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, leg::Integer, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    return getsub(q, (Int(leg),), pred; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, legs, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sectors on every selected leg
satisfy `pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), each selected leg keeps only the
matching entries in its space list, while unselected legs keep copies of their
original space lists. If `preserve_space=true`, all cached leg-space lists are
preserved exactly and only the sectors are filtered. This requires `pred` to keep
whole sectors, so any index-selection return value is rejected when
`preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, legs::LegList, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    positions = _normalize_getsub_predicate_legs(q, legs)
    selected_picks = _collect_getsub_predicate_picks(q, positions, pred)
    return _apply_getsub_picks(q, positions, selected_picks; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, pred::Function; preserve_space::Bool=false, dir=nothing,
           itag=nothing, plev=nothing, lock=nothing, rev=false) -> TLArray

Apply predicate-based `getsub` to every leg selected by the keyword criteria.
The leg selection follows the same matching rules as `findlegs`.
"""
function getsub(q::TLArray{T, QD, N, RD}, pred::Function; preserve_space::Bool=false,
                dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                rev::Bool=false) where {T, QD, N, RD}
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="getsub")
    return getsub(q, legs, pred; preserve_space=preserve_space)
end
