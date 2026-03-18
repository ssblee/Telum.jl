get1jtensor(q::QSpace{T, QD, N, RD}, leg::Int) where {T, QD, N, RD} =
get1jtensor(leginfo(q, leg))

function _resolve_unique_leg(q::QSpace; dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             opname::AbstractString="operation")
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    length(legs) == 1 && return only(legs)
    isempty(legs) && throw(ArgumentError("$opname requires a uniquely specified leg, but no legs matched"))
    throw(ArgumentError("$opname requires a uniquely specified leg, but matched legs $legs"))
end

function _resolve_matching_legs(q::QSpace; dir=nothing, itag=nothing, plev=nothing,
                                lock=nothing, rev::Bool=false,
                                opname::AbstractString="operation")
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    isempty(legs) && throw(ArgumentError("$opname requires at least one matching leg, but no legs matched"))
    return legs
end

function _normalize_legflip_legs(q::QSpace{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("legflip requires at least one leg"))

    for leg in positions
        1 <= leg <= QD || throw(BoundsError(q, leg))
    end

    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "legflip legs must be unique, got $positions"))
    return positions
end

function get1jtensor(q::QSpace; dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false)
    leg = _resolve_unique_leg(q; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, opname="get1jtensor")
    return get1jtensor(q, leg)
end

function get1jtensor(leginfo::leginfo{N}) where N
    rows1 = row{Float64, 2, N, 2+N}[]

    symm, ind = leginfo.symm, leginfo.ind
    inds1 = (change_dir(ind), change_dir(change_green(ind)))
    dir1 = inds1[1].dir
    for (qlabels, RMTd) in leginfo.splist
        RMT1 = QTensor(reshape(Matrix{Float64}(I, RMTd, RMTd), RMTd, RMTd, (1 for _=1:N)...))
        dual_qlabels = Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)

        cgrs1 = CGR{2}[]
        for n in 1:N
            spdim = dimension(symm[n], qlabels[n])
            cgr_qs1 = sort((qlabels[n], dual_qlabels[n]))
            if qlabels[n] == dual_qlabels[n] cgp1 = (1, 2)
            elseif qlabels[n] < dual_qlabels[n] cgp1 = (1, 2)
            else cgp1 = (2, 1) end
            wmat1 = QTensor([sqrt(spdim);;])
            legdir1 = dir1 == '+' ? (2, 0) : (0, 2)

            push!(cgrs1, CGR(symm[n], cgr_qs1, wmat1, cgp1, legdir1))
        end

        push!(rows1, row(Tuple(cgrs1), RMT1))
    end

    # leg 1 = original space, leg 2 = dual space
    ET = eltype(leginfo.splist)
    dual_splist = sort!(ET[(Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N), RMTd) 
                           for (qlabels, RMTd) in leginfo.splist], by=x->x[1])
    spaces1 = (leginfo.splist, dual_splist)

    q1 = QSpace(symm, rows1, inds1, spaces1)
    return q1
end

function legflip(q::QSpace{T, QD, N, RD}, leg::Int) where {T, QD, N, RD}
    1 <= leg <= QD || throw(BoundsError(q, leg))
    j = get1jtensor(q, leg)
    q_flip = contract(q, (leg,), j, (1,); reduce_lock=false)
    perm = (ntuple(i -> i, leg - 1)..., QD, ntuple(i -> leg - 1 + i, QD - leg)...)
    return permutedims(q_flip, perm)
end

function legflip(q::QSpace{T, QD, N, RD}, legs::LegList) where {T, QD, N, RD}
    positions = _normalize_legflip_legs(q, legs)
    q_flip = q
    for leg in positions
        q_flip = legflip(q_flip, leg)
    end
    return q_flip
end

function legflip(q::QSpace; dir=nothing, itag=nothing, plev=nothing,
                 lock=nothing, rev::Bool=false)
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="legflip")
    return legflip(q, legs)
end
