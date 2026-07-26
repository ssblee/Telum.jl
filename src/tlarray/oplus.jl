function _normalize_oplus_dims(dimensions, QD::Int; sort_dims::Bool=true)
    dims = dimensions isa Integer ? (Int(dimensions),) : Tuple(Int(d) for d in dimensions)
    isempty(dims) && throw(ArgumentError("oplus requires at least one dimension"))
    all(d -> 1 <= d <= QD, dims) || throw(ArgumentError(
        "oplus dimensions must lie in 1:$QD, got $(collect(dims))"))
    length(unique(dims)) == length(dims) || throw(ArgumentError(
        "oplus dimensions must be unique, got $(collect(dims))"))
    return sort_dims ? Tuple(sort(collect(dims))) : dims
end

function _normalize_oplus_matrix_dims(dimensions, QD::Int)
    dimensions isa Tuple && length(dimensions) == 2 || throw(ArgumentError(
        "matrix oplus requires exactly two axis-dimension specifications"))

    dims1 = _normalize_oplus_dims(dimensions[1], QD)
    dims2 = _normalize_oplus_dims(dimensions[2], QD)
    isempty(intersect(dims1, dims2)) || throw(ArgumentError(
        "matrix-axis oplus legs must be disjoint, got $(collect(dims1)) and $(collect(dims2))"))
    return dims1, dims2
end

function _splist_dim_map(splist::Vector)
    dims = Dict{Any, Int}()
    for (qlabels, dim) in splist
        if haskey(dims, qlabels)
            throw(ArgumentError("duplicate qlabel sector encountered in space list: $qlabels"))
        end
        dims[qlabels] = dim
    end
    return dims
end

function _sum_splists_many(splists)
    isempty(splists) && throw(ArgumentError("cannot sum an empty collection of space lists"))

    dims = Dict{Any, Int}()
    seen = Set{Any}()
    result = copy(first(splists))
    empty!(result)

    for splist in splists
        for (qlabels, dim) in splist
            dims[qlabels] = get(dims, qlabels, 0) + dim
            if qlabels ∉ seen
                push!(result, (qlabels, 0))
                push!(seen, qlabels)
            end
        end
    end

    for i in eachindex(result)
        qlabels, _ = result[i]
        result[i] = (qlabels, dims[qlabels])
    end

    return result
end

_copy_spaces_tuple(spaces::NTuple{QD, Vector}) where {QD} = ntuple(l -> copy(spaces[l]), QD)
_tlarray_eltype(::AbstractTLArray{T}) where {T} = T

function _oplus_pad_tlarray(q::TLArray{T, QD, N, RD},
                           result_spaces,
                           dims_tuple,
                           start_dim_maps,
                           result_dim_maps) where {T, QD, N, RD}
    dims_set = Set(dims_tuple)
    qlabels = copy(q.qlabels)
    wmatdata, wmatinfo = _copy_wmat_storage(q; deep=true)
    # Padding a direct-sum leg generally destroys diagonal RMT structure.
    RMTs = Vector{Array{T, RD}}(undef, sector_count(q))
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        old_sizes = size(q.RMTs[sector_index])
        new_phys_sizes = collect(old_sizes[1:QD])
        starts = ones(Int, QD)

        for leg in 1:QD
            leg ∈ dims_set || continue
            qlabel = sector_qlabel(q, sector_index, leg)
            new_phys_sizes[leg] = result_dim_maps[leg][qlabel]
            starts[leg] = get(start_dim_maps[leg], qlabel, 1)
        end

        new_sizes = Tuple(vcat(new_phys_sizes, collect(old_sizes[QD+1:end])))
        new_data = zeros(T, new_sizes)
        fill_inds = ntuple(axis -> begin
            if axis <= QD && axis ∈ dims_set
                start = starts[axis]
                stop = start + old_sizes[axis] - 1
                start:stop
            else
                Colon()
            end
        end, RD)
        new_data[fill_inds...] = q.RMTs[sector_index]
        RMTs[sector_index] = new_data
    end
    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, _copy_spaces_tuple(result_spaces))
end

function _zero_tlarray_with_spaces(symm::NTuple{N, Any},
                                  inds::NTuple{QD, TLIndex},
                                  spaces::NTuple{QD, Vector};
                                  T::Type=Float64) where {N, QD}
    QT = qlabeltype(symm)
    qlabels = Matrix{QT}(undef, QD, 0)
    M = n_nonabelian_symmetries(productsymm(symm))
    wmatdata = Float64[]
    wmatinfo = WMatInfo{M}[]
    RMTs = Array{T, QD + N}[]
    return TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs, inds, _copy_spaces_tuple(spaces))
end

function _accumulate_oplus_starts(qs, dims_tuple, QD::Int)
    dims_set = Set(dims_tuple)
    running = [Dict{Any, Int}() for _ in 1:QD]
    starts = Vector{Vector{Dict{Any, Int}}}(undef, length(qs))

    for (qi, q) in enumerate(qs)
        starts[qi] = [Dict{Any, Int}() for _ in 1:QD]
        for leg in 1:QD
            leg ∈ dims_set || continue
            for (qlabels, dim) in q.spaces[leg]
                start = get(running[leg], qlabels, 1)
                starts[qi][leg][qlabels] = start
                running[leg][qlabels] = start + dim
            end
        end
    end

    return starts
end

_qindex_match_for_oplus(a::TLIndex, b::TLIndex) =
    a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.lock == b.lock

function _inds_match_for_oplus(inds1, inds2)
    length(inds1) == length(inds2) || return false
    return all(_qindex_match_for_oplus(idx1, idx2) for (idx1, idx2) in zip(inds1, inds2))
end

function _find_oplus_leg_permutation(ref_inds, inds, entry::Int)
    length(inds) == length(ref_inds) || throw(ArgumentError(
        "TLArray entry $entry has rank $(length(inds)), expected $(length(ref_inds))"))

    QD = length(ref_inds)
    candidates = Vector{Vector{Int}}(undef, QD)
    for i in 1:QD
        candidates[i] = [j for j in 1:QD if _qindex_match_for_oplus(inds[j], ref_inds[i])]
        isempty(candidates[i]) && throw(ArgumentError(
            "TLArray entry $entry has no leg matching reference leg $i " *
            "(same itag, direction, prime level, and lock required; dual ignored)"))
    end

    results = Vector{NTuple{QD, Int}}()
    _enum_leg_perms!(results, candidates, Int[], Set{Int}(), QD)
    isempty(results) && throw(ArgumentError(
        "TLArray entry $entry has no bijective leg permutation matching the reference indices"))
    length(results) > 1 && throw(ArgumentError(
        "TLArray entry $entry has ambiguous leg permutation matching the reference indices"))
    return results[1]
end

function _align_oplus_inputs(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    first(qs) isa AbstractTLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = to_concrete(first(qs))
    aligned = Vector{TLArray}(undef, length(qs))
    aligned[1] = ref
    for i in 2:length(qs)
        q = qs[i]
        q isa AbstractTLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        qinds = inds(q)
        perm = _find_oplus_leg_permutation(ref.inds, qinds, i)
        aligned[i] = to_concrete(
            perm == ntuple(identity, length(ref.inds)) ? q : permutedims(q, perm))
    end
    return aligned
end

function _validate_oplus_common(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    first(qs) isa TLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = first(qs)
    for (i, q) in enumerate(qs)
        q isa TLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        length(q.inds) == length(ref.inds) || throw(ArgumentError(
            "TLArray entry $i has rank $(length(q.inds)), expected $(length(ref.inds))"))
        _inds_match_for_oplus(q.inds, ref.inds) || throw(ArgumentError(
            "TLArray entry $i has different indices " *
            "(same itag, direction, prime level, and lock required; dual ignored)"))
    end

    return ref
end

function _oplus_dims_from_keywords(ref::TLArray; dir=nothing, itag=nothing,
                                   plev=nothing, lock=nothing, rev::Bool=false)
    isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock) &&
        throw(ArgumentError("oplus keyword selection requires at least one selector"))
    dims = Tuple(findlegs(ref; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev))
    isempty(dims) && throw(ArgumentError("oplus keyword selectors did not match any legs"))
    return dims
end

function _build_vector_oplus_spaces(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    QD = length(ref.inds)
    dims_set = Set(dims_tuple)

    return ntuple(leg -> begin
        if leg ∈ dims_set
            _sum_splists_many([q.spaces[leg] for q in qs])
        else
            target = ref.spaces[leg]
            for i in 2:length(qs)
                qs[i].spaces[leg] == target || throw(ArgumentError(
                    "space lists must match on non-oplus leg $leg"))
            end
            copy(target)
        end
    end, QD)
end

function _materialize_vector_oplus(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    result_spaces = _build_vector_oplus_spaces(qs, dims_tuple)
    QD = length(ref.inds)
    result_dim_maps = ntuple(leg -> _splist_dim_map(result_spaces[leg]), QD)
    start_maps = _accumulate_oplus_starts(qs, dims_tuple, QD)
    T = promote_type((_tlarray_eltype(q) for q in qs)...)

    acc = _zero_tlarray_with_spaces(symm(ref), ref.inds, result_spaces; T=T)
    for (q, qstarts) in zip(qs, start_maps)
        padded = _oplus_pad_tlarray(q, result_spaces, dims_tuple, qstarts, result_dim_maps)
        padded = q.inds == ref.inds ? padded : TLArray(padded, ref.inds)
        acc = acc + padded
    end
    return acc
end

function _oplus_matrix_entry(mat, i::Int, j::Int)
    if applicable(isassigned, mat, i, j) && !isassigned(mat, i, j)
        return nothing
    end

    val = mat[i, j]
    if val === nothing || val === missing
        return nothing
    end
    val isa AbstractTLArray || throw(ArgumentError(
        "matrix oplus entry ($i, $j) is neither an AbstractTLArray nor an undefined entry"))
    return to_concrete(val)
end

function _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i::Int, j::Int, QD::Int)
    return ntuple(leg -> begin
        have_first_axis = haskey(first_axis_sources[i], leg)
        have_second_axis = haskey(second_axis_sources[j], leg)
        if have_first_axis && have_second_axis
            first_axis_sources[i][leg] == second_axis_sources[j][leg] || throw(ArgumentError(
                "cannot infer zero TLArray at ($i, $j): inconsistent spaces on leg $leg"))
            copy(first_axis_sources[i][leg])
        elseif have_first_axis
            copy(first_axis_sources[i][leg])
        elseif have_second_axis
            copy(second_axis_sources[j][leg])
        else
            throw(ArgumentError(
                "cannot infer zero TLArray at ($i, $j): missing space information on leg $leg"))
        end
    end, QD)
end

"""
    oplus(qs::AbstractVector, dimensions)

Direct sum of a vector of `TLArray` objects along one or more physical legs.
"""
function oplus(qs::AbstractVector, dimensions)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    aligned = _align_oplus_inputs(collect(qs))
    ref = _validate_oplus_common(aligned)
    dims_tuple = _normalize_oplus_dims(dimensions, length(ref.inds))
    return _materialize_vector_oplus(aligned, dims_tuple)
end

function oplus(qs::AbstractVector; dir=nothing, itag=nothing, plev=nothing,
               lock=nothing, rev::Bool=false)
    isempty(qs) && throw(ArgumentError("oplus requires at least one TLArray"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    aligned = _align_oplus_inputs(collect(qs))
    ref = _validate_oplus_common(aligned)
    dims_tuple = _oplus_dims_from_keywords(ref; dir=dir, itag=itag, plev=plev,
                                           lock=lock, rev=rev)
    return _materialize_vector_oplus(aligned, dims_tuple)
end

function oplus(q1::AbstractTLArray, q2::AbstractTLArray, dimensions)
    return oplus(AbstractTLArray[q1, q2], dimensions)
end

function oplus(q1::AbstractTLArray, q2::AbstractTLArray; dir=nothing, itag=nothing,
               plev=nothing, lock=nothing, rev::Bool=false)
    return oplus(AbstractTLArray[q1, q2]; dir=dir, itag=itag, plev=plev,
                 lock=lock, rev=rev)
end

function _complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    size(mat, 1) > 0 && size(mat, 2) > 0 || throw(ArgumentError(
        "matrix oplus requires a non-empty matrix"))

    defined_positions = Tuple{Int, Int}[]
    defined_qs = TLArray[]
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        q === nothing && continue
        push!(defined_positions, (i, j))
        push!(defined_qs, q)
    end
    isempty(defined_qs) && throw(ArgumentError(
        "matrix oplus requires at least one defined TLArray to infer spaces"))

    aligned_qs = _align_oplus_inputs(defined_qs)
    aligned_by_position = Dict(pos => q for (pos, q) in zip(defined_positions, aligned_qs))
    ref = _validate_oplus_common(aligned_qs)
    first_axis_dims, second_axis_dims = _normalize_oplus_matrix_dims(dimensions, length(ref.inds))
    first_axis_dims_set = Set(first_axis_dims)
    second_axis_dims_set = Set(second_axis_dims)

    first_axis_sources = [Dict{Int, Any}() for _ in axes(mat, 1)]
    second_axis_sources = [Dict{Int, Any}() for _ in axes(mat, 2)]

    for ((i, j), q) in zip(defined_positions, defined_qs)
        for leg in 1:length(ref.inds)
            if leg ∉ second_axis_dims_set
                if haskey(first_axis_sources[i], leg)
                    first_axis_sources[i][leg] == q.spaces[leg] || throw(ArgumentError(
                        "sector $i has incompatible spaces on leg $leg"))
                else
                    first_axis_sources[i][leg] = copy(q.spaces[leg])
                end
            end
            if leg ∉ first_axis_dims_set
                if haskey(second_axis_sources[j], leg)
                    second_axis_sources[j][leg] == q.spaces[leg] || throw(ArgumentError(
                        "column $j has incompatible spaces on leg $leg"))
                else
                    second_axis_sources[j][leg] = copy(q.spaces[leg])
                end
            end
        end
    end

    T = promote_type((_tlarray_eltype(q) for q in defined_qs)...)
    filled = Matrix{TLArray}(undef, size(mat, 1), size(mat, 2))
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        if q === nothing
            spaces = _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i, j, length(ref.inds))
            filled[i, j] = _zero_tlarray_with_spaces(symm(ref), ref.inds, spaces; T=T)
        else
            filled[i, j] = aligned_by_position[(i, j)]
        end
    end

    return filled, first_axis_dims, second_axis_dims
end

"""
    complete_oplus_matrix(mat::AbstractMatrix, dimensions)

Validate a matrix input for `oplus`, infer zero `TLArray` objects for undefined
entries, and return the completed `Matrix{TLArray}`.
"""
function complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    filled, _, _ = _complete_oplus_matrix(mat, dimensions)
    return filled
end

function oplus(mat::AbstractMatrix, dimensions)
    filled, first_axis_dims, second_axis_dims = _complete_oplus_matrix(mat, dimensions)

    col_aggregates = Vector{TLArray}(undef, size(filled, 2))
    for j in axes(filled, 2)
        col_aggregates[j] = _materialize_vector_oplus(vec(filled[:, j]), first_axis_dims)
    end

    return _materialize_vector_oplus(col_aggregates, second_axis_dims)
end
