# Many-input TLArray summation.
#
# The hot path aligns inputs once, groups all sectors by physical q-labels, and
# compresses each multi-contribution sector in a single QR/RMT pass.

function _enum_leg_perms!(results, candidates, current, used, QD)
    i = length(current) + 1
    if i > QD
        push!(results, Tuple(current))
        return
    end
    for j in candidates[i]
        j ∈ used && continue
        push!(current, j)
        push!(used, j)
        _enum_leg_perms!(results, candidates, current, used, QD)
        pop!(current)
        delete!(used, j)
    end
end

# Find the unique permutation perm such that inds2[perm[i]] == inds1[i] and
# spaces2[perm[i]] == spaces1[i] for all i.
function _find_leg_permutation(inds1::NTuple{QD, TLIndex}, spaces1,
                               inds2::NTuple{QD, TLIndex}, spaces2) where {QD}
    candidates = Vector{Vector{Int}}(undef, QD)
    for i in 1:QD
        cs = Int[]
        for j in 1:QD
            if inds2[j] == inds1[i] && spaces2[j] == spaces1[i]
                push!(cs, j)
            end
        end
        isempty(cs) && error(
            "No leg in TLArray matches leg $i of reference TLArray " *
            "(itag=\"$(inds1[i].itags)\", dir='$(inds1[i].dir)')")
        candidates[i] = cs
    end
    results = Vector{NTuple{QD, Int}}()
    _enum_leg_perms!(results, candidates, Int[], Set{Int}(), QD)
    isempty(results) && error(
        "No valid bijective permutation found to match TLArray leg indices")
    length(results) > 1 && error(
        "Ambiguous permutation: $(length(results)) ways to match TLArray leg indices")
    return results[1]
end

function _align_sum_input(ref::TLArray{T, QD, N, RD, QT, PS, M, RMTS1},
                          q::TLArray{TQ, QD, N, RD, QT, PS, M, RMTS2}) where {T, TQ, QD, N, RD, QT, PS, M, RMTS1, RMTS2}
    if ref.inds == q.inds && ref.spaces == q.spaces
        return q
    end
    perm = _find_leg_permutation(ref.inds, ref.spaces, q.inds, q.spaces)
    return permutedims(q, perm)
end

function _needs_sum_alignment(ref::TLArray, q::TLArray)
    return ref.inds != q.inds || ref.spaces != q.spaces
end

function _align_sum_inputs(qs::Tuple{Vararg{<:TLArray}})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
    ref = qs[1]
    any(q -> _needs_sum_alignment(ref, q), qs) || return qs
    return ntuple(i -> _align_sum_input(ref, qs[i]), Val(length(qs)))
end

function _align_sum_inputs(qs::AbstractVector{<:TLArray})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
    ref = qs[1]
    if !any(q -> _needs_sum_alignment(ref, q), qs)
        return qs
    end
    aligned = Vector{TLArray}(undef, length(qs))
    aligned[1] = ref
    for i in 2:length(qs)
        aligned[i] = _align_sum_input(ref, qs[i])
    end
    return aligned
end

@inline _sum_sector_key(q::TLArray{T, QD}, sector_index::Int) where {T, QD} =
    ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD))

function _sum_sector_table(qs, ::Type{QT}, ::Val{QD}) where {QT, QD}
    total = 0
    for q in qs
        total += nsectors(q)
    end

    table = Vector{Tuple{NTuple{QD, QT}, Int, Int}}(undef, total)
    pos = 1
    for input_index in eachindex(qs)
        q = qs[input_index]
        for sector_index in 1:nsectors(q)
            table[pos] = (_sum_sector_key(q, sector_index), input_index, sector_index)
            pos += 1
        end
    end
    sort!(table; by = first)

    unique_count = 0
    pos = 1
    while pos <= length(table)
        unique_count += 1
        key = table[pos][1]
        pos += 1
        while pos <= length(table) && table[pos][1] == key
            pos += 1
        end
    end
    return table, unique_count
end

function _sum_rmt_cpu(::Type{T}, rmt::LurTensor{T, RD, Array{T, RD}}) where {T, RD}
    return rmt
end

function _sum_rmt_cpu(::Type{T}, rmt::LurTensor{S, RD}) where {T, S, RD}
    return LurTensor(Array{T, RD}(rmt.data))
end

@inline _sum_rmt_iszero(rmt::LurTensor) = iszero(sum(abs2, rmt.data))

function _sum_single_contribution!(result_keys::Vector{NTuple{QD, QT}},
                                   result_wmats::Vector{NTuple{M, Matrix{Float64}}},
                                   result_RMTs::Vector{LurTensor{T, RD, Array{T, RD}}},
                                   qs,
                                   entry::Tuple{NTuple{QD, QT}, Int, Int},
                                   ::Val{QD}) where {T, QD, QT, M, RD}
    key, input_index, sector_index = entry
    q = qs[input_index]
    source_rmt = sector_rmt(q, sector_index)
    rmt = QD == 0 || QD == 2 ? LurTensor(Array{T, RD}(source_rmt.data)) :
          _sum_rmt_cpu(T, source_rmt)
    _sum_rmt_iszero(rmt) && return result_keys
    push!(result_keys, key)
    push!(result_wmats, QD == 0 || QD == 2 ? deepcopy(q.wmats[sector_index]) :
                                             q.wmats[sector_index])
    push!(result_RMTs, rmt)
    return result_keys
end

function _sum_multi_contribution!(result_keys::Vector{NTuple{QD, QT}},
                                  result_wmats::Vector{NTuple{M, Matrix{Float64}}},
                                  result_RMTs::Vector{LurTensor{T, RD, Array{T, RD}}},
                                  qs,
                                  table,
                                  interval::UnitRange{Int},
                                  key::NTuple{QD, QT},
                                  ::Type{PS},
                                  ::Val{QD},
                                  ::Val{N},
                                  ::Val{M}) where {T, QD, QT, M, RD, N, PS}
    K = length(interval)
    new_RMTs = Vector{LurTensor{T, RD, Array{T, RD}}}(undef, K)
    out_pos = 1
    for pos in interval
        _, input_index, sector_index = table[pos]
        new_RMTs[out_pos] = _sum_rmt_cpu(T, sector_rmt(qs[input_index], sector_index))
        out_pos += 1
    end

    compressed = _compress_sum_sector(qs, table, interval, new_RMTs, PS, Val(QD), Val(N), Val(M))
    isnothing(compressed) && return result_keys

    sector_wmats, result_RMT = compressed
    _sum_rmt_iszero(result_RMT) && return result_keys

    push!(result_keys, key)
    push!(result_wmats, sector_wmats)
    push!(result_RMTs, result_RMT)
    return result_keys
end

function _sum_prepare_wmat_slot(qs, table, interval::UnitRange{Int}, m::Int,
                                tol::Float64)
    _, first_input, first_sector = table[first(interval)]
    first_wmat = qs[first_input].wmats[first_sector][m]
    nrows = size(first_wmat, 1)

    if nrows == 1
        factors = Vector{Matrix{Float64}}(undef, length(interval))
        out_pos = 1
        for pos in interval
            _, input_index, sector_index = table[pos]
            factors[out_pos] = qs[input_index].wmats[sector_index][m]
            out_pos += 1
        end
        return [1.0;;], factors
    end

    ncols = 0
    for pos in interval
        _, input_index, sector_index = table[pos]
        wmat = qs[input_index].wmats[sector_index][m]
        @assert size(wmat, 1) == nrows "_sum_prepare_wmat_slot requires a common sector dimension"
        ncols += size(wmat, 2)
    end

    concat = Matrix{Float64}(undef, nrows, ncols)
    col = 1
    @views for pos in interval
        _, input_index, sector_index = table[pos]
        wmat = qs[input_index].wmats[sector_index][m]
        width = size(wmat, 2)
        copyto!(view(concat, :, col:(col + width - 1)), wmat)
        col += width
    end
    @assert col == ncols + 1

    F = qr(concat)
    Qfull = Matrix(F.Q)
    Rfull = Matrix(F.R)

    max_norm = 0.0
    @inbounds for i in axes(Rfull, 1)
        nrm_sq = 0.0
        for j in axes(Rfull, 2)
            nrm_sq += abs2(Rfull[i, j])
        end
        max_norm = max(max_norm, sqrt(nrm_sq))
    end
    max_norm == 0.0 && return nothing

    threshold = tol * max_norm
    used = 1
    @inbounds for i in axes(Rfull, 1)
        nrm_sq = 0.0
        for j in axes(Rfull, 2)
            nrm_sq += abs2(Rfull[i, j])
        end
        sqrt(nrm_sq) > threshold && (used = i)
    end

    Q = Qfull[:, 1:used]
    R = Rfull[1:used, :]
    factors = Vector{Matrix{Float64}}(undef, length(interval))
    col = 0
    out_pos = 1
    for pos in interval
        _, input_index, sector_index = table[pos]
        width = size(qs[input_index].wmats[sector_index][m], 2)
        factors[out_pos] = R[:, (col + 1):(col + width)]
        col += width
        out_pos += 1
    end
    @assert col == ncols

    return Q, factors
end

function _sum_combined_factor(factors::Matrix{Matrix{Float64}},
                              contribution_index::Int,
                              ::Val{M}) where {M}
    K = ones(Float64, 1, 1)
    rank_dim = 1
    old_dim = 1

    for m in 1:M
        factor = factors[m, contribution_index]
        r_n, old_n = size(factor)
        K_new = Matrix{Float64}(undef, rank_dim * r_n, old_dim * old_n)
        @inbounds for old_idx in 1:old_dim, old_n_idx in 1:old_n,
                      rank_idx in 1:rank_dim, rank_n_idx in 1:r_n
            K_new[rank_idx + (rank_n_idx - 1) * rank_dim,
                  old_idx + (old_n_idx - 1) * old_dim] =
                K[rank_idx, old_idx] * factor[rank_n_idx, old_n_idx]
        end
        K = K_new
        rank_dim *= r_n
        old_dim *= old_n
    end

    return K
end

function _sum_sector_rank_sizes(factors::Matrix{Matrix{Float64}},
                                ::Type{PS},
                                ::Val{N},
                                ::Val{M}) where {PS, N, M}
    nonabelian_indices = nonabelian_symmetry_indices(PS)
    return ntuple(Val(N)) do n
        slot = findfirst(==(n), nonabelian_indices)
        isnothing(slot) ? 1 : size(factors[slot, 1], 1)
    end
end

function _accumulate_sum_rmts!(
    result_mat::Matrix{T},
    new_RMTs::Vector{LurTensor{T, RD, Array{T, RD}}},
    factors::Matrix{Matrix{Float64}},
    physical_dim::Int,
    ::Val{M},
) where {T, RD, M}
    for i in eachindex(new_RMTs)
        source = new_RMTs[i].data
        source_mat = reshape(source, physical_dim, div(length(source), physical_dim))
        factor = _sum_combined_factor(factors, i, Val(M))
        @assert size(source_mat, 2) == size(factor, 2)
        @assert size(result_mat, 2) == size(factor, 1)
        beta = i == firstindex(new_RMTs) ? zero(T) : one(T)
        mul!(result_mat, source_mat, transpose(factor), one(T), beta)
    end
    return result_mat
end

function _accumulate_sum_rmts!(
    result_mat::Matrix{T},
    new_RMTs::Vector{LurTensor{T, RD, Array{T, RD}}},
    factors::Matrix{Matrix{Float64}},
    physical_dim::Int,
    ::Val{1},
) where {T, RD}
    for i in eachindex(new_RMTs)
        source = new_RMTs[i].data
        source_mat = reshape(source, physical_dim, div(length(source), physical_dim))
        factor = factors[1, i]
        @assert size(source_mat, 2) == size(factor, 2)
        @assert size(result_mat, 2) == size(factor, 1)
        beta = i == firstindex(new_RMTs) ? zero(T) : one(T)
        mul!(result_mat, source_mat, transpose(factor), one(T), beta)
    end
    return result_mat
end

function _compress_sum_sector(
    qs,
    table,
    interval::UnitRange{Int},
    new_RMTs::Vector{LurTensor{T, RD, Array{T, RD}}},
    ::Type{PS},
    ::Val{QD},
    ::Val{N},
    ::Val{M},
    tol::Float64 = 1e-12,
) where {T, RD, PS, QD, N, M}
    K = length(new_RMTs)
    sector_wmats = Vector{Matrix{Float64}}(undef, M)
    factors = Matrix{Matrix{Float64}}(undef, M, K)

    for m in 1:M
        prepared = _sum_prepare_wmat_slot(qs, table, interval, m, tol)
        isnothing(prepared) && return nothing
        common_iso, split_factors = prepared
        sector_wmats[m] = common_iso
        for i in 1:K
            factors[m, i] = split_factors[i]
        end
    end

    source_shape = size(first(new_RMTs).data)
    physical_sizes = ntuple(i -> source_shape[i], Val(QD))
    physical_dim = prod(physical_sizes; init=1)
    rank_sizes = _sum_sector_rank_sizes(factors, PS, Val(N), Val(M))
    rank_dim = prod(rank_sizes; init=1)
    result_dims::NTuple{RD, Int} = (physical_sizes..., rank_sizes...)
    result_data::Array{T, RD} = Array{T, RD}(undef, result_dims)
    result_mat = reshape(result_data, physical_dim, rank_dim)

    _accumulate_sum_rmts!(result_mat, new_RMTs, factors, physical_dim, Val(M))

    return ntuple(m -> sector_wmats[m], Val(M)), LurTensor(result_data)
end

function _sum_tlarrays_aligned(qs::Union{
                                   Tuple{Vararg{<:TLArray{<:Number, QD, N, RD, QT, PS, M}}},
                                   AbstractVector{<:TLArray{<:Number, QD, N, RD, QT, PS, M}}},
                               ::Type{T}) where {T, QD, N, RD, QT, PS, M}
    table, unique_count = _sum_sector_table(qs, QT, Val(QD))
    result_keys = Vector{NTuple{QD, QT}}()
    result_wmats = Vector{NTuple{M, Matrix{Float64}}}()
    result_RMTs = Vector{LurTensor{T, RD, Array{T, RD}}}()
    sizehint!(result_keys, unique_count)
    sizehint!(result_wmats, unique_count)
    sizehint!(result_RMTs, unique_count)

    pos = 1
    while pos <= length(table)
        key = table[pos][1]
        last = pos
        while last < length(table) && table[last + 1][1] == key
            last += 1
        end
        interval = pos:last
        if pos == last
            _sum_single_contribution!(result_keys, result_wmats, result_RMTs,
                                      qs, table[pos], Val(QD))
        else
            _sum_multi_contribution!(result_keys, result_wmats, result_RMTs,
                                     qs, table, interval, key, PS,
                                     Val(QD), Val(N), Val(M))
        end
        pos = last + 1
    end

    qlabels = Matrix{QT}(undef, QD, length(result_keys))
    for sector_index in eachindex(result_keys), leg in 1:QD
        qlabels[leg, sector_index] = result_keys[sector_index][leg]
    end

    ref = qs[begin]
    return TLArray(symm(ref), qlabels, result_wmats, result_RMTs,
                   ref.inds, ref.spaces;
                   drop_zero_sectors=false)
end

function _sum_tlarrays(qs::Tuple{})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
end

function _sum_tlarrays_from_ref(
    ref::TLArray{TR, QD, N, RD, QT, PS, M},
    qs::Union{Tuple{Vararg{<:TLArray{<:Number, QD, N, RD, QT, PS, M}}},
              AbstractVector{<:TLArray{<:Number, QD, N, RD, QT, PS, M}}},
) where {TR, QD, N, RD, QT, PS, M}
    aligned = _align_sum_inputs(qs)
    T = promote_type((_qspace_eltype(q) for q in aligned)...)
    return _sum_tlarrays_aligned(aligned, T)
end

function _sum_tlarrays_from_ref(ref::TLArray{TR, QD, N, RD, QT, PS, M}, qs) where {TR, QD, N, RD, QT, PS, M}
    throw(ArgumentError(
        "TLArray sum requires matching QD, N, RD, qlabel type, product symmetry, " *
        "and w-matrix tuple width; only RMT element/storage may differ"))
end

function _sum_tlarrays(qs::Union{Tuple{Vararg{<:TLArray}}, AbstractVector{<:TLArray}})
    isempty(qs) && throw(ArgumentError("cannot sum an empty collection of TLArray objects"))
    return _sum_tlarrays_from_ref(qs[begin], qs)
end

Base.sum(qs::Tuple{Vararg{<:TLArray}}) = _sum_tlarrays(qs)
Base.sum(qs::AbstractVector{<:TLArray}) = _sum_tlarrays(qs)

function Base.:+(qs1::TLArray{T1, QD, N, RD, QT, PS},
                 qs2::TLArray{T2, QD, N, RD, QT, PS}) where {T1, T2, QD, N, RD, QT, PS}
    return _sum_tlarrays((qs1, qs2))
end

Base.:-(qs1::TLArray, qs2::TLArray) = qs1 + (-1 * qs2)
Base.:+(q::TLArray, x::Number) = q + x * _identity_on_qspace(q)
Base.:+(x::Number, q::TLArray) = q + x
Base.:-(q::TLArray, x::Number) = q + (-x) * _identity_on_qspace(q)
Base.:-(x::Number, q::TLArray) = x * _identity_on_qspace(q) + (-q)
