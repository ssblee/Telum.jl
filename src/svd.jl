# ─── svd ───────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted SVD of a TLArray object.
#
# Arguments:
#   q          : TLArray to decompose (any rank QD, N symmetries)
#   left_legs  : tuple/vector of leg indices forming the left (U) side
#   left_tag   : itag for the new bond leg on the U/S side   (default "svdL")
#   right_tag  : itag for the new bond leg on the S/Vd side  (default "svdR")
#   cutoff     : keep singular values > cutoff * σ_max  (default 1e-12)
#   Nkeep      : keep the Nkeep largest singular values globally, ignoring
#                degeneracy and counting missing sectors as zero singular values
#
# Returns SVDResult(U, S, Vd, kept_list, trunc_list) where:
#   U  : legs (original left legs...,  bond '+')
#   S  : legs (left_tag '-',  right_tag '-')  [diagonal, singular values]
#   Vd : legs (bond '-', original right legs...)
#   kept_list, trunc_list : singular-value metadata lists when get_lists=true,
#                           otherwise nothing
#
# Convention for bond legs:
#   U  bond =  TLIndex(left_tag,  '+')   — incoming (enters U from the right)
#   S  left  = TLIndex(left_tag,  '-')   — outgoing (leaves S to the left)
#   S  right = TLIndex(right_tag, '-')   — outgoing (leaves S to the right)
#   Vd bond  = TLIndex(right_tag, '-')   — outgoing (leaves Vd to the left)
#
# Legs of U and Vd that come from the original tensor inherit their TLIndex
# properties (itags, lock, plev, green, direction).
#
# Algorithm:
#   1. Assign each leg of q a unique internal tag at lock=1.
#   2. Build fusing isometries aL / aR via getIdentity.
#   3. Contract q_work with aL/aR, yielding the rank-2 bipartition M.
#   4. Per-sector SVD on M (sL × sR matrix). Truncate either by cutoff alone
#      or by the Nkeep largest singular values, where missing sectors from
#      M.spaces are treated as explicit zero singular values.
#   5. Build rank-2 U, S, Vd from kept singular values.
#   6. Split fused legs of U and Vd by contracting with conjugate of aL/aR.
#   7. Permute legs to desired order and restore original TLIndex properties.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Implement a version that get left_legs by predicates or various keyword arguments
# TODO: Test svd with trunction for TLArray object (This is not rigorous yet)
struct SVDResult{TU, TS, TVD, TKL, TTL}
    U::TU
    S::TS
    Vd::TVD
    kept_list::TKL
    trunc_list::TTL
end

function SVDResult(U::TLArray, S::TLArray, Vd::TLArray, kept_list, trunc_list)
    return SVDResult{typeof(U), typeof(S), typeof(Vd), typeof(kept_list), typeof(trunc_list)}(
        U, S, Vd, kept_list, trunc_list)
end

_svd_sector_qlabels(q::TLArray, sector_index::Int, N::Int) =
    Tuple(sector_qlabel(q, sector_index, 1)[n] for n in 1:N)

@inline function _svd_physical_qlabels(::Type{QT},
                                       qlabels_by_symm::Tuple{Vararg{Tuple, N}},
                                       cgp_by_symm::Tuple{Vararg{NTuple{QD, Int}, N}},
                                       ::Val{QD}) where {QT, N, QD}
    return ntuple(Val(QD)) do leg
        ntuple(n -> qlabels_by_symm[n][cgp_by_symm[n][leg]], Val(N))::QT
    end
end

_svd_dual_sector(symm, sector) = Tuple(get_dualq(symm[n], sector[n]) for n in 1:length(symm))

_svd_stable_sort_tuple(spaces) = Tuple(sort!(collect(spaces); alg=MergeSort))

_is_zero_array(arr::AbstractArray) = all(iszero, arr)

function _svd_symmetry_stored_leg_order(
    ::Type{QT},
    q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
    sector_index::Int,
    ::Val{n},
) where {T, QD, N, RD, QT, PS, M, RMT, n}
    qlabels_by_phys = ntuple(leg -> sector_qlabel(QT, q, sector_index, leg)[n], Val(QD))

    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for leg in 1:QD
        if q.inds[leg].dir == '+'
            push!(incoming, leg)
        else
            push!(outgoing, leg)
        end
    end

    # Stable sort encodes the TLArray/CGT tie rule: equal direction and qlabel
    # preserve physical tensor-leg order.
    sort!(incoming; by = leg -> qlabels_by_phys[leg], alg=MergeSort)
    sort!(outgoing; by = leg -> qlabels_by_phys[leg], alg=MergeSort)

    nin = length(incoming)
    stored_to_phys = ntuple(i -> i <= nin ? incoming[i] : outgoing[i - nin], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    stored_qlabels = ntuple(i -> qlabels_by_phys[stored_to_phys[i]], Val(QD))
    legdir = (nin, QD - nin)
    return stored_qlabels, phys_to_stored, stored_to_phys, legdir
end

@inline _svd_physical_qlabel_key(::Type{QT},
                                 q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                 sector_index::Int,
                                 ::Val{n}) where {T, QD, N, RD, QT, PS, M, RMT, n} =
    ntuple(leg -> sector_qlabel(QT, q, sector_index, leg)[n], Val(QD))

@inline _svd_physical_product_key(::Type{QT},
                                  q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                  sector_index::Int) where {T, QD, N, RD, QT, PS, M, RMT} =
    ntuple(leg -> sector_qlabel(QT, q, sector_index, leg), Val(QD))

function _svd_physical_key_stored_order(physical_key::NTuple{QD},
                                        inds::NTuple{QD, TLIndex}) where {QD}
    incoming = Int[]
    outgoing = Int[]
    sizehint!(incoming, QD)
    sizehint!(outgoing, QD)
    for leg in 1:QD
        if inds[leg].dir == '+'
            push!(incoming, leg)
        else
            push!(outgoing, leg)
        end
    end

    sort!(incoming; by = leg -> physical_key[leg], alg=MergeSort)
    sort!(outgoing; by = leg -> physical_key[leg], alg=MergeSort)

    nin = length(incoming)
    stored_to_phys = ntuple(i -> i <= nin ? incoming[i] : outgoing[i - nin], Val(QD))
    phys_to_stored = _phys_to_stored_order(stored_to_phys)
    stored_qlabels = ntuple(i -> physical_key[stored_to_phys[i]], Val(QD))
    legdir = (nin, QD - nin)
    return stored_qlabels, phys_to_stored, stored_to_phys, legdir
end

function _svd_physical_key_split_args(::Type{S},
                                      physical_key::NTuple{QD, NTuple{NZ, Int}},
                                      inds::NTuple{QD, TLIndex},
                                      left_legs::NTuple{L, Int}) where {S<:NonabelianSymm, QD, NZ, L}
    @assert NZ == nzops(S)
    stored_qlabels, phys_to_stored, _, legdir =
        _svd_physical_key_stored_order(physical_key, inds)
    upsp, dnsp = _svd_cgt_updn(stored_qlabels, legdir)
    left_legs_canon = _svd_to_cgtidx(phys_to_stored, left_legs)::NTuple{L, Int}
    return upsp, dnsp, left_legs_canon
end

@inline function _svd_physical_key_side_signatures(physical_key::NTuple{QD, QT},
                                                   left_legs::NTuple{L, Int},
                                                   right_legs::NTuple{R, Int}) where {QD, QT, L, R}
    left_signature = ntuple(i -> physical_key[left_legs[i]], Val(L))
    right_signature = ntuple(i -> physical_key[right_legs[i]], Val(R))
    return left_signature, right_signature
end

struct _SVDSplitRow{L, R, QT}
    sector_index::Int
    q::QT
    left_signature::NTuple{L, QT}
    right_signature::NTuple{R, QT}
end

function _svd_cgt_updn(qlabels::NTuple{QD}, legdir::Tuple{Int, Int}) where {QD}
    nin = legdir[1]
    upsp = Tuple(qlabels[i] for i in 1:nin)
    dnsp = Tuple(qlabels[i] for i in nin+1:QD)
    return upsp, dnsp
end

function _svd_to_cgtidx(cgp::NTuple{QD, Int}, lidxs) where {QD}
    return Tuple(Int[cgp[l] for l in lidxs])
end

function _svd_abelian_intermediate_q(S, qlabels::NTuple{QD}, legdir::Tuple{Int, Int}, left_legs_canon) where {QD}
    nin = legdir[1]
    leftset = Set(left_legs_canon)

    merged = _svd_stable_sort_tuple((
        Tuple(qlabels[i] for i in 1:nin if i in leftset)...,
        Tuple(get_dualq(S, qlabels[i]) for i in nin+1:QD if i in leftset)...,
    ))
    outcomes = combine_qlabels(S, merged)
    @assert length(outcomes) == 1
    return outcomes[1][1]
end

function _svd_cgt_split_spaces(qlabels::NTuple{QD},
    legdir::Tuple{Int, Int},
    left_legs_canon,
    right_legs_canon) where {QD}

    nin = legdir[1]
    QLabel = eltype(qlabels)
    left_up = QLabel[]
    left_dn = QLabel[]
    right_up = QLabel[]
    right_dn = QLabel[]

    for l in left_legs_canon
        if l <= nin push!(left_up, qlabels[l])
        else push!(left_dn, qlabels[l]) end
    end

    for l in right_legs_canon
        if l <= nin push!(right_up, qlabels[l])
        else push!(right_dn, qlabels[l]) end
    end

    left_spaces = (Tuple(left_up), Tuple(left_dn))
    right_spaces = (Tuple(right_up), Tuple(right_dn))
    return left_spaces, right_spaces
end

function _svd_cgt_split_spaces(qlabels::NTuple{QD}, cgp::NTuple{QD, Int},
                               legdir::Tuple{Int, Int}, left_legs_canon) where {QD}
    leftset = Set(left_legs_canon)
    right_legs_canon = Tuple(cgp[l] for l in 1:QD if cgp[l] ∉ leftset)
    return _svd_cgt_split_spaces(qlabels, legdir, left_legs_canon, right_legs_canon)
end

function _get_svd_cgt_split_blocks(S, qlabels::NTuple{QD}, wmat::AbstractMatrix{Float64},
                                   cgp::NTuple{QD, Int}, legdir::Tuple{Int, Int},
                                   left_legs, right_legs, physical_key,
                                   cgtsvd_cache = nothing) where {QD}
    left_legs_canon = _svd_to_cgtidx(cgp, left_legs)
    right_legs_canon = _svd_to_cgtidx(cgp, right_legs)
    NZ = length(qlabels[1])
    BlockInfo = NamedTuple{(:q, :omL, :omR, :coeffs),
        Tuple{NTuple{NZ, Int}, Int, Int, Array{Float64, 3}}}

    if isabelian(S)
        q = _svd_abelian_intermediate_q(S, qlabels, legdir, left_legs_canon)
        coeffs = reshape(copy(wmat), 1, 1, size(wmat, 2))
        _is_zero_array(coeffs) && return BlockInfo[]
        return [BlockInfo((q=q, omL=1, omR=1, coeffs=coeffs))]
    end

    upsp, dnsp = _svd_cgt_updn(qlabels, legdir)
    cgtsvd = _svd_get_cgtsvd(S, physical_key, upsp, dnsp, left_legs_canon, cgtsvd_cache)

    blocks = BlockInfo[]
    if cgtsvd isa LurCGT.CGTSVD
        coeff_split = cgtsvd.svd_arr * wmat
        offset = 1
        for (q, omL, omR) in cgtsvd.bond_sps
            width = omL * omR
            coeffs = reshape(coeff_split[offset:offset+width-1, :], omL, omR, size(coeff_split, 2))
            !_is_zero_array(coeffs) && push!(blocks, BlockInfo((q=q, omL=omL, omR=omR, coeffs=coeffs)))
            offset += width
        end
        @assert offset == size(coeff_split, 1) + 1
    else
        q = zero_qlabels((S,))[1]
        om = size(wmat, 1)
        omL, omR = cgtsvd ? (1, om) : (om, 1)
        coeffs = reshape(wmat, omL, omR, size(wmat, 2))
        push!(blocks, BlockInfo((q=q, omL=omL, omR=omR, coeffs=coeffs)))
    end
    return blocks
end

function _get_svd_cgt_split_blocks(S, qlabels::NTuple{QD}, wmat::AbstractMatrix{Float64},
                                   cgp::NTuple{QD, Int}, legdir::Tuple{Int, Int},
                                   left_legs) where {QD}
    leftset = Set(left_legs)
    right_legs = Tuple(l for l in 1:QD if l ∉ leftset)
    physical_key = ntuple(i -> qlabels[cgp[i]], Val(QD))
    return _get_svd_cgt_split_blocks(S, qlabels, wmat, cgp, legdir, left_legs, right_legs, physical_key)
end

@inline _svd_get_cgtsvd(::Type{S}, physical_key, upsp, dnsp, left_legs_canon, cache) where {S<:AbelianSymm} =
    getNsave_CGTSVD(S, upsp, dnsp, left_legs_canon; save=true)

function _svd_get_cgtsvd(::Type{S}, physical_key, upsp, dnsp, left_legs_canon, cache) where {S<:NonabelianSymm}
    cache === nothing && return getNsave_CGTSVD(S, upsp, dnsp, left_legs_canon; save=true)
    if haskey(cache, physical_key)
        return cache[physical_key]
    end
    cgtsvd = getNsave_CGTSVD(S, upsp, dnsp, left_legs_canon; save=true)
    cache[physical_key] = cgtsvd
    return cgtsvd
end

_cgtsvd_cache_key_type(::Type{QT}, ::Val{n}, ::Val{QD}) where {QT, n, QD} =
    NTuple{QD, fieldtype(QT, n)}

@generated function _cgtsvd_cache_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    1 <= n <= length(syms) || return :(throw(BoundsError(product_symms(PS), $n)))
    syms[n] <: AbelianSymm && return :(nothing)

    target = syms[n]
    slot = 0
    seen = Type[]
    for S in syms
        S <: AbelianSymm && continue
        if !(S in seen)
            push!(seen, S)
            slot += 1
        end
        S == target && return :($slot)
    end
    return :(throw(ArgumentError("symmetry index $n has no CGTSVD cache")))
end

@inline _cgtsvd_cache_for(caches, ::Type{S}, ::Type{PS}, ::Val{n}) where {S<:AbelianSymm, PS<:ProductSymm, n} = nothing
@inline function _cgtsvd_cache_for(caches, ::Type{S}, ::Type{PS}, ::Val{n}) where {S<:NonabelianSymm, PS<:ProductSymm, n}
    slot = _cgtsvd_cache_slot(PS, Val(n))
    slot === nothing &&
        throw(ArgumentError("symmetry index $n is Abelian and has no CGTSVD cache"))
    return caches[slot]
end

@generated function _new_cgtsvd_caches(::Type{PS}, ::Type{QT}, ::Val{QD}) where {PS<:ProductSymm, QT, QD}
    syms = product_symms(PS)
    seen = Type[]
    exprs = Expr[]
    for (n, S) in pairs(syms)
        S <: AbelianSymm && continue
        S in seen && continue
        push!(seen, S)
        Key = _cgtsvd_cache_key_type(QT, Val(n), Val(QD))
        # LurCGT standardization can change U/D/L or collapse to a Boolean
        # all-left/all-right shortcut, so the value type is keyed by symmetry.
        Value = Union{Bool, LurCGT.CGTSVD{S}}
        Cache = Dict{Key, Value}
        push!(exprs, :($Cache()))
    end
    return Expr(:tuple, exprs...)
end

_new_cgtsvd_caches(::TLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    _new_cgtsvd_caches(PS, QT, Val(QD))

_svd_symmetry_split_type(::Type{QTN}, ::Val{L}, ::Val{R}) where {QTN, L, R} =
    NamedTuple{(:sector_index, :q, :left_signature, :right_signature,
                :left_iso, :right_iso, :core),
        Tuple{Int, QTN, NTuple{L, QTN}, NTuple{R, QTN},
              Matrix{Float64}, Matrix{Float64}, Array{Float64, 3}}}

function _reduce_svd_cgt_split(block,
                               sector_index::Int,
                               left_signature::NTuple{L, QTN},
                               right_signature::NTuple{R, QTN};
                               tol::Float64 = 1e-12) where {L, R, QTN}
    coeffs = block.coeffs
    left_iso, left_reduced, _ = svd_leg(coeffs, 1; cutoff=tol)
    size(left_iso, 2) == 0 && return nothing

    right_iso, core, _ = svd_leg(left_reduced, 2; cutoff=tol)
    size(right_iso, 2) == 0 && return nothing

    Split = _svd_symmetry_split_type(QTN, Val(L), Val(R))
    return Split((sector_index, block.q, left_signature, right_signature,
                  left_iso, right_iso, core))
end

# Concatenate w-matrices, so the element type is always Float64
# TODO: When GPU support is added, other version should be implemented
function _copy_hcat_mats(mats::Vector{<:AbstractMatrix{Float64}}) 
    nsectors = size(first(mats), 1)
    ncols = sum(size(mat, 2) for mat in mats)
    concat = Matrix{Float64}(undef, nsectors, ncols)

    col = 1
    @views for mat in mats
        width = size(mat, 2)
        copyto!(view(concat, :, col:col+width-1), mat)
        col += width
    end
    @assert col == ncols + 1
    return concat
end

function _qr_shared_isometry(mats::Vector{AW}; tol::Float64 = 1e-12
    ) where {AW<:AbstractMatrix{Float64}}
    nsectors = size(first(mats), 1)
    @assert all(size(mat, 1) == nsectors for mat in mats) "_qr_shared_isometry requires a common sector dimension"

    if nsectors == 1
        Q = similar(first(mats), Float64, (1, 1))
        fill!(Q, 0.0)
        Q[1, 1] = 1.0
        return Q, mats
    end

    concat = _copy_hcat_mats(mats)
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

    factors = Vector{Matrix{Float64}}(undef, length(mats))
    col = 0
    for (i, mat) in pairs(mats)
        width = size(mat, 2)
        factors[i] = R[:, col+1:col+width]
        col += width
    end
    @assert col == size(concat, 2)

    return Q, factors
end

function _qr_shared_isometry(tensors::Vector{AW}; tol::Float64 = 1e-12
    ) where {AW<:AbstractArray{Float64, 3}}
    nsectors = size(first(tensors), 1)
    @assert all(size(tensor, 1) == nsectors for tensor in tensors) "_qr_shared_isometry requires a common sector dimension"

    mats = Vector{Matrix{Float64}}(undef, length(tensors))
    for (i, tensor) in pairs(tensors)
        d2, d3 = size(tensor, 2), size(tensor, 3)
        mats[i] = reshape(tensor, nsectors, d2 * d3)
    end

    shared = _qr_shared_isometry(mats; tol=tol)
    isnothing(shared) && return nothing
    Q, mat_factors = shared

    factors = Vector{Array{Float64, 3}}(undef, length(tensors))
    for i in eachindex(tensors)
        d2, d3 = size(tensors[i], 2), size(tensors[i], 3)
        factors[i] = reshape(mat_factors[i], size(mat_factors[i], 1), d2, d3)
    end

    return Q, factors
end

@inline _svd_split_signature(split, ::Val{:left}) = split.left_signature
@inline _svd_split_signature(split, ::Val{:right}) = split.right_signature
@inline _svd_split_iso(split, ::Val{:left}) = split.left_iso
@inline _svd_split_iso(split, ::Val{:right}) = split.right_iso

function _replace_svd_split_payload(split, common_iso, factor, ::Val{:left})
    return merge(split, (left_iso = common_iso,
                         core = _contract_om_axis(split.core, factor, 1)))
end

function _replace_svd_split_payload(split, common_iso, factor, ::Val{:right})
    return merge(split, (right_iso = common_iso,
                         core = _contract_om_axis(split.core, factor, 2)))
end

function _share_svd_split_side_isometries!(splits::Vector, side;
                                           tol::Float64 = 1e-12)
    isempty(splits) && return splits
    sort!(splits; by = split -> (_svd_split_signature(split, side), split.q), alg=MergeSort)

    pos = 1
    while pos <= length(splits)
        current_key = (_svd_split_signature(splits[pos], side), splits[pos].q)
        nextpos = pos
        while nextpos <= length(splits) &&
              (_svd_split_signature(splits[nextpos], side), splits[nextpos].q) == current_key
            nextpos += 1
        end

        inds = pos:nextpos-1
        if length(inds) > 1
            # TODO: When GPU support is added, 'Array{Float64, 2}' here can be generalized
            mats::Vector{Matrix{Float64}} = [
                _svd_split_iso(splits[i], side)
                for i in inds
            ]
            shared = _qr_shared_isometry(mats; tol=tol)
            @assert !isnothing(shared) "_qr_shared_isometry returned zero rank for reduced SVD blocks"
            common_iso, factors = shared

            for (i, factor) in zip(inds, factors)
                splits[i] = _replace_svd_split_payload(splits[i], common_iso, factor, side)
            end
        end

        pos = nextpos
    end

    return splits
end

function _share_svd_split_isometries!(splits_by_symm,
                                      symm;
                                      tol::Float64 = 1e-12)
    isempty(symm) && return splits_by_symm
    for n in eachindex(symm)
        isabelian(symm[n]) && continue
        _share_svd_split_side_isometries!(splits_by_symm[n], Val(:left); tol=tol)
        _share_svd_split_side_isometries!(splits_by_symm[n], Val(:right); tol=tol)
    end
    return splits_by_symm
end

function _sort_svd_splits_by_sector!(splits_by_symm)
    for splits in splits_by_symm
        sort!(splits; by = split -> (split.sector_index, split.q), alg=MergeSort)
    end
    return splits_by_symm
end

@inline _svd_trivial_iso() = ones(Float64, 1, 1)
@inline _svd_trivial_core() = reshape(ones(Float64, 1), 1, 1, 1)

function _svd_payload_tuples(::Type{PS}, sector_splits::Tuple{Vararg{Any, N}}) where {PS<:ProductSymm, N}
    M = _wmat_tuple_width(PS)
    left_isos = ntuple(Val(M)) do slot
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        sector_splits[n].left_iso
    end
    right_isos = ntuple(Val(M)) do slot
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        sector_splits[n].right_iso
    end
    cores = ntuple(Val(M)) do slot
        n = product_symmetry_index_from_wmat_slot(PS, Val(slot))
        sector_splits[n].core
    end
    return left_isos, right_isos, cores
end

function _get_svd_split_rows(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                             splits_by_symm::Tuple{Vararg{<:AbstractVector, N}},
                             sector_indices::Vector{Int},
                             left_legs::NTuple{L, Int},
                             right_legs::NTuple{R, Int},
                             ::Val{N}) where {T, QD, N, RD, QT, PS, M, RMT, L, R}
    _sort_svd_splits_by_sector!(splits_by_symm)
    Row = _SVDSplitRow{L, R, QT}
    rows = Row[]
    left_isos = NTuple{M, Matrix{Float64}}[]
    right_isos = NTuple{M, Matrix{Float64}}[]
    cores = NTuple{M, Array{Float64, 3}}[]
    (isempty(sector_indices) || any(isempty, splits_by_symm)) &&
        return rows, left_isos, right_isos, cores

    nslots = maximum(sector_indices; init=0)
    choices_by_sector = ntuple(n -> [eltype(splits_by_symm[n])[] for _ in 1:nslots], N)
    for n in 1:N
        for split in splits_by_symm[n]
            push!(choices_by_sector[n][split.sector_index], split)
        end
        @assert all(i -> !isempty(choices_by_sector[n][i]), sector_indices)
    end

    for ri in sector_indices
        physical_key = _svd_physical_product_key(QT, q, ri)
        left_signature, right_signature =
            _svd_physical_key_side_signatures(physical_key, left_legs, right_legs)
        choices = ntuple(n -> choices_by_sector[n][ri], N)
        for splits in Iterators.product(choices...)
            sector_splits = Tuple(splits)
            sector = ntuple(n -> sector_splits[n].q, Val(N))::QT
            liso, riso, core = _svd_payload_tuples(PS, sector_splits)
            push!(rows, Row(ri, sector, left_signature, right_signature))
            push!(left_isos, liso)
            push!(right_isos, riso)
            push!(cores, core)
        end
    end

    perm = sortperm(eachindex(rows);
        by = i -> (rows[i].q, rows[i].left_signature, rows[i].right_signature,
                   rows[i].sector_index),
        alg=MergeSort)
    return rows[perm], left_isos[perm], right_isos[perm], cores[perm]
end

function _get_svd_split_row_classes(rows::Vector{Row}) where {Row}
    QT = fieldtype(Row, :q)
    LeftSig = fieldtype(Row, :left_signature)
    RightSig = fieldtype(Row, :right_signature)
    classes_by_sector = Dict{QT, Vector{Vector{Int}}}()
    isempty(rows) && return classes_by_sector

    pos = 1
    while pos <= length(rows)
        sector = rows[pos].q
        nextpos = pos
        while nextpos <= length(rows) && rows[nextpos].q == sector
            nextpos += 1
        end

        row_range = pos:nextpos-1
        sectors = sort!(unique([rows[i].sector_index for i in row_range]); alg=MergeSort)
        rows_by_sector = Dict{Int, Vector{Int}}()
        left_groups = Dict{LeftSig, Vector{Int}}()
        right_groups = Dict{RightSig, Vector{Int}}()
        for i in row_range
            row = rows[i]
            push!(get!(rows_by_sector, row.sector_index, Int[]), i)
            push!(get!(left_groups, row.left_signature, Int[]), row.sector_index)
            push!(get!(right_groups, row.right_signature, Int[]), row.sector_index)
        end
        for group in values(left_groups)
            sort!(group; alg=MergeSort)
            unique!(group)
        end
        for group in values(right_groups)
            sort!(group; alg=MergeSort)
            unique!(group)
        end

        unassigned = Set(sectors)
        classes = Vector{Vector{Int}}()
        for seed in sectors
            seed in unassigned || continue
            component = Int[seed]
            frontier = Int[seed]
            delete!(unassigned, seed)
            seen_left = Set{LeftSig}()
            seen_right = Set{RightSig}()

            while !isempty(frontier)
                ri = pop!(frontier)
                for row_index in rows_by_sector[ri]
                    row = rows[row_index]
                    if row.left_signature ∉ seen_left
                        push!(seen_left, row.left_signature)
                        for rj in left_groups[row.left_signature]
                            rj in unassigned || continue
                            delete!(unassigned, rj)
                            push!(frontier, rj)
                            push!(component, rj)
                        end
                    end
                    if row.right_signature ∉ seen_right
                        push!(seen_right, row.right_signature)
                        for rj in right_groups[row.right_signature]
                            rj in unassigned || continue
                            delete!(unassigned, rj)
                            push!(frontier, rj)
                            push!(component, rj)
                        end
                    end
                end
            end

            sort!(component; alg=MergeSort)
            class_rows = Int[]
            for ri in component
                append!(class_rows, rows_by_sector[ri])
            end
            sort!(class_rows; by = i -> (rows[i].sector_index, rows[i].left_signature,
                                         rows[i].right_signature), alg=MergeSort)
            push!(classes, class_rows)
        end
        sort!(classes; by = cls -> rows[cls[1]].sector_index, alg=MergeSort)
        classes_by_sector[sector] = classes
        pos = nextpos
    end

    return classes_by_sector
end

function _get_svd_sector_spaces(q::TLArray{T, QD, N, RD, QT},
    left_legs,
    right_legs) where {T, QD, N, RD, QT}
    active_indices = [sector_index for sector_index in sector_slots(q) if !q.iszero[sector_index]]
    sector_spaces = [begin
        split_spaces = ntuple(N) do n
            qlabels, cgp, _, legdir =
                _svd_symmetry_stored_leg_order(QT, q, sector_index, Val(n))
            left_legs_canon = _svd_to_cgtidx(cgp, left_legs)
            right_legs_canon = _svd_to_cgtidx(cgp, right_legs)
            _svd_cgt_split_spaces(qlabels, legdir, left_legs_canon, right_legs_canon)
        end
        (ntuple(n -> split_spaces[n][1], N), ntuple(n -> split_spaces[n][2], N))
    end for sector_index in active_indices]
    isempty(sector_spaces) && return first.(sector_spaces), last.(sector_spaces)

    left_values = first.(sector_spaces)
    right_values = last.(sector_spaces)
    nslots = maximum(sector_slots(q); init=0)
    left_by_slot = Vector{eltype(left_values)}(undef, nslots)
    right_by_slot = Vector{eltype(right_values)}(undef, nslots)
    for (i, sector_index) in pairs(active_indices)
        left_by_slot[sector_index] = left_values[i]
        right_by_slot[sector_index] = right_values[i]
    end
    return left_by_slot, right_by_slot
end

function _get_svd_symmetry_splits(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                  sector_index::Int,
                                  left_legs::NTuple{L, Int},
                                  right_legs::NTuple{R, Int},
                                  ::Val{n};
                                  tol::Float64 = 1e-12,
                                  cgtsvd_caches = _new_cgtsvd_caches(q)) where {T, QD, N, RD, QT, PS, M, RMT, L, R, n}
    physical_key = _svd_physical_qlabel_key(QT, q, sector_index, Val(n))
    QTN = typeof(physical_key[1])
    splits = Vector{_svd_symmetry_split_type(QTN, Val(L), Val(R))}()
    left_signature, right_signature =
        _svd_physical_key_side_signatures(physical_key, left_legs, right_legs)
    qlabels, cgp, _, legdir =
        _svd_symmetry_stored_leg_order(QT, q, sector_index, Val(n))
    cache = _cgtsvd_cache_for(cgtsvd_caches, symm(q)[n], productsymm(q), Val(n))
    for block in _get_svd_cgt_split_blocks(symm(q)[n], qlabels, sector_wmat(q, sector_index, n),
                                           cgp, legdir, left_legs, right_legs, physical_key, cache)
        split = _reduce_svd_cgt_split(
            block, sector_index, left_signature, right_signature; tol=tol)
        isnothing(split) || push!(splits, split)
    end
    return splits
end

function _get_svd_cgt_split_sectors(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                    left_legs::NTuple{L, Int},
                                    right_legs::NTuple{R, Int};
                                    tol::Float64 = 1e-12) where {T, QD, N, RD, QT, PS, M, RMT, L, R}
    splits_by_symm = ntuple(Val(N)) do n
        QTN = fieldtype(QT, n)
        Vector{_svd_symmetry_split_type(QTN, Val(L), Val(R))}()
    end
    cgtsvd_caches = _new_cgtsvd_caches(q)
    for ri in sector_slots(q)
        q.iszero[ri] && continue
        for n in 1:N
            append!(splits_by_symm[n], _get_svd_symmetry_splits(
                q, ri, left_legs, right_legs, Val(n); tol=tol,
                cgtsvd_caches=cgtsvd_caches))
        end
    end
    _share_svd_split_isometries!(splits_by_symm, symm(q); tol=tol)
    return splits_by_symm
end

function _svd_expand_core_axis(A::AbstractArray{T}, core::Array{Float64, 3}, axis::Int) where {T}
    omLr, omRr, omM = size(core)
    merged = _contract_om_axis(A, reshape(core, omLr * omRr, omM), axis)
    dims = size(merged)
    return reshape(merged, dims[1:axis-1]..., omLr, omRr, dims[axis+1:end]...)
end

@inline _svd_payload_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? nothing : nonabelian_wmat_slot(PS, Val(n))

@inline _svd_left_iso(payload::NTuple{M, Matrix{Float64}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _svd_trivial_iso() : payload[nonabelian_wmat_slot(PS, Val(n))]

@inline _svd_right_iso(payload::NTuple{M, Matrix{Float64}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _svd_trivial_iso() : payload[nonabelian_wmat_slot(PS, Val(n))]

@inline _svd_core(payload::NTuple{M, Array{Float64, 3}}, ::Type{PS}, ::Val{n}) where {M, PS<:ProductSymm, n} =
    isabelian(product_symms(PS)[n]) ? _svd_trivial_core() : payload[nonabelian_wmat_slot(PS, Val(n))]

function _svd_sector_class_matrix(q::TLArray{T, QD, N, RD, QT, PS},
                               sector_index::Int,
                               left_payload,
                               right_payload,
                               core_payload,
                               left_legs::NTuple{NL, Int},
                               right_legs::NTuple{NR, Int}) where {T, QD, N, RD, QT, PS, NL, NR}
    data = sector_rmt(q, sector_index)
    for n in 1:N
        data = _svd_expand_core_axis(data, _svd_core(core_payload, PS, Val(n)), QD + 2n - 1)
    end

    perm = Tuple(vcat(
        collect(left_legs),
        [QD + 2n - 1 for n in 1:N],
        collect(right_legs),
        [QD + 2n for n in 1:N],
    ))
    permed = permutedims(data, perm)
    rmt_size = size(sector_rmt(q, sector_index))
    left_dim = prod(rmt_size[leg] for leg in left_legs) *
               prod(size(_svd_left_iso(left_payload, PS, Val(n)), 2) for n in 1:N)
    right_dim = prod(rmt_size[leg] for leg in right_legs) *
                prod(size(_svd_right_iso(right_payload, PS, Val(n)), 2) for n in 1:N)
    return reshape(permed, left_dim, right_dim)
end

@inline _svd_row_signature(row, ::Val{:left}) = row.left_signature
@inline _svd_row_signature(row, ::Val{:right}) = row.right_signature
@inline _svd_signature_field(::Val{:left}) = :left_signature
@inline _svd_signature_field(::Val{:right}) = :right_signature

function _svd_ordered_unique_split_rows(rows::Vector{Row},
                                        class_row_indices::Vector{Int},
                                        side) where {Row}
    Sig = fieldtype(Row, _svd_signature_field(side))
    ordered = Int[]
    seen = Set{Sig}()
    for row_index in class_row_indices
        row = rows[row_index]
        sig = _svd_row_signature(row, side)
        sig in seen && continue
        push!(seen, sig)
        push!(ordered, row_index)
    end
    return ordered
end

function _svd_class_side_infos(q::TLArray{T, QD, N, RD, QT, PS},
                               rows::Vector{Row},
                               class_row_indices::Vector{Int},
                               left_payloads,
                               right_payloads,
                               legs::NTuple{L, Int},
                               side) where {T, QD, N, RD, QT, PS, L, Row}
    ordered = _svd_ordered_unique_split_rows(rows, class_row_indices, side)
    Sig = fieldtype(Row, _svd_signature_field(side))
    Info = NamedTuple{(:signature, :row_index, :sector_index, :phys_dims, :om_dims, :range),
        Tuple{Sig, Int, Int, NTuple{L, Int}, NTuple{N, Int}, UnitRange{Int}}}
    infos = Info[]
    ranges = Dict{Sig, UnitRange{Int}}()
    offset = 0

    for row_index in ordered
        row = rows[row_index]
        sig = _svd_row_signature(row, side)
        ri = row.sector_index
        rmt_size = size(sector_rmt(q, ri))
        phys_dims = ntuple(i -> rmt_size[legs[i]], L)
        om_dims = ntuple(Val(N)) do n
            side === Val(:left) ?
                size(_svd_left_iso(left_payloads[row_index], PS, Val(n)), 2) :
                size(_svd_right_iso(right_payloads[row_index], PS, Val(n)), 2)
        end
        block_size = prod(phys_dims; init=1) * prod(om_dims; init=1)
        block_range = offset + 1:offset + block_size
        push!(infos, Info((sig, row_index, ri, phys_dims, om_dims, block_range)))
        ranges[sig] = block_range
        offset += block_size
    end

    return infos, ranges, offset
end

function _svd_build_side_cgt_metadata(source_qlabels::NTuple{QD},
                                      source_cgp::NTuple{QD, Int},
                                      source_legdir::Tuple{Int, Int},
                                      phys_legs::NTuple{L, Int},
                                      bond_q,
                                      bond_first::Bool,
                                      wmat::AbstractMatrix{Float64}) where {QD, L}
    stored_phys = sort!([(source_cgp[leg], leg) for leg in phys_legs]; by = first, alg=MergeSort)
    nin = source_legdir[1]

    Entry = Tuple{typeof(bond_q), Int}
    incoming = Entry[]
    outgoing = Entry[]
    if bond_first push!(outgoing, (bond_q, 0)) end
    for (stored_pos, leg) in stored_phys
        entry = (source_qlabels[stored_pos], leg)
        if stored_pos <= nin
            push!(incoming, entry)
        else
            push!(outgoing, entry)
        end
    end
    if !bond_first push!(outgoing, (bond_q, length(source_qlabels)+1)) end
    sort!(outgoing; by = first, alg=MergeSort)

    incoming_qlabels = Tuple(entry[1] for entry in incoming)
    outgoing_qlabels = Tuple(entry[1] for entry in outgoing)
    qlabels = (incoming_qlabels..., outgoing_qlabels...)
    legdir = (length(incoming), length(outgoing))

    source_to_stored = Dict{Int, Int}()
    for (stored_pos, (_, source)) in enumerate((incoming..., outgoing...))
        source_to_stored[source] = stored_pos
    end

    final_sources = bond_first ? (0, phys_legs...) : (phys_legs..., length(source_qlabels)+1)
    @assert issorted(final_sources)
    final_cgp = Tuple(source_to_stored[source] for source in final_sources)

    return (qlabels = qlabels, wmat = wmat, cgp = final_cgp, legdir = legdir)
end

function _build_svd_cgtsvd_class(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                                 sector::QT,
                                 rows::Vector{Row},
                                 class_row_indices::Vector{Int},
                                 left_payloads,
                                 right_payloads,
                                 core_payloads,
                                 left_legs::NTuple{NL, Int},
                                 right_legs::NTuple{NR, Int}) where {T, QD, N, RD, QT, PS, M, RMT, NL, NR, Row}
    left_infos, left_ranges, total_left = _svd_class_side_infos(
        q, rows, class_row_indices, left_payloads, right_payloads, left_legs, Val(:left))

    right_infos, right_ranges, total_right = _svd_class_side_infos(
        q, rows, class_row_indices, left_payloads, right_payloads, right_legs, Val(:right))

    Tmat = promote_type(T, Float64)
    mat = zeros(Tmat, total_left, total_right)
    for row_index in class_row_indices
        row = rows[row_index]
        block_mat = _svd_sector_class_matrix(
            q, row.sector_index, left_payloads[row_index], right_payloads[row_index],
            core_payloads[row_index], left_legs, right_legs)
        lrange = left_ranges[row.left_signature]
        rrange = right_ranges[row.right_signature]
        mat[lrange, rrange] .+= block_mat
    end

    F = svd(mat; full=false)
    dimq = prod(Float64(dimension(symm(q)[n], sector[n])) for n in 1:N)
    scale = sqrt(dimq)
    return (
        sector = sector,
        left_infos = left_infos,
        right_infos = right_infos,
        U = F.U .* scale,
        S = F.S ./ scale,
        Vt = F.Vt .* scale,
    )
end

function _svd_cgtsvd_entry_degeneracy(symm::NTuple{N, Any},
                                      sector::NTuple{N, Tuple{Vararg{Int}}}) where {N}
    total = 1
    for n in 1:N
        total *= dimension(symm[n], sector[n])
    end
    return total
end

function _select_svd_cgtsvd_entries(class_results,
                                    symm::NTuple{N, Any},
                                    cutoff::Float64,
                                    Nkeep;
                                    get_lists::Bool = false) where {N}
    return _select_svd_cgtsvd_entries(class_results, symm, cutoff, Nkeep, Val(get_lists))
end

function _select_svd_cgtsvd_entries(class_results,
                                    ::NTuple{N, Any},
                                    cutoff::Float64,
                                    Nkeep,
                                    ::Val{false}) where {N}
    entries = Tuple{Float64, Int, Int}[]
    sv_global_max = 0.0

    for (ci, result) in enumerate(class_results)
        isempty(result.S) && continue
        sv_global_max = max(sv_global_max, result.S[1])
        for j in eachindex(result.S)
            push!(entries, (result.S[j], ci, j))
        end
    end

    thresh = cutoff * sv_global_max
    filter!(x -> x[1] > thresh, entries)
    sort!(entries; by = x -> -x[1], alg=MergeSort)
    if !isnothing(Nkeep)
        resize!(entries, min(Nkeep, length(entries)))
    end

    keep_per_class = [Int[] for _ in 1:length(class_results)]
    for (_, ci, j) in entries
        push!(keep_per_class[ci], j)
    end
    for keep in keep_per_class
        sort!(keep)
    end

    return keep_per_class, nothing, nothing
end

function _select_svd_cgtsvd_entries(class_results,
                                    symm::NTuple{N, Any},
                                    cutoff::Float64,
                                    Nkeep,
                                    ::Val{true}) where {N}
    entries = Tuple{Float64, Int, Int}[]
    sv_global_max = 0.0
    Entry = Tuple{Float64, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}
    FullEntry = Tuple{Float64, Int, Int, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}
    full_entries = FullEntry[]
    sector_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (ci, result) in enumerate(class_results)
        isempty(result.S) && continue
        sv_global_max = max(sv_global_max, result.S[1])
        sector = result.sector::NTuple{N, Tuple{Vararg{Int}}}
        offset = get(sector_counts, sector, 0)
        degeneracy = _svd_cgtsvd_entry_degeneracy(symm, sector)
        for j in eachindex(result.S)
            display_sv = result.S[j] * sqrt(Float64(degeneracy))
            push!(entries, (result.S[j], ci, j))
            push!(full_entries, (display_sv, ci, j, degeneracy, sector, offset + j))
        end
        sector_counts[sector] = offset + length(result.S)
    end

    thresh = cutoff * sv_global_max
    filter!(x -> x[1] > thresh, entries)
    sort!(entries; by = x -> -x[1], alg=MergeSort)
    if !isnothing(Nkeep)
        resize!(entries, min(Nkeep, length(entries)))
    end

    keep_per_class = [Int[] for _ in 1:length(class_results)]
    for (_, ci, j) in entries
        push!(keep_per_class[ci], j)
    end
    for keep in keep_per_class
        sort!(keep)
    end

    selected = Set{Tuple{Int, Int}}()
    for (_, ci, j) in entries
        push!(selected, (ci, j))
    end

    kept_list = Entry[]
    trunc_list = Entry[]
    sort!(full_entries; by = x -> -x[1], alg=MergeSort)
    for (sv, ci, j, degeneracy, sector, full_index) in full_entries
        entry = (sv, degeneracy, sector, full_index)
        if (ci, j) in selected
            push!(kept_list, entry)
        else
            push!(trunc_list, entry)
        end
    end
    return keep_per_class, kept_list, trunc_list
end

function _svd_cgtsvd_class_ranges(class_results::AbstractVector, keep_per_class, ::Val{N}) where {N}
    class_ranges = Vector{UnitRange{Int}}(undef, length(class_results))
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    sector_counts = Dict{Sector, Int}()
    sector_order = Sector[]

    for (ci, result) in enumerate(class_results)
        nkeep = length(keep_per_class[ci])
        if nkeep == 0
            class_ranges[ci] = 1:0
            continue
        end

        sector = result.sector::Sector
        if !haskey(sector_counts, sector)
            sector_counts[sector] = 0
            push!(sector_order, sector)
        end
        start = sector_counts[sector] + 1
        stop = sector_counts[sector] + nkeep
        class_ranges[ci] = start:stop
        sector_counts[sector] = stop
    end

    return class_ranges, sector_counts, sector_order
end

_svd_cgtsvd_class_ranges(class_results, keep_per_class) =
    _svd_cgtsvd_class_ranges(class_results, keep_per_class, Val(length(first(class_results).sector)))

@inline _svd_append_class_result(results::Nothing, result) = typeof(result)[result]
@inline function _svd_append_class_result(results::Vector{R}, result::R) where {R}
    push!(results, result)
    return results
end

function _build_svd_cgtsvd_S(symm,
                             bond_splist,
                             left_tag::AbstractString,
                             right_tag::AbstractString,
                             sector_values)
    N = length(symm)
    base = get1jtensor(leginfo(symm, TLIndex(left_tag, '-'), bond_splist))
    qlabels = copy(base.qlabels)
    wmats = deepcopy(base.wmats)
    inds_S = (TLIndex(left_tag, '+'), TLIndex(right_tag, '+'))
    spaces_S = (bond_splist, base.spaces[2])
    RMTs = DiagRMT{Float64, 2 + N}[]

    for ri in sector_slots(base)
        base.iszero[ri] && continue
        sector = _svd_sector_qlabels(base, ri, N)
        svals = sector_values[sector]
        count = length(svals)
        cgt_dim = prod(Float64(dimension(symm[n], sector[n])) for n in 1:N)
        scale = sqrt(cgt_dim)
        push!(RMTs, _diag_rmt_from_values(svals, Val(2 + N), (1, 2), scale))
    end

    return TLArray(symm, qlabels, wmats, RMTs, inds_S, spaces_S)
end

function _svd_right_legs(::Val{QD}, left_legs::NTuple{L, Int}) where {QD, L}
    return ntuple(Val(QD - L)) do k
        nright = 0
        for leg in 1:QD
            if leg ∉ left_legs
                nright += 1
                nright == k && return leg
            end
        end
        throw(BoundsError())
    end
end

function svd_std(q::TLArray{T, QD, N, RD, QT, PS, M, RMT},
                 left_legs::NTuple{L, Int},
                 left_tag::AbstractString = "svdL",
                 right_tag::AbstractString = "svdR";
                 cutoff::Float64 = 1e-12,
                 Nkeep::Union{Nothing, Int} = nothing,
                 get_lists::Bool = false) where {T, QD, N, RD, QT, PS, M, RMT, L}
    @assert isnothing(Nkeep) || Nkeep >= 0 "Nkeep must be non-negative"
    @assert issorted(left_legs)
    @assert length(unique(left_legs)) == length(left_legs)

    R = QD - L
    right_legs::NTuple{R, Int} = _svd_right_legs(Val(QD), left_legs)

    splits_by_symm = _get_svd_cgt_split_sectors(q, left_legs, right_legs; tol=cutoff)
    active_sector_indices = [ri for ri in sector_slots(q) if !q.iszero[ri]]
    split_rows, left_payloads, right_payloads, core_payloads =
        _get_svd_split_rows(q, splits_by_symm, active_sector_indices, left_legs, right_legs, Val(N))
    split_row_classes = _get_svd_split_row_classes(split_rows)

    class_results = nothing
    for sector in sort!(collect(keys(split_row_classes)); alg=MergeSort)
        for class_row_indices in split_row_classes[sector]
            result = _build_svd_cgtsvd_class(q, sector, split_rows, class_row_indices,
                                             left_payloads, right_payloads, core_payloads,
                                             left_legs, right_legs)
            class_results = _svd_append_class_result(class_results, result)
        end
    end
    isnothing(class_results) && throw(ArgumentError("svd produced no CGTSVD classes"))

    keep_per_class, kept_list, trunc_list =
        _select_svd_cgtsvd_entries(class_results, symm(q), cutoff, Nkeep;
                                   get_lists=get_lists)
    class_ranges, sector_counts, sector_order =
        _svd_cgtsvd_class_ranges(class_results, keep_per_class, Val(N))

    Tout = promote_type(T, Float64)
    qlabels_U = NTuple{L + 1, QT}[]
    qlabels_Vd = NTuple{R + 1, QT}[]
    wmat_buffers_U = _wmat_buffers(PS)
    wmat_buffers_Vd = _wmat_buffers(PS)
    RMTs_U = Array{Tout, L + 1 + N}[]
    RMTs_Vd = Array{Tout, R + 1 + N}[]

    Sector = NTuple{N, Tuple{Vararg{Int}}}
    sector_values = Dict{Sector, Vector{Float64}}()
    for sector in sector_order
        svals = Float64[]
        for (ci, result) in enumerate(class_results)
            result.sector == sector || continue
            append!(svals, result.S[keep_per_class[ci]])
        end
        sector_values[sector] = svals
    end

    for (ci, result) in enumerate(class_results)
        keep = keep_per_class[ci]
        isempty(keep) && continue

        sector = result.sector
        dual_sector = _svd_dual_sector(symm(q), sector)
        sector_count = sector_counts[sector]
        class_range = class_ranges[ci]

        for info in result.left_infos
            left_payload = left_payloads[info.row_index]
            part = result.U[info.range, keep]
            full = zeros(eltype(result.U), length(info.range), sector_count)
            full[:, class_range] = part
            tmp = reshape(full, info.phys_dims..., info.om_dims..., sector_count)
            perm = Tuple(vcat(collect(1:L), L + N + 1, collect(L+1:L+N)))
            data = permutedims(tmp, perm)
            rmt_U = data

            cgts_U = ntuple(N) do n
                source_qlabels, source_cgp, _, source_legdir =
                    _svd_symmetry_stored_leg_order(QT, q, info.sector_index, Val(n))
                wmat = isabelian(symm(q)[n]) ? _svd_trivial_iso() :
                    copy(_svd_left_iso(left_payload, PS, Val(n)))
                _svd_build_side_cgt_metadata(
                    source_qlabels, source_cgp, source_legdir, left_legs,
                    sector[n], false, wmat)
            end
            push!(qlabels_U,
                  _svd_physical_qlabels(QT,
                                         ntuple(n -> cgts_U[n].qlabels, Val(N)),
                                         ntuple(n -> cgts_U[n].cgp, Val(N)),
                                         Val(L + 1)))
            for n in 1:N
                _push_wmat!(wmat_buffers_U, PS, n, cgts_U[n].wmat)
            end
            push!(RMTs_U, rmt_U)
        end

        for info in result.right_infos
            right_payload = right_payloads[info.row_index]
            part = result.Vt[keep, info.range]
            full = zeros(eltype(result.Vt), sector_count, length(info.range))
            full[class_range, :] = part
            rmt_Vd = reshape(full, sector_count, info.phys_dims..., info.om_dims...)

            cgts_Vd = ntuple(N) do n
                source_qlabels, source_cgp, _, source_legdir =
                    _svd_symmetry_stored_leg_order(QT, q, info.sector_index, Val(n))
                wmat = isabelian(symm(q)[n]) ? _svd_trivial_iso() :
                    copy(_svd_right_iso(right_payload, PS, Val(n)))
                _svd_build_side_cgt_metadata(
                    source_qlabels, source_cgp, source_legdir, right_legs,
                    dual_sector[n], true, wmat)
            end
            push!(qlabels_Vd,
                  _svd_physical_qlabels(QT,
                                         ntuple(n -> cgts_Vd[n].qlabels, Val(N)),
                                         ntuple(n -> cgts_Vd[n].cgp, Val(N)),
                                         Val(R + 1)))
            for n in 1:N
                _push_wmat!(wmat_buffers_Vd, PS, n, cgts_Vd[n].wmat)
            end
            push!(RMTs_Vd, rmt_Vd)
        end
    end

    bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (sector, sector_counts[sector]) for sector in sector_order
    ]
    dual_bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (_svd_dual_sector(symm(q), sector), sector_counts[sector]) for sector in sector_order
    ]
    sort!(dual_bond_splist; by = first, alg=MergeSort)

    inds_U = (ntuple(i -> q.inds[left_legs[i]], Val(L))..., TLIndex(left_tag, '-'))
    inds_Vd = (TLIndex(right_tag, '-'), ntuple(i -> q.inds[right_legs[i]], Val(R))...)

    spaces_U = (ntuple(i -> q.spaces[left_legs[i]], Val(L))..., bond_splist)
    spaces_Vd = (dual_bond_splist, ntuple(i -> q.spaces[right_legs[i]], Val(R))...)

    U_qlabels = Matrix{QT}(undef, L + 1, length(qlabels_U))
    for sector_index in eachindex(qlabels_U), leg in 1:(L + 1)
        U_qlabels[leg, sector_index] = qlabels_U[sector_index][leg]
    end
    Vd_qlabels = Matrix{QT}(undef, R + 1, length(qlabels_Vd))
    for sector_index in eachindex(qlabels_Vd), leg in 1:(R + 1)
        Vd_qlabels[leg, sector_index] = qlabels_Vd[sector_index][leg]
    end

    wmats_U = _wmat_vector_from_buffers(PS, wmat_buffers_U, length(RMTs_U))
    wmats_Vd = _wmat_vector_from_buffers(PS, wmat_buffers_Vd, length(RMTs_Vd))
    U = TLArray(symm(q), U_qlabels, wmats_U, RMTs_U, inds_U, spaces_U)
    S = _build_svd_cgtsvd_S(symm(q), bond_splist, left_tag, right_tag, sector_values)
    Vd = TLArray(symm(q), Vd_qlabels, wmats_Vd, RMTs_Vd, inds_Vd, spaces_Vd)
    return SVDResult(U, S, Vd, kept_list, trunc_list)
end

"""
    LinearAlgebra.svd(q, left_legs, left_tag="svdL", right_tag="svdR"; cutoff=1e-12, Nkeep=nothing, get_lists=false)

CGTSVD-based SVD of a `TLArray`, returning `SVDResult(U, S, Vd, kept_list, trunc_list)`.

Unlike `svd`, this path uses the direct split-space class SVD assembly and
returns the bond convention requested for CGTSVD:

- `U` bond leg: outgoing `'-'`
- `S` legs: incoming `'+'`, incoming `'+'`
- `Vd` bond leg: outgoing `'-'`
"""
function LinearAlgebra.svd(q::TLArray{T, QD, N, RD},
                    left_legs,
                    left_tag::AbstractString = "svdL",
                    right_tag::AbstractString = "svdR";
                    cutoff::Float64 = 1e-12,
                    Nkeep::Union{Nothing, Int} = nothing,
                    get_lists::Bool = false) where {T, QD, N, RD}
    left_legs_ = Tuple(_normalize_svd_left_legs(left_legs, QD))
    return svd_std(q, left_legs_, left_tag, right_tag;
                   cutoff=cutoff, Nkeep=Nkeep, get_lists=get_lists)
end

svd_cgtsvd(q::TLArray, args...; kwargs...) = svd(q, args...; kwargs...)

function LinearAlgebra.svd(q::TLArray{T, QD, N, RD},
                    left_tag::AbstractString = "svdL",
                    right_tag::AbstractString = "svdR";
                    dir=nothing,
                    itag=nothing,
                    plev=nothing,
                    lock=nothing,
                    rev::Bool=false,
                    cutoff::Float64=1e-12,
                    Nkeep::Union{Nothing, Int}=nothing,
                    get_lists::Bool=false) where {T, QD, N, RD}
    left_legs = _select_svd_left_legs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return svd(q, left_legs, left_tag, right_tag; cutoff=cutoff, Nkeep=Nkeep, get_lists=get_lists)
end

function _normalize_svd_left_legs(left_legs, rank::Int)
    legs = collect(Int, left_legs)
    isempty(legs) && throw(ArgumentError("svd requires at least one left leg"))
    length(legs) < rank || throw(ArgumentError("svd requires at most $(rank - 1) left legs for rank-$rank input"))
    all(1 .<= legs .<= rank) || throw(ArgumentError("svd left_legs must lie between 1 and $rank"))
    length(unique(legs)) == length(legs) || throw(ArgumentError("svd left_legs must not contain duplicates"))
    return collect(sort(legs))
end

function _select_svd_left_legs(q::TLArray; dir=nothing, itag=nothing, plev=nothing,
                               lock=nothing, rev::Bool=false)
    if isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock)
        throw(ArgumentError("keyword-based svd requires at least one of dir, itag, plev, or lock"))
    end

    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    rank = length(q.inds)
    1 <= length(legs) <= rank - 1 || throw(ArgumentError("keyword-based svd selected legs $legs, but the number of selected legs must be between 1 and $(rank - 1)"))
    return sort(legs)
end
