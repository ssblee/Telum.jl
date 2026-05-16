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
# Returns (U, S, Vd) where:
#   U  : legs (original left legs...,  bond '+')
#   S  : legs (left_tag '-',  right_tag '-')  [diagonal, singular values]
#   Vd : legs (bond '-', original right legs...)
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
_svd_sector_qlabels(q::TLArray, sector_index::Int, N::Int) =
    Tuple(sector_qlabel(q, sector_index, 1)[n] for n in 1:N)

@inline function _svd_physical_qlabels(::Type{QT}, cgrs::NTuple{N, CGR{QD}}) where {QT, N, QD}
    return ntuple(Val(QD)) do leg
        ntuple(n -> cgrs[n].qlabels[cgrs[n].cgp[leg]], Val(N))::QT
    end
end

_svd_dual_sector(symm, sector) = Tuple(get_dualq(symm[n], sector[n]) for n in 1:length(symm))

_svd_stable_sort_tuple(spaces) = Tuple(sort!(collect(spaces); alg=MergeSort))

_is_zero_array(arr::AbstractArray) = all(iszero, arr)

struct _ReducedSVDCGRBlock{NZ}
    sector_index::Int
    left_spaces::Tuple{Tuple{Vararg{NTuple{NZ, Int}}}, Tuple{Vararg{NTuple{NZ, Int}}}}
    right_spaces::Tuple{Tuple{Vararg{NTuple{NZ, Int}}}, Tuple{Vararg{NTuple{NZ, Int}}}}
    q::NTuple{NZ, Int}
    left_iso::Matrix{Float64}
    right_iso::Matrix{Float64}
    core::Array{Float64, 3}
end

function _svd_cgr_updn(cgr::CGR{QD}) where {QD}
    nin = cgr.legdir[1]
    upsp = Tuple(cgr.qlabels[i] for i in 1:nin)
    dnsp = Tuple(cgr.qlabels[i] for i in nin+1:QD)
    return upsp, dnsp
end

function _svd_to_cgridx(cgr::CGR, lidxs)
    return Tuple(Int[cgr.cgp[l] for l in lidxs])
end

_svd_cgr_leftlegs(cgr::CGR, left_legs) = _svd_to_cgridx(cgr, left_legs)

function _svd_abelian_intermediate_q(cgr::CGR{QD}, left_legs_canon) where {QD}
    S = symm(cgr)
    nin = cgr.legdir[1]
    leftset = Set(left_legs_canon)

    merged = _svd_stable_sort_tuple((
        Tuple(cgr.qlabels[i] for i in 1:nin if i in leftset)...,
        Tuple(get_dualq(S, cgr.qlabels[i]) for i in nin+1:QD if i in leftset)...,
    ))
    outcomes = combine_qlabels(S, merged)
    @assert length(outcomes) == 1
    return outcomes[1][1]
end

function _svd_cgr_split_spaces(cgr::CGR{QD}, 
    left_legs_canon, 
    right_legs_canon) where {QD}

    nin = cgr.legdir[1]
    QLabel = eltype(cgr.qlabels)
    left_up = QLabel[]
    left_dn = QLabel[]
    right_up = QLabel[]
    right_dn = QLabel[]

    for l in left_legs_canon
        if l <= nin push!(left_up, cgr.qlabels[l])
        else push!(left_dn, cgr.qlabels[l]) end
    end

    for l in right_legs_canon
        if l <= nin push!(right_up, cgr.qlabels[l])
        else push!(right_dn, cgr.qlabels[l]) end
    end

    left_spaces = (Tuple(left_up), Tuple(left_dn))
    right_spaces = (Tuple(right_up), Tuple(right_dn))
    return left_spaces, right_spaces
end

function _svd_cgr_split_spaces(cgr::CGR{QD}, left_legs_canon) where {QD}
    leftset = Set(left_legs_canon)
    right_legs_canon = Tuple(cgr.cgp[l] for l in 1:QD if cgr.cgp[l] ∉ leftset)
    return _svd_cgr_split_spaces(cgr, left_legs_canon, right_legs_canon)
end

function _get_svd_cgr_split_blocks(cgr::CGR{QD}, left_legs, right_legs) where {QD}
    left_legs_canon = _svd_to_cgridx(cgr, left_legs)
    right_legs_canon = _svd_to_cgridx(cgr, right_legs)
    NZ = length(cgr.qlabels[1])
    BlockInfo = NamedTuple{(:q, :omL, :omR, :coeffs),
        Tuple{NTuple{NZ, Int}, Int, Int, Array{Float64, 3}}}

    if isabelian(symm(cgr))
        q = _svd_abelian_intermediate_q(cgr, left_legs_canon)
        coeffs = reshape(copy(cgr.wmat.data), 1, 1, size(cgr.wmat.data, 2))
        _is_zero_array(coeffs) && return BlockInfo[]
        return [BlockInfo((q=q, omL=1, omR=1, coeffs=coeffs))]
    end

    upsp, dnsp = _svd_cgr_updn(cgr)
    cgtsvd = getNsave_CGTSVD(symm(cgr), upsp, dnsp, left_legs_canon; save=true)

    blocks = BlockInfo[]
    if cgtsvd isa LurCGT.CGTSVD
        coeff_split = cgtsvd.svd_arr * cgr.wmat.data
        offset = 1
        for (q, omL, omR) in cgtsvd.bond_sps
            width = omL * omR
            coeffs = reshape(coeff_split[offset:offset+width-1, :], omL, omR, size(coeff_split, 2))
            !_is_zero_array(coeffs) && push!(blocks, BlockInfo((q=q, omL=omL, omR=omR, coeffs=coeffs)))
            offset += width
        end
        @assert offset == size(coeff_split, 1) + 1
    else
        q = zero_qlabels((symm(cgr),))[1]
        om = size(cgr.wmat.data, 1)
        omL, omR = cgtsvd ? (1, om) : (om, 1)
        coeffs = reshape(cgr.wmat.data, omL, omR, size(cgr.wmat.data, 2))
        push!(blocks, BlockInfo((q=q, omL=omL, omR=omR, coeffs=coeffs)))
    end
    return blocks
end

function _get_svd_cgr_split_blocks(cgr::CGR{QD}, left_legs) where {QD}
    leftset = Set(left_legs)
    right_legs = Tuple(l for l in 1:QD if l ∉ leftset)
    return _get_svd_cgr_split_blocks(cgr, left_legs, right_legs)
end

function _reduce_svd_cgr_block(block,
                               sector_index::Int,
                               left_spaces,
                               right_spaces;
                               tol::Float64 = 1e-12)
    coeffs = block.coeffs
    left_iso, left_reduced, _ = svd_leg(coeffs, 1; cutoff=tol)
    size(left_iso, 2) == 0 && return nothing

    right_iso, core, _ = svd_leg(left_reduced, 2; cutoff=tol)
    size(right_iso, 2) == 0 && return nothing

    return _ReducedSVDCGRBlock{length(block.q)}(
        sector_index, left_spaces, right_spaces, block.q, left_iso, right_iso, core)
end

function _reconstruct_reduced_svd_cgr_block(block::_ReducedSVDCGRBlock)
    omLr, omRr, omM = size(block.core)
    left_applied = reshape(
        block.left_iso * reshape(block.core, omLr, :),
        size(block.left_iso, 1), omRr, omM)
    right_applied = reshape(
        block.right_iso * reshape(permutedims(left_applied, (2, 1, 3)), omRr, :),
        size(block.right_iso, 1), size(block.left_iso, 1), omM)
    return permutedims(right_applied, (2, 1, 3))
end

# Concatenate w-matrices, so the element type is always Float64
# TODO: When GPU support is added, other version should be implemented
function _copy_hcat_mats(mats::Vector{LurTensor{Float64, 2, Array{Float64, 2}}}) 
    nsectors = size(first(mats), 1)
    ncols = sum(size(mat, 2) for mat in mats)
    concat = Matrix{Float64}(undef, nsectors, ncols)

    col = 1
    @views for mat in mats
        width = size(mat, 2)
        copyto!(view(concat, :, col:col+width-1), mat.data)
        col += width
    end
    @assert col == ncols + 1
    return concat
end

function _qr_shared_isometry(mats::Vector{LurTensor{Float64, 2, AW}}; tol::Float64 = 1e-12
    ) where {AW<:AbstractArray{Float64, 2}}
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
    Q = LurTensor(Qfull[:, 1:used])
    R = Rfull[1:used, :]

    factors = Vector{LurTensor{Float64, 2, AW}}(undef, length(mats))
    col = 0
    for (i, mat) in pairs(mats)
        width = size(mat, 2)
        factors[i] = LurTensor(R[:, col+1:col+width])
        col += width
    end
    @assert col == size(concat, 2)

    return Q, factors
end

function _qr_shared_isometry(tensors::Vector{LurTensor{Float64, 3, AW}}; tol::Float64 = 1e-12
    ) where {AW<:AbstractArray{Float64, 3}}
    nsectors = size(first(tensors), 1)
    @assert all(size(tensor, 1) == nsectors for tensor in tensors) "_qr_shared_isometry requires a common sector dimension"

    mats = Vector{LurTensor{Float64, 2, Matrix{Float64}}}(undef, length(tensors))
    for (i, tensor) in pairs(tensors)
        d2, d3 = size(tensor, 2), size(tensor, 3)
        mats[i] = LurTensor(reshape(tensor.data, nsectors, d2 * d3))
    end

    shared = _qr_shared_isometry(mats; tol=tol)
    isnothing(shared) && return nothing
    Q, mat_factors = shared

    factors = Vector{LurTensor{Float64, 3, Array{Float64, 3}}}(undef, length(tensors))
    for i in eachindex(tensors)
        d2, d3 = size(tensors[i], 2), size(tensors[i], 3)
        factors[i] = LurTensor(reshape(mat_factors[i].data, size(mat_factors[i], 1), d2, d3))
    end

    return Q, factors
end

function _share_svd_sector_side_isometries!(blocks::Vector{<:_ReducedSVDCGRBlock},
                                         side::Symbol;
                                         tol::Float64 = 1e-12)
    isempty(blocks) && return blocks
    spacekey(block) = side === :left ? block.left_spaces : block.right_spaces
    sort!(blocks; by = block -> (spacekey(block), block.q), alg=MergeSort)

    pos = 1
    while pos <= length(blocks)
        current_key = (spacekey(blocks[pos]), blocks[pos].q)
        nextpos = pos
        while nextpos <= length(blocks) &&
              (spacekey(blocks[nextpos]), blocks[nextpos].q) == current_key
            nextpos += 1
        end

        inds = pos:nextpos-1
        if length(inds) > 1
            # TODO: When GPU support is added, 'Array{Float64, 2}' here can be generalized
            mats::Vector{LurTensor{Float64, 2, Array{Float64, 2}}} = [
                LurTensor(side === :left ?  blocks[i].left_iso : blocks[i].right_iso)
                for i in inds
            ]
            shared = _qr_shared_isometry(mats; tol=tol)
            @assert !isnothing(shared) "_qr_shared_isometry returned zero rank for reduced SVD blocks"
            common_iso, factors = shared

            for (i, factor) in zip(inds, factors)
                block = blocks[i]
                if side === :left
                    new_core = _contract_om_axis(LurTensor(block.core), factor, 1)
                    blocks[i] = _ReducedSVDCGRBlock{length(block.q)}(
                        block.sector_index, block.left_spaces, block.right_spaces,
                        block.q, common_iso, block.right_iso, new_core)
                else
                    new_core = _contract_om_axis(LurTensor(block.core), factor, 2)
                    blocks[i] = _ReducedSVDCGRBlock{length(block.q)}(
                        block.sector_index, block.left_spaces, block.right_spaces,
                        block.q, block.left_iso, common_iso, new_core)
                end
            end
        end

        pos = nextpos
    end

    return blocks
end

function _share_svd_sector_isometries!(blocks_by_symm,
                                    symm;
                                    tol::Float64 = 1e-12)
    isempty(symm) && return blocks_by_symm
    for n in eachindex(symm)
        isabelian(symm[n]) && continue
        _share_svd_sector_side_isometries!(blocks_by_symm[n], :left; tol=tol)
        _share_svd_sector_side_isometries!(blocks_by_symm[n], :right; tol=tol)
    end
    return blocks_by_symm
end

function _sort_svd_blocks_by_sector!(blocks_by_symm)
    for blocks in blocks_by_symm
        sort!(blocks; by = block -> (block.sector_index, block.q), alg=MergeSort)
    end
    return blocks_by_symm
end

function _get_svd_intermediate_sector_dict(blocks_by_symm::Tuple{Vararg{<:AbstractVector, N}},
                                         nsectors::Int) where {N}
    _sort_svd_blocks_by_sector!(blocks_by_symm)
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    qsectors = Dict{Sector, Vector{Int}}()
    (nsectors == 0 || any(isempty, blocks_by_symm)) && return qsectors

    qchoices_by_sector = ntuple(_ -> [Tuple{Vararg{Int}}[] for _ in 1:nsectors], N)
    for n in 1:N
        @assert first(blocks_by_symm[n]).sector_index == 1
        @assert last(blocks_by_symm[n]).sector_index == nsectors
        for block in blocks_by_symm[n]
            qlist = qchoices_by_sector[n][block.sector_index]
            (isempty(qlist) || qlist[end] != block.q) && push!(qlist, block.q)
        end
        @assert all(!isempty, qchoices_by_sector[n])
    end

    for ri in 1:nsectors
        choices = ntuple(n -> qchoices_by_sector[n][ri], N)
        for sector in Iterators.product(choices...)
            push!(get!(qsectors, Tuple(sector), Int[]), ri)
        end
    end

    return qsectors
end

function _get_svd_sector_spaces(q::TLArray{T, QD, N, RD},
    left_legs,
    right_legs) where {T, QD, N, RD}
    sector_spaces = [begin
        split_spaces = ntuple(N) do n
            cgr = _sector_cgr(q, sector_index, n)
            left_legs_canon = _svd_to_cgridx(cgr, left_legs)
            right_legs_canon = _svd_to_cgridx(cgr, right_legs)
            _svd_cgr_split_spaces(cgr, left_legs_canon, right_legs_canon)
        end
        (ntuple(n -> split_spaces[n][1], N), ntuple(n -> split_spaces[n][2], N))
    end for sector_index in 1:nsectors(q)]
    return first.(sector_spaces), last.(sector_spaces)
end

function _get_svd_intermediate_sector_equivclasses(
    left_sigs,
    right_sigs,
    qsectors::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}},
) where {N}
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    classes_by_sector = Dict{Sector, Vector{Vector{Int}}}()
    isempty(qsectors) && return classes_by_sector
    LeftSig = eltype(left_sigs)
    RightSig = eltype(right_sigs)

    for sector in sort!(collect(keys(qsectors)); alg=MergeSort)
        sectors = sort!(copy(qsectors[sector]); alg=MergeSort)
        left_groups = Dict{LeftSig, Vector{Int}}()
        right_groups = Dict{RightSig, Vector{Int}}()

        for ri in sectors
            push!(get!(left_groups, left_sigs[ri], Int[]), ri)
            push!(get!(right_groups, right_sigs[ri], Int[]), ri)
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
                left_sig = left_sigs[ri]
                if left_sig ∉ seen_left
                    push!(seen_left, left_sig)
                    for rj in left_groups[left_sig]
                        rj in unassigned || continue
                        delete!(unassigned, rj)
                        push!(frontier, rj)
                        push!(component, rj)
                    end
                end

                right_sig = right_sigs[ri]
                if right_sig ∉ seen_right
                    push!(seen_right, right_sig)
                    for rj in right_groups[right_sig]
                        rj in unassigned || continue
                        delete!(unassigned, rj)
                        push!(frontier, rj)
                        push!(component, rj)
                    end
                end
            end

            sort!(component; alg=MergeSort)
            push!(classes, component)
        end
        sort!(classes; by = cls -> cls[1], alg=MergeSort)
        classes_by_sector[sector] = classes
    end

    return classes_by_sector
end

function _get_svd_intermediate_sector_equivclasses(
    q::TLArray{T, QD, N, RD},
    left_legs,
    qsectors::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}},
) where {T, QD, N, RD}
    left_legs_ = _normalize_svd_left_legs(left_legs, QD)
    right_legs_ = [l for l in 1:QD if l ∉ left_legs_]
    left_sigs, right_sigs = _get_svd_sector_spaces(q, Tuple(left_legs_), Tuple(right_legs_))
    return _get_svd_intermediate_sector_equivclasses(left_sigs, right_sigs, qsectors)
end

function _get_svd_sector_split_blocks(q::TLArray{T, QD, N, RD},
                                   sector_index::Int,
                                   left_legs,
                                   right_legs;
                                   tol::Float64 = 1e-12) where {T, QD, N, RD}
    split_spaces = ntuple(N) do n
        cgr = _sector_cgr(q, sector_index, n)
        left_legs_canon = _svd_to_cgridx(cgr, left_legs)
        right_legs_canon = _svd_to_cgridx(cgr, right_legs)
        _svd_cgr_split_spaces(cgr, left_legs_canon, right_legs_canon)
    end
    symm_blocks = ntuple(N) do n
        cgr = _sector_cgr(q, sector_index, n)
        left_spaces_n, right_spaces_n = split_spaces[n]
        reduced = Vector{_ReducedSVDCGRBlock{nzops(cgr.symm)}}()
        for block in _get_svd_cgr_split_blocks(cgr, left_legs, right_legs)
            reduced_block = _reduce_svd_cgr_block(
                block, sector_index, left_spaces_n, right_spaces_n; tol=tol)
            isnothing(reduced_block) || push!(reduced, reduced_block)
        end
        reduced
    end
    any(isempty, symm_blocks) && return nothing
    return symm_blocks
end

function _get_svd_cgt_split_sectors(q::TLArray{T, QD, N, RD},
                                 left_legs,
                                 right_legs;
                                 tol::Float64 = 1e-12) where {T, QD, N, RD}
    blocks_by_symm = ntuple(n -> Vector{_ReducedSVDCGRBlock{nzops(symm(q)[n])}}(), N)
    for ri in 1:nsectors(q)
        sector_blocks = _get_svd_sector_split_blocks(q, ri, left_legs, right_legs; tol=tol)
        isnothing(sector_blocks) && continue
        for n in 1:N
            append!(blocks_by_symm[n], sector_blocks[n])
        end
    end
    return blocks_by_symm
end

function _get_svd_cgt_split_sectors(q::TLArray{T, QD, N, RD},
                                 left_legs;
                                 tol::Float64 = 1e-12) where {T, QD, N, RD}
    left_legs_ = _normalize_svd_left_legs(left_legs, QD)
    right_legs_ = [l for l in 1:QD if l ∉ left_legs_]
    return _get_svd_cgt_split_sectors(q, Tuple(left_legs_), Tuple(right_legs_); tol=tol)
end

function _build_svd_block_lookup(blocks)
    lookup = Dict{Any, Any}()
    for block in blocks
        key = (block.sector_index, block.q)
        @assert !haskey(lookup, key) "duplicate CGTSVD block for sector $(block.sector_index), q=$(block.q)"
        lookup[key] = block
    end
    return lookup
end

function _preprocess_svd_cgtsvd(q::TLArray{T, QD, N, RD},
                                left_legs;
                                tol::Float64 = 1e-12) where {T, QD, N, RD}
    left_legs_ = _normalize_svd_left_legs(left_legs, QD)
    right_legs_ = [l for l in 1:QD if l ∉ left_legs_]
    left_signatures, right_signatures = 
    _get_svd_sector_spaces(q, left_legs_, right_legs_)

    blocks_by_symm = _get_svd_cgt_split_sectors(q, left_legs_, right_legs_; tol=tol)
    _share_svd_sector_isometries!(blocks_by_symm, symm(q); tol=tol)
    intermediate_qsectors = _get_svd_intermediate_sector_dict(blocks_by_symm, nsectors(q))
    intermediate_qsector_classes = _get_svd_intermediate_sector_equivclasses(
        left_signatures, right_signatures, intermediate_qsectors)
    block_lookup_by_symm = ntuple(n -> _build_svd_block_lookup(blocks_by_symm[n]), N)
    return (
        left_legs = Tuple(left_legs_),
        right_legs = Tuple(right_legs_),
        left_signatures = left_signatures,
        right_signatures = right_signatures,
        blocks_by_symm = blocks_by_symm,
        block_lookup_by_symm = block_lookup_by_symm,
        intermediate_qsectors = intermediate_qsectors,
        intermediate_qsector_classes = intermediate_qsector_classes,
    )
end

function _svd_expand_core_axis(A::AbstractArray{T}, core::Array{Float64, 3}, axis::Int) where {T}
    omLr, omRr, omM = size(core)
    merged = _contract_om_axis(LurTensor(A), LurTensor(reshape(core, omLr * omRr, omM)), axis)
    dims = size(merged)
    return reshape(merged, dims[1:axis-1]..., omLr, omRr, dims[axis+1:end]...)
end

function _svd_blocks_for_sector(prep, sector_index::Int, sector)
    ntuple(length(sector)) do n
        key = (sector_index, sector[n])
        @assert haskey(prep.block_lookup_by_symm[n], key) "missing CGTSVD block for sector $sector_index, sector $sector, symmetry $n"
        prep.block_lookup_by_symm[n][key]
    end
end

function _svd_sector_class_matrix(q::TLArray{T, QD, N, RD},
                               sector_index::Int,
                               sector_blocks,
                               left_legs::NTuple{NL, Int},
                               right_legs::NTuple{NR, Int}) where {T, QD, N, RD, NL, NR}
    data = sector_rmt(q, sector_index).data
    for n in 1:N
        data = _svd_expand_core_axis(data, sector_blocks[n].core, QD + 2n - 1)
    end

    perm = Tuple(vcat(
        collect(left_legs),
        [QD + 2n - 1 for n in 1:N],
        collect(right_legs),
        [QD + 2n for n in 1:N],
    ))
    permed = permutedims(data, perm)
    left_dim = prod(size(sector_rmt(q, sector_index).data, leg) for leg in left_legs) *
               prod(size(sector_blocks[n].left_iso, 2) for n in 1:N)
    right_dim = prod(size(sector_rmt(q, sector_index).data, leg) for leg in right_legs) *
                prod(size(sector_blocks[n].right_iso, 2) for n in 1:N)
    return reshape(permed, left_dim, right_dim)
end

function _svd_ordered_unique_signatures(sector_signatures, class_sectors::Vector{Int})
    Sig = eltype(sector_signatures)
    ordered = Sig[]
    seen = Set{Sig}()
    for ri in class_sectors
        sig = sector_signatures[ri]
        sig in seen && continue
        push!(seen, sig)
        push!(ordered, sig)
    end
    return ordered
end

function _svd_class_side_infos(q::TLArray{T, QD, N, RD},
                               prep,
                               sector,
                               class_sectors::Vector{Int},
                               sector_signatures,
                               legs::NTuple{L, Int},
                               side::Symbol) where {T, QD, N, RD, L}
    ordered = _svd_ordered_unique_signatures(sector_signatures, class_sectors)
    infos = Any[]
    ranges = Dict{eltype(sector_signatures), UnitRange{Int}}()
    offset = 0

    for sig in ordered
        ri = first(filter(rj -> sector_signatures[rj] == sig, class_sectors))
        sector_blocks = _svd_blocks_for_sector(prep, ri, sector)
        phys_dims = ntuple(i -> size(sector_rmt(q, ri).data, legs[i]), L)
        om_dims = ntuple(n -> size(side === :left ? sector_blocks[n].left_iso : sector_blocks[n].right_iso, 2), N)
        block_size = prod(phys_dims; init=1) * prod(om_dims; init=1)
        block_range = offset + 1:offset + block_size
        push!(infos, (signature=sig, sector_index=ri, phys_dims=phys_dims, om_dims=om_dims, range=block_range))
        ranges[sig] = block_range
        offset += block_size
    end

    return infos, ranges, offset
end

function _svd_build_side_cgr(symm,
                             source_cgr::CGR,
                             phys_legs::NTuple{L, Int},
                             bond_q,
                             bond_first::Bool,
                             wmat::LurTensor{Float64, 2}) where {L}
    stored_phys = sort!([(source_cgr.cgp[leg], leg) for leg in phys_legs]; by = first, alg=MergeSort)
    nin = source_cgr.legdir[1]

    Entry = Tuple{typeof(bond_q), Int}
    incoming = Entry[]
    outgoing = Entry[]
    if bond_first push!(outgoing, (bond_q, 0)) end
    for (stored_pos, leg) in stored_phys
        entry = (source_cgr.qlabels[stored_pos], leg)
        if stored_pos <= nin
            push!(incoming, entry)
        else
            push!(outgoing, entry)
        end
    end
    if !bond_first push!(outgoing, (bond_q, length(source_cgr.qlabels)+1)) end
    sort!(outgoing; by = first, alg=MergeSort)

    incoming_qlabels = Tuple(entry[1] for entry in incoming)
    outgoing_qlabels = Tuple(entry[1] for entry in outgoing)
    qlabels = (incoming_qlabels..., outgoing_qlabels...)
    legdir = (length(incoming), length(outgoing))

    source_to_stored = Dict{Int, Int}()
    for (stored_pos, (_, source)) in enumerate((incoming..., outgoing...))
        source_to_stored[source] = stored_pos
    end

    final_sources = bond_first ? (0, phys_legs...) : (phys_legs..., length(source_cgr.qlabels)+1)
    @assert issorted(final_sources)
    final_cgp = Tuple(source_to_stored[source] for source in final_sources)

    return CGR(symm, qlabels, wmat, final_cgp, legdir)
end

function _build_svd_cgtsvd_class(q::TLArray{T, QD, N, RD},
                                 prep,
                                 sector,
                                 class_sectors::Vector{Int},
                                 left_legs::NTuple{NL, Int},
                                 right_legs::NTuple{NR, Int}) where {T, QD, N, RD, NL, NR}
    left_infos, left_ranges, total_left = _svd_class_side_infos(
        q, prep, sector, class_sectors, prep.left_signatures, left_legs, :left)
    
    right_infos, right_ranges, total_right = _svd_class_side_infos(
        q, prep, sector, class_sectors, prep.right_signatures, right_legs, :right)

    Tmat = promote_type(T, Float64)
    mat = zeros(Tmat, total_left, total_right)
    for ri in class_sectors
        sector_blocks = _svd_blocks_for_sector(prep, ri, sector)
        block_mat = _svd_sector_class_matrix(q, ri, sector_blocks, left_legs, right_legs)
        lrange = left_ranges[prep.left_signatures[ri]]
        rrange = right_ranges[prep.right_signatures[ri]]
        mat[lrange, rrange] .+= block_mat
    end


    F = svd(mat; full=false)
    dimq = prod(Float64(dimension(symm(q)[n], sector[n])) for n in 1:N)
    scale = sqrt(dimq)
    return (
        sector = sector,
        class_sectors = class_sectors,
        left_infos = left_infos,
        right_infos = right_infos,
        U = F.U .* scale,
        # `get1jtensor` contributes the remaining 1/sqrt(dimq) normalization
        # when the center bridge is materialized as a rank-2 TLArray sector.
        S = F.S ./ scale,
        Vt = F.Vt .* scale,
    )
end

function _select_svd_cgtsvd_entries(class_results, cutoff::Float64, Nkeep)
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
    return keep_per_class
end

function _svd_cgtsvd_class_ranges(class_results, keep_per_class)
    class_ranges = Vector{UnitRange{Int}}(undef, length(class_results))
    sector_counts = Dict{Any, Int}()
    sector_order = Any[]

    for (ci, result) in enumerate(class_results)
        nkeep = length(keep_per_class[ci])
        if nkeep == 0
            class_ranges[ci] = 1:0
            continue
        end

        sector = result.sector
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

function _build_svd_cgtsvd_S(symm,
                             bond_splist,
                             left_tag::AbstractString,
                             right_tag::AbstractString,
                             sector_values)
    N = length(symm)
    base = get1jtensor(leginfo(symm, TLIndex(left_tag, '-'), bond_splist))
    qlabels = copy(base.qlabels)
    wmats = deepcopy(base.wmats)
    RMTs = LurTensor{Float64, 2 + N, Array{Float64, 2 + N}}[]

    for ri in 1:nsectors(base)
        sector = _svd_sector_qlabels(base, ri, N)
        svals = sector_values[sector]
        count = length(svals)
        rmt = LurTensor(reshape(Matrix(Diagonal(svals)), count, count, ones(Int, N)...))
        push!(RMTs, rmt)
    end

    inds_S = (TLIndex(left_tag, '+'), TLIndex(right_tag, '+'))
    spaces_S = (bond_splist, base.spaces[2])
    return _field_tlarray(symm, qlabels, wmats, RMTs, inds_S, spaces_S)
end

function _assemble_svd_cgtsvd(q::TLArray{T, QD, N, RD},
                              left_legs::NTuple{NL, Int},
                              right_legs::NTuple{NR, Int},
                              left_tag::AbstractString,
                              right_tag::AbstractString,
                              prep;
                              cutoff::Float64 = 1e-12,
                              Nkeep::Union{Nothing, Int} = nothing) where {T, QD, N, RD, NL, NR}
    class_results = Any[]
    for sector in sort!(collect(keys(prep.intermediate_qsector_classes)); alg=MergeSort)
        for class_sectors in prep.intermediate_qsector_classes[sector]
            push!(class_results,
                  _build_svd_cgtsvd_class(q, prep, sector, class_sectors, left_legs, right_legs))
        end
    end

    keep_per_class = _select_svd_cgtsvd_entries(class_results, cutoff, Nkeep)
    class_ranges, sector_counts, sector_order = _svd_cgtsvd_class_ranges(class_results, keep_per_class)

    Tout = promote_type(T, Float64)
    QT = qlabeltype(q)
    qlabels_U = NTuple{NL + 1, QT}[]
    qlabels_Vd = NTuple{NR + 1, QT}[]
    wmats_U = ntuple(_ -> LurTensor{Float64, 2, Matrix{Float64}}[], Val(N))
    wmats_Vd = ntuple(_ -> LurTensor{Float64, 2, Matrix{Float64}}[], Val(N))
    RMTs_U = LurTensor{Tout, NL + 1 + N, Array{Tout, NL + 1 + N}}[]
    RMTs_Vd = LurTensor{Tout, NR + 1 + N, Array{Tout, NR + 1 + N}}[]

    sector_values = Dict{Any, Vector{Float64}}()
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
            sector_blocks = _svd_blocks_for_sector(prep, info.sector_index, sector)
            part = result.U[info.range, keep]
            full = zeros(eltype(result.U), length(info.range), sector_count)
            full[:, class_range] = part
            tmp = reshape(full, info.phys_dims..., info.om_dims..., sector_count)
            perm = Tuple(vcat(collect(1:NL), NL + N + 1, collect(NL+1:NL+N)))
            data = permutedims(tmp, perm)
            rmt_U = LurTensor(data)

            cgrs_U = ntuple(N) do n
                _svd_build_side_cgr(
                    symm(q)[n],
                    _sector_cgr(q, info.sector_index, n),
                    left_legs,
                    sector[n],
                    false,
                    LurTensor(copy(sector_blocks[n].left_iso)),
                )
            end
            push!(qlabels_U, _svd_physical_qlabels(QT, cgrs_U))
            for n in 1:N
                push!(wmats_U[n], cgrs_U[n].wmat)
            end
            push!(RMTs_U, rmt_U)
        end

        for info in result.right_infos
            sector_blocks = _svd_blocks_for_sector(prep, info.sector_index, sector)
            part = result.Vt[keep, info.range]
            full = zeros(eltype(result.Vt), sector_count, length(info.range))
            full[class_range, :] = part
            rmt_Vd = LurTensor(reshape(full, sector_count, info.phys_dims..., info.om_dims...))

            cgrs_Vd = ntuple(N) do n
                _svd_build_side_cgr(
                    symm(q)[n],
                    _sector_cgr(q, info.sector_index, n),
                    right_legs,
                    dual_sector[n],
                    true,
                    LurTensor(copy(sector_blocks[n].right_iso)),
                )
            end
            push!(qlabels_Vd, _svd_physical_qlabels(QT, cgrs_Vd))
            for n in 1:N
                push!(wmats_Vd[n], cgrs_Vd[n].wmat)
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

    inds_U = (ntuple(i -> q.inds[left_legs[i]], NL)..., TLIndex(left_tag, '-'))
    inds_Vd = (TLIndex(right_tag, '-'), ntuple(i -> q.inds[right_legs[i]], NR)...)

    spaces_U = (ntuple(i -> q.spaces[left_legs[i]], NL)..., bond_splist)
    spaces_Vd = (dual_bond_splist, ntuple(i -> q.spaces[right_legs[i]], NR)...)

    U_qlabels = Matrix{QT}(undef, length(qlabels_U), NL + 1)
    for sector_index in eachindex(qlabels_U), leg in 1:(NL + 1)
        U_qlabels[sector_index, leg] = qlabels_U[sector_index][leg]
    end
    Vd_qlabels = Matrix{QT}(undef, length(qlabels_Vd), NR + 1)
    for sector_index in eachindex(qlabels_Vd), leg in 1:(NR + 1)
        Vd_qlabels[sector_index, leg] = qlabels_Vd[sector_index][leg]
    end

    U = _field_tlarray(symm(q), U_qlabels, wmats_U, RMTs_U, inds_U, spaces_U)
    S = _build_svd_cgtsvd_S(symm(q), bond_splist, left_tag, right_tag, sector_values)
    Vd = _field_tlarray(symm(q), Vd_qlabels, wmats_Vd, RMTs_Vd, inds_Vd, spaces_Vd)
    return U, S, Vd
end

"""
    LinearAlgebra.svd(q, left_legs, left_tag="svdL", right_tag="svdR"; cutoff=1e-12, Nkeep=nothing)

CGTSVD-based SVD of a `TLArray`, returning `(U, S, Vd)`.

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
                    Nkeep::Union{Nothing, Int} = nothing) where {T, QD, N, RD}
    @assert isnothing(Nkeep) || Nkeep >= 0 "Nkeep must be non-negative"
    prep = _preprocess_svd_cgtsvd(q, left_legs; tol=cutoff)
    return _assemble_svd_cgtsvd(
        q, prep.left_legs, prep.right_legs, left_tag, right_tag, prep;
        cutoff=cutoff, Nkeep=Nkeep)
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
                    Nkeep::Union{Nothing, Int}=nothing) where {T, QD, N, RD}
    left_legs = _select_svd_left_legs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return svd(q, left_legs, left_tag, right_tag; cutoff=cutoff, Nkeep=Nkeep)
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
