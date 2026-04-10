# ─── svd ───────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted SVD of a QSpace object.
#
# Arguments:
#   q          : QSpace to decompose (any rank QD, N symmetries)
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
#   U  bond =  QIndex(left_tag,  '+')   — incoming (enters U from the right)
#   S  left  = QIndex(left_tag,  '-')   — outgoing (leaves S to the left)
#   S  right = QIndex(right_tag, '-')   — outgoing (leaves S to the right)
#   Vd bond  = QIndex(right_tag, '-')   — outgoing (leaves Vd to the left)
#
# Legs of U and Vd that come from the original tensor inherit their QIndex
# properties (itags, lock, plev, green, direction).
#
# Algorithm:
#   1. Assign each leg of q a unique internal tag at lock=1.
#   2. Build fusing isometries aL / aR via getIdentity.
#   3. Contract q_work with aL/aR, yielding the rank-2 bipartition M.
#   4. Per-row SVD on M (sL × sR matrix). Truncate either by cutoff alone
#      or by the Nkeep largest singular values, where missing sectors from
#      M.spaces are treated as explicit zero singular values.
#   5. Build rank-2 U, S, Vd from kept singular values.
#   6. Split fused legs of U and Vd by contracting with conjugate of aL/aR.
#   7. Permute legs to desired order and restore original QIndex properties.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Implement a version that get left_legs by predicates or various keyword arguments
# TODO: Test svd with trunction for QSpace object (This is not rigorous yet)
_svd_sector_qlabels(r, N::Int) = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:N)

_svd_dual_sector(symm, sector) = Tuple(get_dualq(symm[n], sector[n]) for n in 1:length(symm))

_svd_stable_sort_tuple(spaces) = Tuple(sort!(collect(spaces); alg=MergeSort))

_is_zero_array(arr::AbstractArray) = all(iszero, arr)

struct _ReducedSVDCGRBlock{NZ}
    row_index::Int
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

function _svd_abelian_intermediate_q(cgr::CGR{QD}, left_legs_canon) where {QD}
    S = cgr.symm
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

function _get_svd_cgr_split_blocks(cgr::CGR{QD}, left_legs, right_legs) where {QD}
    left_legs_canon = _svd_to_cgridx(cgr, left_legs)
    right_legs_canon = _svd_to_cgridx(cgr, right_legs)
    NZ = length(cgr.qlabels[1])
    BlockInfo = NamedTuple{(:q, :omL, :omR, :coeffs),
        Tuple{NTuple{NZ, Int}, Int, Int, Array{Float64, 3}}}

    if isabelian(cgr.symm)
        q = _svd_abelian_intermediate_q(cgr, left_legs_canon)
        coeffs = reshape(copy(cgr.wmat.data), 1, 1, size(cgr.wmat.data, 2))
        _is_zero_array(coeffs) && return BlockInfo[]
        return [BlockInfo((q=q, omL=1, omR=1, coeffs=coeffs))]
    end

    upsp, dnsp = _svd_cgr_updn(cgr)
    cgtsvd = getNsave_CGTSVD(cgr.symm, upsp, dnsp, left_legs_canon; save=true)

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
        q = zero_qlabels((cgr.symm,))[1]
        om = size(cgr.wmat.data, 1)
        omL, omR = cgtsvd ? (1, om) : (om, 1)
        coeffs = reshape(cgr.wmat.data, omL, omR, size(cgr.wmat.data, 2))
        push!(blocks, BlockInfo((q=q, omL=omL, omR=omR, coeffs=coeffs)))
    end
    return blocks
end

function _reduce_svd_cgr_block(block,
                               row_index::Int,
                               left_spaces,
                               right_spaces;
                               tol::Float64 = 1e-12)
    coeffs = block.coeffs
    left_iso, left_reduced, _ = svd_leg(coeffs, 1; cutoff=tol)
    size(left_iso, 2) == 0 && return nothing

    right_iso, core, _ = svd_leg(left_reduced, 2; cutoff=tol)
    size(right_iso, 2) == 0 && return nothing

    return _ReducedSVDCGRBlock{length(block.q)}(
        row_index, left_spaces, right_spaces, block.q, left_iso, right_iso, core)
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

function _qr_shared_isometry(mats::Vector{<:AbstractMatrix}; tol::Float64 = 1e-12)
    nrows = size(first(mats), 1)
    @assert all(size(mat, 1) == nrows for mat in mats) "_qr_shared_isometry requires a common row dimension"

    if nrows == 1 return [1.0;;], mats end

    concat = hcat(mats...)
    F = qr(concat)
    Q = Matrix(F.Q)
    R = Matrix(F.R)
    row_norms = [norm(@view R[i, :]) for i in 1:size(R, 1)]
    max_norm = maximum(row_norms; init=0.0)
    used = max_norm == 0.0 ? 0 : something(findlast(x -> x > tol * max_norm, row_norms), 0)

    used > 0 || error("_qr_shared_isometry received only unused columns")
    Q = Q[:, 1:used]
    R = R[1:used, :]

    factors = Matrix{Float64}[]
    col = 0
    for mat in mats
        width = size(mat, 2)
        push!(factors, R[:, col+1:col+width])
        col += width
    end
    @assert col == size(concat, 2)

    return Q, factors
end

function _share_svd_row_side_isometries!(blocks::Vector{<:_ReducedSVDCGRBlock},
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
            mats = [
                side === :left ?
                    blocks[i].left_iso :
                    blocks[i].right_iso
                for i in inds
            ]
            common_iso, factors = _qr_shared_isometry(mats; tol=tol)

            for (i, factor) in zip(inds, factors)
                block = blocks[i]
                if side === :left
                    new_core = _contract_om_axis(block.core, factor, 1)
                    blocks[i] = _ReducedSVDCGRBlock{length(block.q)}(
                        block.row_index, block.left_spaces, block.right_spaces,
                        block.q, common_iso, block.right_iso, new_core)
                else
                    new_core = _contract_om_axis(block.core, factor, 2)
                    blocks[i] = _ReducedSVDCGRBlock{length(block.q)}(
                        block.row_index, block.left_spaces, block.right_spaces,
                        block.q, block.left_iso, common_iso, new_core)
                end
            end
        end

        pos = nextpos
    end

    return blocks
end

function _share_svd_row_isometries!(blocks_by_symm,
                                    symm;
                                    tol::Float64 = 1e-12)
    isempty(symm) && return blocks_by_symm
    for n in eachindex(symm)
        isabelian(symm[n]) && continue
        _share_svd_row_side_isometries!(blocks_by_symm[n], :left; tol=tol)
        _share_svd_row_side_isometries!(blocks_by_symm[n], :right; tol=tol)
    end
    return blocks_by_symm
end

function _sort_svd_blocks_by_row!(blocks_by_symm)
    for blocks in blocks_by_symm
        sort!(blocks; by = block -> (block.row_index, block.q), alg=MergeSort)
    end
    return blocks_by_symm
end

function _get_svd_intermediate_qrow_dict(blocks_by_symm::Tuple{Vararg{<:AbstractVector, N}},
                                         nrows::Int) where {N}
    _sort_svd_blocks_by_row!(blocks_by_symm)
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    qrows = Dict{Sector, Vector{Int}}()
    (nrows == 0 || any(isempty, blocks_by_symm)) && return qrows

    qchoices_by_row = ntuple(_ -> [Tuple{Vararg{Int}}[] for _ in 1:nrows], N)
    for n in 1:N
        @assert first(blocks_by_symm[n]).row_index == 1
        @assert last(blocks_by_symm[n]).row_index == nrows
        for block in blocks_by_symm[n]
            qlist = qchoices_by_row[n][block.row_index]
            (isempty(qlist) || qlist[end] != block.q) && push!(qlist, block.q)
        end
        @assert all(!isempty, qchoices_by_row[n])
    end

    for ri in 1:nrows
        choices = ntuple(n -> qchoices_by_row[n][ri], N)
        for sector in Iterators.product(choices...)
            push!(get!(qrows, Tuple(sector), Int[]), ri)
        end
    end

    return qrows
end

function _get_svd_row_spaces(q::QSpace{T, QD, N, RD}, 
    left_legs,
    right_legs) where {T, QD, N, RD}
    row_spaces = [begin
        split_spaces = ntuple(N) do n
            left_legs_canon = _svd_to_cgridx(r.cgrs[n], left_legs)
            right_legs_canon = _svd_to_cgridx(r.cgrs[n], right_legs)
            _svd_cgr_split_spaces(r.cgrs[n], left_legs_canon, right_legs_canon)
        end
        (ntuple(n -> split_spaces[n][1], N), ntuple(n -> split_spaces[n][2], N))
    end for r in q.rows]
    return first.(row_spaces), last.(row_spaces)
end

function _get_svd_intermediate_qrow_equivclasses(
    left_sigs,
    right_sigs,
    qrows::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}},
) where {N}
    Sector = NTuple{N, Tuple{Vararg{Int}}}
    classes_by_sector = Dict{Sector, Vector{Vector{Int}}}()
    isempty(qrows) && return classes_by_sector
    LeftSig = eltype(left_sigs)
    RightSig = eltype(right_sigs)

    for sector in sort!(collect(keys(qrows)); alg=MergeSort)
        rows = sort!(copy(qrows[sector]); alg=MergeSort)
        left_groups = Dict{LeftSig, Vector{Int}}()
        right_groups = Dict{RightSig, Vector{Int}}()

        for ri in rows
            push!(get!(left_groups, left_sigs[ri], Int[]), ri)
            push!(get!(right_groups, right_sigs[ri], Int[]), ri)
        end

        unassigned = Set(rows)
        classes = Vector{Vector{Int}}()
        for seed in rows
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

function _get_svd_intermediate_qrow_equivclasses(
    q::QSpace{T, QD, N, RD},
    left_legs,
    qrows::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}},
) where {T, QD, N, RD}
    left_sigs, right_sigs = _get_svd_row_spaces(q, left_legs)
    return _get_svd_intermediate_qrow_equivclasses(left_sigs, right_sigs, qrows)
end

function _get_svd_row_split_blocks(r::row{T, QD, N, RD},
                                   left_legs,
                                   right_legs,
                                   row_index::Int;
                                   tol::Float64 = 1e-12) where {T, QD, N, RD}
    split_spaces = ntuple(N) do n
        left_legs_canon = _svd_to_cgridx(r.cgrs[n], left_legs)
        right_legs_canon = _svd_to_cgridx(r.cgrs[n], right_legs)
        _svd_cgr_split_spaces(r.cgrs[n], left_legs_canon, right_legs_canon)
    end
    symm_blocks = ntuple(N) do n
        left_spaces_n, right_spaces_n = split_spaces[n]
        reduced = Vector{_ReducedSVDCGRBlock{nzops(r.cgrs[n].symm)}}()
        for block in _get_svd_cgr_split_blocks(r.cgrs[n], left_legs, right_legs)
            reduced_block = _reduce_svd_cgr_block(
                block, row_index, left_spaces_n, right_spaces_n; tol=tol)
            isnothing(reduced_block) || push!(reduced, reduced_block)
        end
        reduced
    end
    any(isempty, symm_blocks) && return nothing
    return symm_blocks
end

function _get_svd_cgt_split_rows(q::QSpace{T, QD, N, RD},
                                 left_legs,
                                 right_legs;
                                 tol::Float64 = 1e-12) where {T, QD, N, RD}
    blocks_by_symm = ntuple(n -> Vector{_ReducedSVDCGRBlock{nzops(q.symm[n])}}(), N)
    for (ri, r) in enumerate(q.rows)
        row_blocks = _get_svd_row_split_blocks(r, left_legs, right_legs, ri; tol=tol)
        isnothing(row_blocks) && continue
        for n in 1:N
            append!(blocks_by_symm[n], row_blocks[n])
        end
    end
    return blocks_by_symm
end

function _build_svd_block_lookup(blocks)
    lookup = Dict{Any, Any}()
    for block in blocks
        key = (block.row_index, block.q)
        @assert !haskey(lookup, key) "duplicate CGTSVD block for row $(block.row_index), q=$(block.q)"
        lookup[key] = block
    end
    return lookup
end

function _preprocess_svd_cgtsvd(q::QSpace{T, QD, N, RD},
                                left_legs;
                                tol::Float64 = 1e-12) where {T, QD, N, RD}
    left_legs_ = _normalize_svd_left_legs(left_legs, QD)
    right_legs_ = [l for l in 1:QD if l ∉ left_legs_]
    left_signatures, right_signatures = 
    _get_svd_row_spaces(q, left_legs_, right_legs_)

    blocks_by_symm = _get_svd_cgt_split_rows(q, left_legs_, right_legs_; tol=tol)
    _share_svd_row_isometries!(blocks_by_symm, q.symm; tol=tol)
    intermediate_qrows = _get_svd_intermediate_qrow_dict(blocks_by_symm, length(q.rows))
    intermediate_qrow_classes = _get_svd_intermediate_qrow_equivclasses(
        left_signatures, right_signatures, intermediate_qrows)
    block_lookup_by_symm = ntuple(n -> _build_svd_block_lookup(blocks_by_symm[n]), N)
    return (
        left_legs = Tuple(left_legs_),
        right_legs = Tuple(right_legs_),
        left_signatures = left_signatures,
        right_signatures = right_signatures,
        blocks_by_symm = blocks_by_symm,
        block_lookup_by_symm = block_lookup_by_symm,
        intermediate_qrows = intermediate_qrows,
        intermediate_qrow_classes = intermediate_qrow_classes,
    )
end

function _svd_expand_core_axis(A::AbstractArray{T}, core::Array{Float64, 3}, axis::Int) where {T}
    omLr, omRr, omM = size(core)
    merged = _contract_om_axis(A, reshape(core, omLr * omRr, omM), axis)
    dims = size(merged)
    return reshape(merged, dims[1:axis-1]..., omLr, omRr, dims[axis+1:end]...)
end

function _svd_row_blocks_for_sector(prep, row_index::Int, sector)
    ntuple(length(sector)) do n
        key = (row_index, sector[n])
        @assert haskey(prep.block_lookup_by_symm[n], key) "missing CGTSVD block for row $row_index, sector $sector, symmetry $n"
        prep.block_lookup_by_symm[n][key]
    end
end

function _svd_row_class_matrix(r::row{T, QD, N, RD},
                               row_blocks,
                               left_legs::NTuple{NL, Int},
                               right_legs::NTuple{NR, Int}) where {T, QD, N, RD, NL, NR}
    data = r.RMT.data
    for n in 1:N
        data = _svd_expand_core_axis(data, row_blocks[n].core, QD + 2n - 1)
    end

    perm = Tuple(vcat(
        collect(left_legs),
        [QD + 2n - 1 for n in 1:N],
        collect(right_legs),
        [QD + 2n for n in 1:N],
    ))
    permed = permutedims(data, perm)
    left_dim = prod(size(r.RMT.data, leg) for leg in left_legs) *
               prod(size(row_blocks[n].left_iso, 2) for n in 1:N)
    right_dim = prod(size(r.RMT.data, leg) for leg in right_legs) *
                prod(size(row_blocks[n].right_iso, 2) for n in 1:N)
    return reshape(permed, left_dim, right_dim)
end

function _svd_ordered_unique_signatures(row_signatures, class_rows::Vector{Int})
    Sig = eltype(row_signatures)
    ordered = Sig[]
    seen = Set{Sig}()
    for ri in class_rows
        sig = row_signatures[ri]
        sig in seen && continue
        push!(seen, sig)
        push!(ordered, sig)
    end
    return ordered
end

function _svd_class_side_infos(q::QSpace{T, QD, N, RD},
                               prep,
                               sector,
                               class_rows::Vector{Int},
                               row_signatures,
                               legs::NTuple{L, Int},
                               side::Symbol) where {T, QD, N, RD, L}
    ordered = _svd_ordered_unique_signatures(row_signatures, class_rows)
    infos = Any[]
    ranges = Dict{eltype(row_signatures), UnitRange{Int}}()
    offset = 0

    for sig in ordered
        ri = first(filter(rj -> row_signatures[rj] == sig, class_rows))
        row_blocks = _svd_row_blocks_for_sector(prep, ri, sector)
        phys_dims = ntuple(i -> size(q.rows[ri].RMT.data, legs[i]), L)
        om_dims = ntuple(n -> size(side === :left ? row_blocks[n].left_iso : row_blocks[n].right_iso, 2), N)
        block_size = prod(phys_dims; init=1) * prod(om_dims; init=1)
        block_range = offset + 1:offset + block_size
        push!(infos, (signature=sig, row_index=ri, phys_dims=phys_dims, om_dims=om_dims, range=block_range))
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

function _build_svd_cgtsvd_class(q::QSpace{T, QD, N, RD},
                                 prep,
                                 sector,
                                 class_rows::Vector{Int},
                                 left_legs::NTuple{NL, Int},
                                 right_legs::NTuple{NR, Int}) where {T, QD, N, RD, NL, NR}
    left_infos, left_ranges, total_left = _svd_class_side_infos(
        q, prep, sector, class_rows, prep.left_signatures, left_legs, :left)
    
    right_infos, right_ranges, total_right = _svd_class_side_infos(
        q, prep, sector, class_rows, prep.right_signatures, right_legs, :right)

    Tmat = promote_type(T, Float64)
    mat = zeros(Tmat, total_left, total_right)
    for ri in class_rows
        row_blocks = _svd_row_blocks_for_sector(prep, ri, sector)
        block_mat = _svd_row_class_matrix(q.rows[ri], row_blocks, left_legs, right_legs)
        lrange = left_ranges[prep.left_signatures[ri]]
        rrange = right_ranges[prep.right_signatures[ri]]
        mat[lrange, rrange] .+= block_mat
    end


    F = svd(mat; full=false)
    dimq = prod(Float64(dimension(q.symm[n], sector[n])) for n in 1:N)
    scale = sqrt(dimq)
    return (
        sector = sector,
        class_rows = class_rows,
        left_infos = left_infos,
        right_infos = right_infos,
        U = F.U .* scale,
        # `get1jtensor` contributes the remaining 1/sqrt(dimq) normalization
        # when the center bridge is materialized as a rank-2 QSpace row.
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
    base = get1jtensor(leginfo{N}(symm, QIndex(left_tag, '-'), bond_splist))
    rows_S = row{Float64, 2, N, 2 + N}[]

    for r in base.rows
        sector = _svd_sector_qlabels(r, N)
        svals = sector_values[sector]
        count = length(svals)
        rmt = LurTensor(reshape(Matrix(Diagonal(svals)), count, count, ones(Int, N)...))
        push!(rows_S, row(r.cgrs, rmt))
    end

    inds_S = (QIndex(left_tag, '+'), QIndex(right_tag, '+'))
    spaces_S = (bond_splist, base.spaces[2])
    return QSpace(symm, rows_S, inds_S, spaces_S)
end

function _assemble_svd_cgtsvd(q::QSpace{T, QD, N, RD},
                              left_legs::NTuple{NL, Int},
                              right_legs::NTuple{NR, Int},
                              left_tag::AbstractString,
                              right_tag::AbstractString,
                              prep;
                              cutoff::Float64 = 1e-12,
                              Nkeep::Union{Nothing, Int} = nothing) where {T, QD, N, RD, NL, NR}
    class_results = Any[]
    for sector in sort!(collect(keys(prep.intermediate_qrow_classes)); alg=MergeSort)
        for class_rows in prep.intermediate_qrow_classes[sector]
            push!(class_results,
                  _build_svd_cgtsvd_class(q, prep, sector, class_rows, left_legs, right_legs))
        end
    end

    keep_per_class = _select_svd_cgtsvd_entries(class_results, cutoff, Nkeep)
    class_ranges, sector_counts, sector_order = _svd_cgtsvd_class_ranges(class_results, keep_per_class)

    Tout = promote_type(T, Float64)
    rows_U = row{Tout, NL + 1, N, NL + 1 + N}[]
    rows_Vd = row{Tout, NR + 1, N, NR + 1 + N}[]

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
        dual_sector = _svd_dual_sector(q.symm, sector)
        sector_count = sector_counts[sector]
        class_range = class_ranges[ci]

        for info in result.left_infos
            row_blocks = _svd_row_blocks_for_sector(prep, info.row_index, sector)
            part = result.U[info.range, keep]
            full = zeros(eltype(result.U), length(info.range), sector_count)
            full[:, class_range] = part
            tmp = reshape(full, info.phys_dims..., info.om_dims..., sector_count)
            perm = Tuple(vcat(collect(1:NL), NL + N + 1, collect(NL+1:NL+N)))
            data = permutedims(tmp, perm)
            rmt_U = LurTensor(data)

            cgrs_U = ntuple(N) do n
                _svd_build_side_cgr(
                    q.symm[n],
                    q.rows[info.row_index].cgrs[n],
                    left_legs,
                    sector[n],
                    false,
                    LurTensor(copy(row_blocks[n].left_iso)),
                )
            end
            push!(rows_U, row(cgrs_U, rmt_U))
        end

        for info in result.right_infos
            row_blocks = _svd_row_blocks_for_sector(prep, info.row_index, sector)
            part = result.Vt[keep, info.range]
            full = zeros(eltype(result.Vt), sector_count, length(info.range))
            full[class_range, :] = part
            rmt_Vd = LurTensor(reshape(full, sector_count, info.phys_dims..., info.om_dims...))

            cgrs_Vd = ntuple(N) do n
                _svd_build_side_cgr(
                    q.symm[n],
                    q.rows[info.row_index].cgrs[n],
                    right_legs,
                    dual_sector[n],
                    true,
                    LurTensor(copy(row_blocks[n].right_iso)),
                )
            end
            push!(rows_Vd, row(cgrs_Vd, rmt_Vd))
        end
    end

    bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (sector, sector_counts[sector]) for sector in sector_order
    ]
    dual_bond_splist = Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}[
        (_svd_dual_sector(q.symm, sector), sector_counts[sector]) for sector in sector_order
    ]
    sort!(dual_bond_splist; by = first, alg=MergeSort)

    inds_U = (ntuple(i -> q.inds[left_legs[i]], NL)..., QIndex(left_tag, '-'))
    inds_Vd = (QIndex(right_tag, '-'), ntuple(i -> q.inds[right_legs[i]], NR)...)

    spaces_U = (ntuple(i -> q.spaces[left_legs[i]], NL)..., bond_splist)
    spaces_Vd = (dual_bond_splist, ntuple(i -> q.spaces[right_legs[i]], NR)...)

    U = QSpace(q.symm, rows_U, inds_U, spaces_U)
    S = _build_svd_cgtsvd_S(q.symm, bond_splist, left_tag, right_tag, sector_values)
    Vd = QSpace(q.symm, rows_Vd, inds_Vd, spaces_Vd)
    return U, S, Vd
end

"""
    LinearAlgebra.svd(q, left_legs, left_tag="svdL", right_tag="svdR"; cutoff=1e-12, Nkeep=nothing)

CGTSVD-based SVD of a `QSpace`, returning `(U, S, Vd)`.

Unlike `svd`, this path uses the direct split-space class SVD assembly and
returns the bond convention requested for CGTSVD:

- `U` bond leg: outgoing `'-'`
- `S` legs: incoming `'+'`, incoming `'+'`
- `Vd` bond leg: outgoing `'-'`
"""
function LinearAlgebra.svd(q::QSpace{T, QD, N, RD},
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

function LinearAlgebra.svd(q::QSpace{T, QD, N, RD},
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

function _select_svd_left_legs(q::QSpace; dir=nothing, itag=nothing, plev=nothing,
                               lock=nothing, rev::Bool=false)
    if isnothing(dir) && isnothing(itag) && isnothing(plev) && isnothing(lock)
        throw(ArgumentError("keyword-based svd requires at least one of dir, itag, plev, or lock"))
    end

    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    rank = length(q.inds)
    1 <= length(legs) <= rank - 1 || throw(ArgumentError("keyword-based svd selected legs $legs, but the number of selected legs must be between 1 and $(rank - 1)"))
    return sort(legs)
end

function svd_old(q::QSpace{T, QD, N, RD},
                           left_legs,
                           left_tag ::AbstractString = "svdL",
                           right_tag::AbstractString = "svdR";
                           cutoff   ::Float64 = 1e-12,
                           Nkeep    ::Union{Nothing, Int} = nothing,
) where {T, QD, N, RD}

    symm = q.symm
    left_legs  = _normalize_svd_left_legs(left_legs, QD)
    right_legs = [l for l in 1:QD if l ∉ left_legs]
    NL, NR = length(left_legs), length(right_legs)
    @assert isnothing(Nkeep) || Nkeep >= 0 "Nkeep must be non-negative"

    # ── Step 1: stamp unique internal tags on every leg (lock=1) ─────────────
    internal_tags = ntuple(l -> "__svd_leg_$(l)__", QD)
    q_work = QSpace(q.symm, q.rows,
        ntuple(l -> QIndex(internal_tags[l], q.inds[l].dir,
                           q.inds[l].plev, 1, q.inds[l].green), QD),
        q.spaces)  # reuse existing spaces since rows unchanged

    # ── Step 2: build fusing isometries ──────────────────────────────────────
    left_cur  = sort(left_legs)
    right_cur = sort(right_legs)

    # getIdentity((q, leg_idx)...; itag=tag) returns legs directly contractable
    # with the selected legs of q, plus one fused output leg.
    # We pass legs in sorted order so that their ordering is consistent with the
    # free1 ordering produced by the upcoming contractions (which also sorts by index).
    aL = getIdentity(((q_work, l) for l in left_cur)...;  itag=left_tag)
    aR = getIdentity(((q_work, l) for l in right_cur)...; itag=right_tag)

    # ── Step 3: contract q_work → rank-2 bipartition M ───────────────────────
    # Contract q_work over its left-side legs with aL's first NL legs.
    # Result legs: (remaining q_work legs in original order, fused aL leg)
    # The remaining legs from q_work are exactly right_cur, landing at positions
    # 1..NR in the result (free1 = right_cur legs), then fused aL at NR+1.
    M = contract(q_work, Tuple(left_cur), aL, Tuple(1:NL); reduce_lock=false)
    # Now contract M's first NR legs (right_cur legs) with aR's first NR legs.
    M = contract(M, Tuple(1:NR), aR, Tuple(1:NR); reduce_lock=false)
    # Result M is rank-2:
    #   M.inds[1] = aL fused leg  (left_tag,  '-')
    #   M.inds[2] = aR fused leg  (right_tag, '-')
    # M.rows[r].RMT.data has shape (sL, sR, 1,...,1).

    # ── Step 4: collect singular values ───────────────────────────────────────
    # For each row of M the RMT has shape (sL, sR, 1,...,1) since rank-2 CGTs
    # have no outer multiplicity. We SVD the (sL × sR) matrix directly.
    # By default, truncation follows the existing cutoff logic.
    # When Nkeep is provided instead, we keep the Nkeep largest singular values
    # globally, ignore degeneracy, and count sectors missing from M.rows as
    # explicit zero singular values using the space lists of M.
    sv_global_max = 0.0
    all_sv_entries = Tuple{Float64, Int, Int}[]  # (sv, row_idx, sv_idx)
    selected_entries = Tuple{Float64, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]  # (sv, row_idx, sector, sv_idx)
    row_sector_map = Dict{Int, NTuple{N, Tuple{Vararg{Int}}}}()
    left_space_dims = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()
    right_space_dims = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (sector, dim) in M.spaces[1]
        left_space_dims[sector] = dim
    end
    for (sector, dim) in M.spaces[2]
        right_space_dims[sector] = dim
    end

    for (ri, r) in enumerate(M.rows)
        rmt    = r.RMT.data                                # (sL, sR, 1,...,1)
        sL, sR = size(rmt, 1), size(rmt, 2)
        sector = _svd_sector_qlabels(r, N)
        row_sector_map[ri] = sector

        # Treat RMT as (sL × sR) matrix.
        mat = reshape(rmt, sL, sR)
        F   = svd(mat)
        isempty(F.S) && continue

        sv_global_max = max(sv_global_max, F.S[1])
        for j in eachindex(F.S)
            push!(all_sv_entries, (F.S[j], ri, j))
        end
    end

    if isnothing(Nkeep)
        # Sort descending by singular value.
        sort!(all_sv_entries; by = x -> -x[1])

        # Apply absolute cutoff.
        thresh = cutoff * sv_global_max
        filter!(x -> x[1] > thresh, all_sv_entries)

        for (sv, ri, j) in all_sv_entries
            push!(selected_entries, (sv, ri, row_sector_map[ri], j))
        end
    else
        for (sv, ri, j) in all_sv_entries
            push!(selected_entries, (sv, ri, row_sector_map[ri], j))
        end

        present_sectors = Set(values(row_sector_map))
        for (sector, dimL) in M.spaces[1]
            sector in present_sectors && continue
            dimR = get(right_space_dims, _svd_dual_sector(symm, sector), 0)
            for j in 1:min(dimL, dimR)
                push!(selected_entries, (0.0, 0, sector, j))
            end
        end

        sort!(selected_entries; by = x -> -x[1])
        resize!(selected_entries, min(Nkeep, length(selected_entries)))
    end

    # Group kept singular-value indices by source row / missing sector.
    keep_per_row = Dict{Int, Vector{Int}}()
    keep_missing = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()
    selected_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()
    for (_, ri, sector, j) in selected_entries
        selected_counts[sector] = get(selected_counts, sector, 0) + 1
        if ri == 0
            keep_missing[sector] = get(keep_missing, sector, 0) + 1
        else
            push!(get!(keep_per_row, ri, Int[]), j)
        end
    end

    # ── Step 5: build rank-2 U, S, Vd QSpaces ─────────────────────────────────
    # For each kept sector (RMT shape is (sL, sR, 1,...,1)):
    #   U  RMT: (sL, chi, 1,...,1)     legs: (left_tag '-', left_tag '+')
    #   S  RMT: (chi, chi, 1,...,1)    legs: (left_tag '-', right_tag '-')
    #   Vd RMT: (chi, sR, 1,...,1)     legs: (right_tag '-', right_tag '+')
    # S inherits M's qlabels and legdir (both legs outgoing).
    # U and Vd have identity CGT parts: legdir=(1,1), same qlabel on both legs.
    # wmat = sqrt(dim) to satisfy CGT part = identity matrix
    rows_U  = row{Float64, 2, N, 2 + N}[]
    rows_S  = row{Float64, 2, N, 2 + N}[]
    rows_Vd = row{Float64, 2, N, 2 + N}[]

    for (ri, keep_idxs) in sort(collect(keep_per_row); by = x -> x[1])
        r        = M.rows[ri]
        rmt      = r.RMT.data
        sL, sR   = size(rmt, 1), size(rmt, 2)
        mat      = reshape(rmt, sL, sR)
        F        = svd(mat)
        k        = sort(keep_idxs)
        chi      = length(k)

        Umat  = F.U[:, k]                              # (sL, chi)
        Svals = F.S[k]                                 # (chi,)
        Vtmat = F.Vt[k, :]                             # (chi, sR)

        rmt_U  = LurTensor(reshape(Umat,                       sL,  chi, ones(Int, N)...))
        rmt_S  = LurTensor(reshape(Matrix(Diagonal(Svals)),    chi, chi, ones(Int, N)...))
        rmt_Vd = LurTensor(reshape(transpose(Vtmat),           sR, chi,  ones(Int, N)...))

        # U: identity CGT on left_ql, legdir=(1,1)
        cgrs_U  = ntuple(N) do n
            cgr_M   = r.cgrs[n]
            left_ql = cgr_M.qlabels[cgr_M.cgp[1]]
            dim_n   = dimension(symm[n], left_ql)
            wmat_n  = LurTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (left_ql, left_ql), wmat_n, (2, 1), (1, 1))
        end
        # S: same qlabels and legdir as M (both legs outgoing)
        cgrs_S  = ntuple(N) do n
            cgr_M    = r.cgrs[n]
            left_ql  = cgr_M.qlabels[cgr_M.cgp[1]]
            dim_n    = dimension(symm[n], left_ql)
            wmat_n   = LurTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], cgr_M.qlabels, wmat_n, cgr_M.cgp, (0, 2))
        end
        # Vd: identity CGT on right_ql, legdir=(1,1)
        cgrs_Vd = ntuple(N) do n
            cgr_M    = r.cgrs[n]
            right_ql = cgr_M.qlabels[cgr_M.cgp[2]]
            dim_n    = dimension(symm[n], right_ql)
            wmat_n   = LurTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (right_ql, right_ql), wmat_n, (2, 1), (1, 1))
        end

        push!(rows_U,  row(cgrs_U,  rmt_U))
        push!(rows_S,  row(cgrs_S,  rmt_S))
        push!(rows_Vd, row(cgrs_Vd, rmt_Vd))
    end

    for (sector, chi) in sort(collect(keep_missing); by = x -> x[1])
        sL = left_space_dims[sector]
        right_sector = _svd_dual_sector(symm, sector)
        sR = right_space_dims[right_sector]

        Umat = Matrix{Float64}(I, sL, sL)[:, 1:chi]
        Vmat = Matrix{Float64}(I, sR, sR)[:, 1:chi]

        rmt_U = LurTensor(reshape(Umat, sL, chi, ones(Int, N)...))
        rmt_Vd = LurTensor(reshape(Vmat, sR, chi, ones(Int, N)...))

        cgrs_U = ntuple(N) do n
            left_ql = sector[n]
            dim_n = dimension(symm[n], left_ql)
            wmat_n = LurTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (left_ql, left_ql), wmat_n, (2, 1), (1, 1))
        end

        cgrs_Vd = ntuple(N) do n
            right_ql = right_sector[n]
            dim_n = dimension(symm[n], right_ql)
            wmat_n = LurTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (right_ql, right_ql), wmat_n, (2, 1), (1, 1))
        end

        push!(rows_U, row(cgrs_U, rmt_U))
        push!(rows_Vd, row(cgrs_Vd, rmt_Vd))
    end

    bond_splist = eltype(M.spaces[1])[]
    for (qlabels, _) in M.spaces[1]
        count = get(selected_counts, qlabels, 0)
        count == 0 && continue
        push!(bond_splist, (qlabels, count))
    end
    dual_bond_splist = eltype(M.spaces[2])[
        (_svd_dual_sector(symm, qlabels), count) for (qlabels, count) in bond_splist
    ]
    sort!(dual_bond_splist; by = x -> x[1])

    inds_U  = (QIndex(left_tag,  '-'), QIndex(left_tag,  '+'))
    inds_S  = (QIndex(left_tag,  '-'), QIndex(right_tag, '-'))
    inds_Vd = (QIndex(right_tag, '-'), QIndex(right_tag, '+'))
    
    spaces_U  = (M.spaces[1], bond_splist)
    spaces_S  = (bond_splist, dual_bond_splist)
    spaces_Vd = (M.spaces[2], dual_bond_splist)

    U_rank2  = QSpace(symm, rows_U,  inds_U,  spaces_U)
    S        = QSpace(symm, rows_S,  inds_S,  spaces_S)
    Vd_rank2 = QSpace(symm, rows_Vd, inds_Vd, spaces_Vd)

    # ── Step 6: split fused legs of U and Vd ──────────────────────────────────
    # Contract U_rank2's fused left leg (leg 1) with aL's fused output leg (leg NL+1)
    # to recover the NL original left legs.
    # Contract Vd_rank2's fused right leg (leg 2) with aR's fused output leg (leg NR+1)
    # to recover the NR original right legs.
    # (QIndex direction matching is not enforced by contract; legs are given explicitly.)
    U_split  = contract(U_rank2,  (1,),     aL', (NL + 1,); reduce_lock=false)
    Vd_split = contract(Vd_rank2, (1,),     aR', (NR + 1,); reduce_lock=false)
    # After splitting:
    #   U_split  legs: [bond (left_tag '+'), aL split legs 1..NL]
    #   Vd_split legs: [Vd bond (right_tag '-'), aR split legs 1..NR]

    # ── Step 7: permute legs to desired order and restore original QIndex properties ──
    # Desired order:
    #   U  : (left_legs[1], ..., left_legs[NL], bond)
    #   Vd : (bond, right_legs[1], ..., right_legs[NR])
    # After splitting, legs have internal tags like "__svd_leg_{orig}__"
    # and the bond leg has left_tag or right_tag.
    
    # Build permutation for U_split: desired order is (left_legs..., bond)
    u_perm = Int[]
    for orig in left_legs
        push!(u_perm, findleg(U_split, idx -> idx.itags == internal_tags[orig]))
    end
    push!(u_perm, findleg(U_split, idx -> idx.itags == left_tag))  # bond leg at the end

    # Build permutation for Vd_split: desired order is (bond, right_legs...)
    vd_perm = Int[]
    push!(vd_perm, findleg(Vd_split, idx -> idx.itags == right_tag))  # bond leg first
    for orig in right_legs
        push!(vd_perm, findleg(Vd_split, idx -> idx.itags == internal_tags[orig]))
    end
    
    # Apply permutations
    U_final  = permutedims(U_split, Tuple(u_perm))
    Vd_final = permutedims(Vd_split, Tuple(vd_perm))
    
    # Restore original QIndex properties (itags, lock, plev, green, dir) for non-bond legs
    # U legs 1:NL inherit from original left_legs, leg NL+1 is the bond
    u_inds_final = ntuple(NL + 1) do i
        if i <= NL
            orig = left_legs[i]
            q.inds[orig]  # inherit all properties from original
        else
            QIndex(left_tag, '+')  # bond leg: incoming into U
        end
    end
    
    # Vd leg 1 is the bond, legs 2:NR+1 inherit from original right_legs
    vd_inds_final = ntuple(NR + 1) do i
        if i == 1
            QIndex(right_tag, '+')  # bond leg: outgoing from Vd
        else
            orig = right_legs[i - 1]
            q.inds[orig]  # inherit all properties from original
        end
    end
    
    # Reconstruct QSpaces with final indices (spaces already permuted correctly)
    U  = QSpace(symm, U_final.rows,  u_inds_final,  U_final.spaces)
    Vd = QSpace(symm, Vd_final.rows, vd_inds_final, Vd_final.spaces)
    
    return U, S, Vd
end

function svd_old(q::QSpace{T, QD, N, RD},
                           left_tag::AbstractString="svdL",
                           right_tag::AbstractString="svdR";
                           dir=nothing,
                           itag=nothing,
                           plev=nothing,
                           lock=nothing,
                           rev::Bool=false,
                           cutoff::Float64=1e-12,
                           Nkeep::Union{Nothing, Int}=nothing,
) where {T, QD, N, RD}
    left_legs = _select_svd_left_legs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return svd_old(q, left_legs, left_tag, right_tag; cutoff=cutoff, Nkeep=Nkeep)
end

