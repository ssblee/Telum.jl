# ─── permutedims ─────────────────────────────────────────────────────────
#
# Permute the legs of a QSpace object.
#
# Arguments:
#   q    : QSpace to permute
#   perm : permutation tuple/vector of length QD
#          perm[new_pos] = old_pos means the leg at old_pos moves to new_pos
#
# Returns: new QSpace with permuted legs
#
# Algorithm:
#   1. Permute QIndex tuple according to perm.
#   2. For each row, permute CGRs:
#      - Update cgp: new_cgp[new_leg] = old_cgp[perm[new_leg]]
#      - Check if _check_cgr_qlabel_order would pass with the new cgp
#      - If not, reorder stored qlabels and apply CGTperm transformation to wmat
#   3. Permute RMT first QD dimensions according to perm.
#   4. Assemble and return new QSpace.
# ─────────────────────────────────────────────────────────────────────────────

function _reorder_perm_part(qlabels::NTuple{N, NTuple{NZ, Int}},
    cgp_part::NTuple{N, Int}) where {NZ, N}

    perm = zeros(Int, N)
    st = 1
    while st <= N
        ed = st
        while ed <= N && qlabels[ed] == qlabels[st] ed += 1 end
        perm[st:ed-1] = sortperm(collect(cgp_part[st:ed-1])) .+ (st-1)
        st = ed
    end
    return perm
end

# Compute the reordering permutation for stored positions to satisfy the order constraint.
# Returns reorder where reorder[new_stored_pos] = old_stored_pos.
function _compute_reorder_permutation(qlabels::NTuple{QD, NTuple{NZ, Int}}, 
    cgp::NTuple{QD, Int}, 
    legdir::Tuple{Int, Int}) where {NZ, QD}

    m, k = legdir
    # Build cgp_inv: cgp_inv[stored_pos] = physical_leg
    cgp_inv = zeros(Int, QD)
    for l in 1:QD cgp_inv[cgp[l]] = l end

    # For each direction group, sort by (qlabel, physical_leg) to satisfy constraint.
    # After sorting, same qlabels will have physical legs in increasing order.
    
    # Incoming: positions 1:m
    up_perm = _reorder_perm_part(qlabels[1:m], Tuple(cgp_inv[1:m]))
    # Outgoing: positions m+1:m+k
    dn_perm = _reorder_perm_part(qlabels[m+1:m+k], Tuple(cgp_inv[m+1:m+k]))
    
    # reorder[new] = old: new stored position gets qlabel from old position
    reorder = (up_perm..., (dn_perm .+ m)...)
    
    return reorder
end

# Permute a single CGR according to the physical leg permutation perm.
# Returns a new CGR with updated cgp (and possibly reordered qlabels + transformed wmat).
function _permute_cgr(cgr::CGR{QD, NZ}, 
    perm::NTuple{QD, Int}, 
    symm_type) where {QD, NZ}

    # New cgp: new_cgp[new_leg] = old_cgp[perm[new_leg]]
    new_cgp = Tuple(cgr.cgp[perm[l]] for l in 1:QD)
    
    # Need to reorder stored qlabels and apply CGTperm transformation
    m, k = cgr.legdir
    reorder = _compute_reorder_permutation(cgr.qlabels, new_cgp, cgr.legdir)

    upsp = Tuple(cgr.qlabels[i] for i in 1:m)    # incoming qlabels, already sorted
    dnsp = Tuple(cgr.qlabels[m+i] for i in 1:k)  # outgoing qlabels, already sorted
    
    # New qlabels after reordering
    new_qlabels = Tuple(cgr.qlabels[reorder[i]] for i in 1:QD)
    # The permutation 'reorder' should permute only same qlabels
    @assert new_qlabels == cgr.qlabels

    cgtperm_obj = getNsave_CGTperm(symm_type, upsp, dnsp, reorder)

    # Update cgp: new_cgp[leg] pointed to old stored position,
    # after reorder that qlabel is at reorder_inv[old_pos]
    inv_reorder = invperm(reorder)
    final_cgp = Tuple(inv_reorder[new_cgp[l]] for l in 1:QD)

    if isnothing(cgtperm_obj)
        # Permutation is identity, or symmetry is Abelian
        return CGR(cgr.symm, cgr.qlabels, cgr.wmat, final_cgp, cgr.legdir)
    end
    
    # Apply CGTperm. CGTperm transforms from old OM basis to new OM basis
    new_wmat_data = cgtperm_obj.perm_arr * cgr.wmat.data
    new_wmat = QTensor(new_wmat_data)
    
    return CGR(cgr.symm, cgr.qlabels, new_wmat, final_cgp, cgr.legdir)
end

# Permute a row: permute CGRs and RMT
function _permute_row(r::row{T, QD, N, RD}, perm::NTuple{QD, Int}, symm) where {T, QD, N, RD}
    # Permute each CGR
    new_cgrs = ntuple(n -> _permute_cgr(r.cgrs[n], perm, symm[n]), N)
    
    # Permute RMT: first QD dimensions according to perm, keep OM dimensions at the end
    rmt_perm = (perm..., ntuple(n -> QD + n, N)...)
    new_rmt_data = permutedims(r.RMT.data, rmt_perm)
    new_rmt = QTensor(new_rmt_data)
    
    return row(new_cgrs, new_rmt)
end

"""
    permutedims(q::QSpace, perm)

Permute the legs of a QSpace object.

# Arguments
- `q`: QSpace to permute
- `perm`: permutation (tuple or vector) of length equal to the rank of q.
          `perm[new_pos] = old_pos` means the leg at `old_pos` moves to `new_pos`.

# Returns
New QSpace with permuted legs.
"""
function Base.permutedims(q::QSpace{T, QD, N, RD}, perm) where {T, QD, N, RD}
    perm = Tuple(perm)
    @assert length(perm) == QD "permutation length $(length(perm)) != QSpace rank $QD"
    @assert sort(collect(perm)) == collect(1:QD) "perm must be a valid permutation of 1:$QD"
    
    # 1. Permute QIndex tuple
    new_inds = Tuple(q.inds[perm[l]] for l in 1:QD)
    
    # 2. Permute spaces tuple
    new_spaces = Tuple(q.spaces[perm[l]] for l in 1:QD)
    
    # 3 & 4. Permute each row (CGRs and RMT)
    new_rows = row{T, QD, N, RD}[_permute_row(r, perm, q.symm) for r in q.rows]
    
    # 5. Assemble and return new QSpace with precomputed spaces
    return QSpace(q.symm, new_rows, new_inds, new_spaces)
end
