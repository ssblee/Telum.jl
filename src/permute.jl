# ─── permutedims ─────────────────────────────────────────────────────────
#
# Permute the legs of a TLArray object.
#
# Arguments:
#   q    : TLArray to permute
#   perm : permutation tuple/vector of length QD
#          perm[new_pos] = old_pos means the leg at old_pos moves to new_pos
#
# Returns: new TLArray with permuted legs
#
# Algorithm:
#   1. Permute TLIndex tuple according to perm.
#   2. For each sector, permute CGRs:
#      - Update cgp: new_cgp[new_leg] = old_cgp[perm[new_leg]]
#      - Check if _check_cgr_qlabel_order would pass with the new cgp
#      - If not, reorder stored qlabels and apply CGTperm transformation to wmat
#   3. Permute RMT first QD dimensions according to perm.
#   4. Assemble and return new TLArray.
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
        return CGR(symm(cgr), cgr.qlabels, cgr.wmat, final_cgp, cgr.legdir)
    end
    
    # Apply CGTperm. CGTperm transforms from old OM basis to new OM basis
    new_wmat_data = cgtperm_obj.perm_arr * cgr.wmat.data
    new_wmat = LurTensor(new_wmat_data)
    
    return CGR(symm(cgr), cgr.qlabels, new_wmat, final_cgp, cgr.legdir)
end

function _permute_sector_wmat(q::TLArray{T, QD, N, RD}, sector_index::Int,
                              perm::NTuple{QD, Int}, n::Int, symm) where {T, QD, N, RD}
    qlabels, cgp, legdir = _sector_cgr_metadata(q, sector_index, n)
    cgr = CGR(symm[n], qlabels, sector_wmat(q, sector_index, n), cgp, legdir)
    return _permute_cgr(cgr, perm, symm[n]).wmat
end

function _permute_sector_rmt(q::TLArray{T, QD, N, RD}, sector_index::Int,
                             perm::NTuple{QD, Int}) where {T, QD, N, RD}
    rmt_perm = (perm..., ntuple(n -> QD + n, N)...)
    return LurTensor(permutedims(sector_rmt(q, sector_index).data, rmt_perm))
end

function _permute_sector_wmats(q::TLArray{T, QD, N}, perm::NTuple{QD, Int}, symm) where {T, QD, N}
    return ntuple(Val(N)) do n
        src = q.wmats[n]
        out = similar(src, nsectors(q))
        for sector_index in 1:nsectors(q)
            out[sector_index] = _permute_sector_wmat(q, sector_index, perm, n, symm)
        end
        out
    end
end

function _permute_sector_rmts(q::TLArray{T, QD, N, RD}, perm::NTuple{QD, Int}) where {T, QD, N, RD}
    out = similar(q.RMTs, nsectors(q))
    for sector_index in 1:nsectors(q)
        out[sector_index] = _permute_sector_rmt(q, sector_index, perm)
    end
    return out
end

"""
    permutedims(q::TLArray, perm)

Permute the legs of a TLArray object.

# Arguments
- `q`: TLArray to permute
- `perm`: permutation (tuple or vector) of length equal to the rank of q.
          `perm[new_pos] = old_pos` means the leg at `old_pos` moves to `new_pos`.

# Returns
New TLArray with permuted legs.
"""
function Base.permutedims(q::TLArray{T, QD, N, RD}, perm) where {T, QD, N, RD}
    perm = Tuple(perm)
    @assert length(perm) == QD "permutation length $(length(perm)) != TLArray rank $QD"
    @assert sort(collect(perm)) == collect(1:QD) "perm must be a valid permutation of 1:$QD"
    
    # 1. Permute TLIndex tuple
    new_inds = Tuple(q.inds[perm[l]] for l in 1:QD)
    
    # 2. Permute spaces tuple
    new_spaces = Tuple(q.spaces[perm[l]] for l in 1:QD)
    
    # 3 & 4. Permute each sector (CGR metadata and RMT)
    new_qlabels = copy(q.qlabels[:, collect(perm)])
    new_wmats = _permute_sector_wmats(q, perm, symm(q))
    new_RMTs = _permute_sector_rmts(q, perm)
    
    # 5. Assemble and return new TLArray with precomputed spaces
    return _field_tlarray(symm(q), new_qlabels, new_wmats, new_RMTs, new_inds, new_spaces)
end
