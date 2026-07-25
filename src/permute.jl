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

function _permuted_sector_wmat(symm_type,
                               qlabels::NTuple{QD, NTuple{NZ, Int}},
                               wmat::AbstractMatrix{Float64},
                               cgp::NTuple{QD, Int},
                               legdir::Tuple{Int, Int},
                               perm::NTuple{QD, Int}) where {QD, NZ}
    new_cgp = Tuple(cgp[perm[l]] for l in 1:QD)
    
    # Need to reorder stored qlabels and apply CGTperm transformation
    m, k = legdir
    reorder = _compute_reorder_permutation(qlabels, new_cgp, legdir)

    upsp = Tuple(qlabels[i] for i in 1:m)    # incoming qlabels, already sorted
    dnsp = Tuple(qlabels[m+i] for i in 1:k)  # outgoing qlabels, already sorted
    
    # New qlabels after reordering
    new_qlabels = Tuple(qlabels[reorder[i]] for i in 1:QD)
    # The permutation 'reorder' should permute only same qlabels
    @assert new_qlabels == qlabels

    cgtperm_obj = getNsave_CGTperm(symm_type, upsp, dnsp, reorder)

    if isnothing(cgtperm_obj)
        # Permutation is identity, or symmetry is Abelian
        return wmat
    end
    
    # Apply CGTperm. CGTperm transforms from old OM basis to new OM basis
    return cgtperm_obj.perm_arr * wmat
end

function _permute_sector_wmat(q::AbstractTLArray{T, QD, N, RD}, sector_index::Int,
                              perm::NTuple{QD, Int}, n::Int, symm) where {T, QD, N, RD}
    qlabels, cgp, legdir = _sector_cgt_metadata(q, sector_index, n)
    return _permuted_sector_wmat(symm[n], qlabels, sector_wmat(q, sector_index, n),
                                 cgp, legdir, perm)
end

function _permute_sector_wmat(q::AbstractTLArray{T, QD, N, RD, QT, PS}, sector_index::Int,
                              perm::NTuple{QD, Int}, ::Val{n}) where {T, QD, N, RD, QT, PS, n}
    qlabels, cgp, legdir = _sector_cgt_metadata(q, sector_index, n)
    return _permuted_sector_wmat(product_symms(PS)[n], qlabels,
                                 sector_wmat(q, sector_index, Val(n)),
                                 cgp, legdir, perm)
end

Base.permutedims(q::AbstractTLArray, perm) = _view_permutedims(q, perm)
