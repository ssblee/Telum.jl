# N: the number of symmetries
# splist: A list of (symmetry sector qlabels, RMT dim)
struct leginfo{N}
    symm::NTuple{N, Any} # symmetry tuple (same for all legs)
    ind::QIndex # Corresponding QIndex object
    splist::Vector{Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}}
end

# Use precomputed spaces from QSpace
function leginfo(q::QSpace{T, QD, N}, i::Int) where {T, QD, N}
    return leginfo{N}(q.symm, q.inds[i], q.spaces[i])
end

# Variadic entry point: accepts multiple (QSpace, Int) pairs as positional arguments
# Keyword arguments control the fused output leg's QIndex properties
function getIdentity(legs::Vararg{Tuple{QSpace, Int}};
                     itag::AbstractString="", plev::Int=0, lock::Int=0)
    leginfos = Tuple(leginfo(q, i) for (q, i) in legs)
    return getIdentity(leginfos; itag=itag, plev=plev, lock=lock)
end
    
# For abelian symmetries: unique outcome is the sum of all qlabels; OM is always 1
# Returns Vector of (qlabel, om) pairs
function combine_qlabels(::Type{S}, 
    qlabels_per_leg::NTuple{D, NTuple{1, Int}}) where {S <: AbelianSymm, D}
    result = qlabels_per_leg[1][1]
    for d in 2:D
        result = add_qn(S, result, qlabels_per_leg[d][1])
    end
    return Tuple{NTuple{1, Int}, Int}[((result,), 1)]
end

# For non-abelian symmetries: use ValidOuts + OMList to enumerate fused irreps with OMs
# Returns Vector of (qlabel, om) pairs
function combine_qlabels(::Type{S}, 
    qlabels_per_leg::NTuple{D, NTuple{NZ, Int}}) where {S <: NonabelianSymm, D, NZ}
    incom = Tuple(sort(collect(qlabels_per_leg)))
    vo = getNsave_validout(S, incom)
    return Tuple{NTuple{NZ, Int}, Int}[
        (q, getNsave_omlist(S, incom, q).totalOM)
        for q in vo.out_spaces
    ]
end

function getIdentity(leginfos::NTuple{D, leginfo{N}};
                     itag::AbstractString="", plev::Int=0, lock::Int=0) where {D, N}

    for i in 1:D-1 @assert leginfos[i].symm == leginfos[i+1].symm end
    symm = leginfos[1].symm

    # For incoming legs (dir == '+'), dualize every qlabel so that
    # the fusion rule sees all legs as outgoing (charge conservation: sum = 0).
    leginfos_adj = Tuple(
        if leginfos[d].ind.dir == '+'
            ET = eltype(leginfos[d].splist)
            new_splist = ET[(Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N), dim)
                            for (qlabels, dim) in leginfos[d].splist]
            sort!(new_splist; by = x -> x[1])
            leginfo{N}(leginfos[d].symm, leginfos[d].ind, new_splist)
        else
            leginfos[d]
        end
        for d in 1:D
    )

    # (row indices into each leginfo's splist..., total RMT dim, oms...,
    # starting index, ending index along fused axis)
    merged_info = Dict{NTuple{N, Tuple{Vararg{Int}}},
    Vector{NTuple{D+N+3, Int}}}()

    nrows = Tuple(length(info.splist) for info in leginfos_adj)

    for i in CartesianIndices(nrows)
        I = i.I
        RMT_dims = Tuple(leginfos_adj[d].splist[I[d]][2] for d in 1:D)
        RMT_dim = prod(RMT_dims)
        qlabels = Tuple(leginfos_adj[d].splist[I[d]][1] for d in 1:D)

        # For each symmetry n, fuse the D qlabels to get all valid (qlabel, om) pairs
        outcomes_per_symm = Tuple(
            combine_qlabels(symm[n], Tuple(qlabels[d][n] for d in 1:D))
            for n in 1:N
        )
        num_outcomes = Tuple(length(outcomes_per_symm[n]) for n in 1:N)
        for j in CartesianIndices(num_outcomes)
            J = j.I
            fused_qlabels = Tuple(outcomes_per_symm[n][J[n]][1] for n in 1:N)
            oms = Tuple(outcomes_per_symm[n][J[n]][2] for n in 1:N)
            sti = 1
            if !haskey(merged_info, fused_qlabels)
                merged_info[fused_qlabels] = Vector{NTuple{D+N+3, Int}}()
            else sti = merged_info[fused_qlabels][end][end] + 1 end
            total_dim = RMT_dim * prod(oms)
            entry = (I..., RMT_dim, oms..., sti, sti+total_dim-1)
            push!(merged_info[fused_qlabels], entry)
        end
    end

    # Build rows of the internal fused QSpace.
    # Before the final fixup for originally-incoming legs, the tensor has
    # D selected legs + 1 fused output leg, where selected incoming legs have
    # been dualized so every selected leg can be fused as incoming.
    # RMT shape: (rmts_dims..., space_cnt).
    # For each entry, the block [:,...,:, sti:edi] is an identity matrix
    # (repeated prod_oms times along the outer-multiplicity sub-blocks).
    # Each CGR has a D+1-leg qlabel tuple and an OM×OM identity wmat.
    rows = row{Float64, D+1, N, D+1+N}[]

    for (fused_qlabels, entries) in merged_info
        #println(fused_qlabels)
        space_cnt = entries[end][end]  # last edi across all entries

        # 1. Scale by sqrt of total outgoing CGT dimension
        cgt_dim_out = prod(dimension(symm[n], fused_qlabels[n]) for n in 1:N)

        for entry in entries
            #println("  ", entry)
            orig_ind  = entry[1:D]
            RMT_dim   = entry[D+1]
            oms       = entry[D+2:D+N+1]
            sti       = entry[end-1]
            edi       = entry[end]
            prod_oms  = prod(oms)

            # Dimensions of RMTs of input legs at this sector
            rmts_dims = Tuple(leginfos_adj[d].splist[orig_ind[d]][2] for d in 1:D)

            # Build RMT: zeros everywhere, identity blocks at [:,...,:, sti:edi]
            # The block is laid out as prod_oms copies of a RMT_dim×RMT_dim identity.
            RMT_data = zeros(Float64, rmts_dims..., space_cnt, prod_oms)
            id_block  = reshape(Matrix{Float64}(I, RMT_dim, RMT_dim), rmts_dims..., RMT_dim)
            for m in 1:prod_oms
                cs = sti + (m - 1) * RMT_dim
                RMT_data[ntuple(_ -> (:), D)..., cs:cs+RMT_dim-1, m] = id_block
            end
            RMT_data .*= sqrt(cgt_dim_out)
            RMT_t = QTensor(reshape(RMT_data, rmts_dims..., space_cnt, oms...))

            # Build one CGR per symmetry
            cgrs_list = CGR{D+1}[]
            for n in 1:N
                input_qls    = Tuple(leginfos_adj[d].splist[orig_ind[d]][1][n] for d in 1:D)
                perm         = sortperm(collect(input_qls))
                inv_perm     = invperm(perm)
                cgr_qlabels  = (input_qls[perm]..., fused_qlabels[n])
                om_n         = oms[n]
                wmat         = QTensor(Matrix{Float64}(I, om_n, om_n))
                cgp          = (inv_perm..., D+1)
                push!(cgrs_list, CGR(symm[n], cgr_qlabels, wmat, cgp, (D, 1)))
            end

            push!(rows, row(Tuple(cgrs_list), RMT_t))
        end
    end

    # Assemble an internal QSpace whose selected legs are ready for fusion.
    # For originally-incoming legs we expose the dualized leg with green=true so
    # we can contract a 1j tensor and recover a directly-contractable external leg.
    fused_ind = QIndex(itag, '-', plev, lock)
    inds = (ntuple(d -> begin
        idx = leginfos[d].ind.dir == '+' ? change_green(leginfos[d].ind) : leginfos[d].ind
        to_incoming(idx)
    end, D)..., fused_ind)
    
    # Build spaces tuple: first D legs use the adjusted spaces, last leg from merged_info
    # Fused leg space: for each fused_qlabel, RMT dim = total dimension (last edi)
    fused_splist = Vector{Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}}()
    for (fused_qlabels, entries) in merged_info
        space_dim = entries[end][end]  # last edi = total RMT dimension for this sector
        push!(fused_splist, (fused_qlabels, space_dim))
    end
    sort!(fused_splist; by = x -> x[1])  # sort by qlabels for consistency
    
    spaces = (ntuple(d -> leginfos_adj[d].splist, D)..., fused_splist)

    q = QSpace(symm, rows, inds, spaces)

    # For an originally-incoming selected leg, attach a 1j tensor so the returned
    # leg is directly contractable with the original tensor leg.
    for d in 1:D
        leginfos[d].ind.dir == '+' || continue
        j = get1jtensor(leginfos[d])
        q = contract(q, (d,), j, (2,); reduce_lock=false)
        perm = (ntuple(i -> i, d - 1)..., D + 1, ntuple(i -> d - 1 + i, D + 1 - d)...)
        q = permutedims(q, perm)
    end

    return q
end
