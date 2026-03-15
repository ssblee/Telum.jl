# ── getLocalSpace ────────────────────────────────────────────────────────────
#
# Entry point:
#
#   IS, operators... = getLocalSpace(opts::LocalSpaceOptions)
#
# Internal pipeline (common to all option types):
#
#   1. getSymmetryInfo(opts)
#        └─► (symm_tuple, weights, lowering_ops)
#              |
#              ▼
#   2. decompose_space(...)          [decomp_space.jl]
#        └─► (ortho_vecs, sector_index_map)   ← symmetry-resolved basis
#              |
#              ▼
#   3. getIROPs(opts, ortho_vecs)
#        └─► NamedTuple of IROP matrices, e.g. (F=..., S=..., Z=..., IS=...)
#              |
#              ▼
#   4. compress_irops(...)           [internal helper below]
#        └─► one QSpace object per IROP
#
# To add a new space type:
#   a) Define a new subtype of LocalSpaceOptions in options.jl.
#   b) Implement getSymmetryInfo(opts::YourOptions) and
#      getIROPs(opts::YourOptions, ortho_vecs, sector_map) below.
#   c) getLocalSpace itself needs no changes.
# ─────────────────────────────────────────────────────────────────────────────

# ── Generic hooks (must be specialised for each option type) ─────────────────

"""
    getSymmetryInfo(opts::LocalSpaceOptions)
        -> (symm, weights, lowering_ops)

Return the ingredients needed to decompose the local Hilbert space into
symmetry sectors:

- `symm`         – `NTuple{N, Type{<:Symmetry}}` listing the relevant symmetries.
- `weights`      – `NTuple{N, Vector}`, weight (charge) of each basis state for
                   each symmetry.
- `z_ops`        – `NTuple{N, Vector{<:AbstractMatrix}}`, diagonal Cartan / charge
                   operators (one per symmetry; U(1) has one, SU(2) has one, SU(3) two).
- `lowering_ops` – `NTuple{N, Vector{<:AbstractMatrix}}`, lowering operators
                   (empty vector for purely abelian symmetries).
"""
function getSymmetryInfo(opts::LocalSpaceOptions)
    error("getSymmetryInfo not implemented for $(typeof(opts))")
end

# ── Generic pipeline ─────────────────────────────────────────────────────────

"""
    getLocalSpace(opts::LocalSpaceOptions)

Construct the symmetry-adapted local state space and irreducible operator
representations (IROPs) for the physical site described by `opts`.

Returns a named tuple whose fields correspond to the standard operators of the
given space type (e.g. `IS`, `Z`, `F`, `S` for a spinful fermionic site).
"""

function reduceto2d(symm::NTuple{N, Any}, 
    data_full::Vector{Tuple{NTuple{3, NTuple{N, Tuple{Vararg{Int}}}}, Array{Float64, M}}}) where {N, M}

    @assert M == N + 3
    reduced = Vector{Tuple{NTuple{2, NTuple{N, Tuple{Vararg{Int}}}}, Array{Float64, 2+N}}}()

    for (qlabels, block) in data_full
        block_idx = (:, :, 1, (Colon() for _=1:N)...)
        push!(reduced, (qlabels[1:2], block[block_idx...]))
    end
    return reduced
end

function multiplyNreduce!(symm::NTuple{N, Any}, 
    data_full::Vector{Tuple{NTuple{3, NTuple{N, Tuple{Vararg{Int}}}}, Array{Float64, M}}}, 
    float_fac::Float64) where {N, M}

    @assert M == N + 3
    for (_, block) in data_full block .*= float_fac end

    third_dim_qlabel = first(data_full)[1][3]
    is_third_singleton = all(all(iszero, qlabel) for qlabel in third_dim_qlabel)
    if is_third_singleton data_full = reduceto2d(symm, data_full) end
    return data_full
end

# Convert mult_ind from decompose_space to spaces format for QSpace
# mult_ind: Dict{qlabels => Vector{(start_idx, end_idx)}}
# returns: Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}} - list of (RMT_dim, qlabels)
# Sorted by qlabels for consistency
function _mult_ind_to_splist(symm::NTuple{N, Any}, 
    mult_ind::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Tuple{Int, Int}}}) where N
    
    splist = Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}}()
    for (qlabels, ranges) in mult_ind
        # RMT dimension = outer multiplicity (number of ranges) 
        # Each range represents one copy of the irrep, so outer_mult = length(ranges)
        rmt_dim = length(ranges)
        push!(splist, (rmt_dim, qlabels))
    end
    # Sort by qlabels (second element of each tuple)
    sort!(splist; by = x -> x[2])
    return splist
end

function getLocalSpace(opts::LocalSpaceOptions, 
    tags::NTuple{3, String}=("", "", ""))
    # Step 1 – symmetry operators that define the local Hilbert space structure
    symm, weights, lowering_ops, mwirops = getSymmetryInfo(opts)

    # Step 2 – decompose local space into symmetry sectors
    ortho_vecs, space_list = decompose_space(symm, weights, lowering_ops)
    
    # Convert space_list to splist format for QSpace.spaces field
    # First two legs share the same local space structure
    local_splist = _mult_ind_to_splist(symm, space_list)

    dir = ['+', '-', '-']
    QSpaces = Dict{Symbol, QSpace}()
    for (name, (mwirop, float_fac)) in mwirops
        # Step 3 – get full IROP in the symmetry-adapted basis
        irop_full, irop_qlabel = get_IROP(symm, weights, lowering_ops, mwirop)
        transf_basis!(irop_full, ortho_vecs)

        # Step 4 – compress each IROP matrix into a QSpace object
        data_full = decompose_irop(symm, irop_full, space_list, irop_qlabel)
        # Multiply float factor and reduce to 2D if the third dimension is singleton
        reduced = multiplyNreduce!(symm, data_full, float_fac)

        inds = Tuple(QIndex(tags[i], dir[i]) for i in 1:3)
        rows = get_rows(reduced, symm); qd = get_qd(rows[1])
        
        # Build spaces tuple: first two legs use local_splist, third leg (if exists) computed from rows
        if qd == 2
            spaces = (local_splist, local_splist)
        else
            # For 3-leg operators, third leg space is computed from rows
            third_splist = _compute_spaces(rows)[3]
            spaces = (local_splist, local_splist, third_splist)
        end
        
        q = QSpace(symm, rows, inds[1:qd], spaces)
        QSpaces[name] = q
    end
    return NamedTuple(QSpaces)
end