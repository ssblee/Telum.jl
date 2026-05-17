using Printf
using LinearAlgebra
include("LurTensor.jl")
include("utils.jl")
include("localspaces/localspaces.jl")

# A compile-time tag for a direct product of symmetry groups. Telum records
# symmetry identities in type parameters; use `symm(q)` to retrieve them
# without going through dynamic property access.
abstract type ProductSymm{Syms<:Tuple{Vararg{Symmetry}}} <: Symmetry end

ProductSymm(syms::Type{<:Symmetry}...) = ProductSymm{Tuple{syms...}}

product_symms(::Type{<:ProductSymm{Syms}}) where {Syms} = Tuple(Syms.parameters)
nsymms(::Type{<:ProductSymm{Syms}}) where {Syms} = length(Syms.parameters)

function productsymm(symm::Tuple)
    all(s -> s isa Type{<:Symmetry}, symm) || throw(ArgumentError(
        "all entries of a product symmetry must be symmetry types, got $symm"))
    return ProductSymm(symm...)
end

productsymm(::Type{PS}) where {PS<:ProductSymm} = PS
product_symms(symm::Tuple) = product_symms(productsymm(symm))
nsymms(symm::Tuple) = nsymms(productsymm(symm))

@generated function nonabelian_symmetry_indices(::Type{PS}) where {PS<:ProductSymm}
    syms = product_symms(PS)
    inds = Tuple(i for i in eachindex(syms) if !isabelian(syms[i]))
    return :($inds)
end

@inline n_nonabelian_symmetries(::Type{PS}) where {PS<:ProductSymm} =
    length(nonabelian_symmetry_indices(PS))

@generated function is_stored_wmat_symmetry(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    return :($(1 <= n <= length(syms) && !isabelian(syms[n])))
end

@generated function nonabelian_wmat_slot(::Type{PS}, ::Val{n}) where {PS<:ProductSymm, n}
    syms = product_symms(PS)
    slot = 0
    for i in eachindex(syms)
        isabelian(syms[i]) || (slot += 1)
        i == n && break
    end
    1 <= n <= length(syms) && !isabelian(syms[n]) ||
        return :(throw(ArgumentError("symmetry index $n is Abelian and has no stored w-matrix slot")))
    return :($slot)
end

function nonabelian_wmat_slot(::Type{PS}, n::Int) where {PS<:ProductSymm}
    syms = product_symms(PS)
    1 <= n <= length(syms) || throw(BoundsError(syms, n))
    slot = 0
    for i in eachindex(syms)
        isabelian(syms[i]) || (slot += 1)
        i == n && break
    end
    isabelian(syms[n]) &&
        throw(ArgumentError("symmetry index $n is Abelian and has no stored w-matrix slot"))
    return slot
end

@inline _trivial_wmat() = LurTensor(ones(1, 1))

const DenseWMat = LurTensor{Float64, 2, Matrix{Float64}}

@inline _wmat_matrix(::Type{PS}, nsectors::Int) where {PS<:ProductSymm} =
    Matrix{DenseWMat}(undef, n_nonabelian_symmetries(PS), nsectors)
@inline _wmat_matrix(symm::NTuple, nsectors::Int) =
    _wmat_matrix(productsymm(symm), nsectors)

@generated function _wmat_buffers(::Type{PS}) where {PS<:ProductSymm}
    M = length(Tuple(i for (i, S) in pairs(product_symms(PS)) if !isabelian(S)))
    return Expr(:tuple, (:(DenseWMat[]) for _ in 1:M)...)
end
@inline _wmat_buffers(symm::NTuple) = _wmat_buffers(productsymm(symm))

@inline function _push_wmat!(buffers, ::Type{PS}, n::Int, wmat) where {PS<:ProductSymm}
    isabelian(product_symms(PS)[n]) && return buffers
    push!(buffers[nonabelian_wmat_slot(PS, n)], wmat)
    return buffers
end

@inline _push_wmat!(buffers, symm::NTuple, n::Int, wmat) =
    _push_wmat!(buffers, productsymm(symm), n, wmat)

function _wmat_matrix_from_buffers(::Type{PS}, buffers, nsectors::Int) where {PS<:ProductSymm}
    out = _wmat_matrix(PS, nsectors)
    for slot in axes(out, 1)
        length(buffers[slot]) == nsectors ||
            throw(ArgumentError("w-matrix buffer $slot has length $(length(buffers[slot])) but expected $nsectors"))
        for sector_index in 1:nsectors
            out[slot, sector_index] = buffers[slot][sector_index]
        end
    end
    return out
end
@inline _wmat_matrix_from_buffers(symm::NTuple, buffers, nsectors::Int) =
    _wmat_matrix_from_buffers(productsymm(symm), buffers, nsectors)

@inline function _set_sector_wmat!(wmats::AbstractMatrix, ::Type{PS}, sector_index::Int, n::Int, wmat) where {PS<:ProductSymm}
    isabelian(product_symms(PS)[n]) && return wmats
    wmats[nonabelian_wmat_slot(PS, n), sector_index] = wmat
    return wmats
end
@inline _set_sector_wmat!(wmats::AbstractMatrix, symm::NTuple, sector_index::Int, n::Int, wmat) =
    _set_sector_wmat!(wmats, productsymm(symm), sector_index, n, wmat)

# ─── Tag helpers ──────────────────────────────────────────────────────────────
#
# Tags are stored internally as an Itag wrapper around a canonical
# comma-separated String sorted alphabetically, e.g. "bond,site,u1"
# (ITensor convention). Empty string means no tags. Individual tags contain no
# commas or whitespace.
#
# ─────────────────────────────────────────────────────────────────────────────

_itag_text(tags::AbstractString) = String(tags)

# Split tag text into sorted vector of non-empty individual tags.
_parse_itag(tags::AbstractString) = sort!(filter!(!isempty, strip.(split(_itag_text(tags), ','))))

# Canonical form: sorted, comma-joined, no spaces.
_normalize_itag_text(tags::AbstractString) = join(_parse_itag(tags), ',')

struct Itag <: AbstractString
    value::String

    Itag(tags::AbstractString) = new(_normalize_itag_text(tags))
end

Itag(tags::Itag) = tags
_itag_text(tags::Itag) = tags.value
_normalize_itag(tags::AbstractString) = Itag(tags)

Base.String(tags::Itag) = tags.value
Base.convert(::Type{String}, tags::Itag) = tags.value
Base.ncodeunits(tags::Itag) = ncodeunits(tags.value)
Base.codeunit(::Type{Itag}) = codeunit(String)
Base.codeunit(tags::Itag, i::Int) = codeunit(tags.value, i)
Base.isvalid(tags::Itag, i::Int) = isvalid(tags.value, i)
Base.iterate(tags::Itag) = iterate(tags.value)
Base.iterate(tags::Itag, i::Int) = iterate(tags.value, i)
Base.length(tags::Itag) = length(tags.value)
Base.lastindex(tags::Itag) = lastindex(tags.value)
Base.thisind(tags::Itag, i::Int) = thisind(tags.value, i)
Base.nextind(tags::Itag, i::Int) = nextind(tags.value, i)
Base.nextind(tags::Itag, i::Int, n::Int) = nextind(tags.value, i, n)
Base.prevind(tags::Itag, i::Int) = prevind(tags.value, i)
Base.prevind(tags::Itag, i::Int, n::Int) = prevind(tags.value, i, n)
Base.show(io::IO, tags::Itag) = print(io, tags.value)
Base.isempty(tags::Itag) = isempty(tags.value)
Base.hash(tags::Itag, h::UInt) = hash(tags.value, h)
Base.:(==)(a::Itag, b::Itag) = a.value == b.value
Base.:(==)(a::Itag, b::AbstractString) = a.value == _normalize_itag_text(b)
Base.:(==)(a::AbstractString, b::Itag) = _normalize_itag_text(a) == b.value
Base.isequal(a::Itag, b::Itag) = (a == b)
Base.isequal(a::Itag, b::AbstractString) = (a == b)
Base.isequal(a::AbstractString, b::Itag) = (a == b)

# True iff all comma-separated tags in `query` are present in `base`.
function _has_itag(base::AbstractString, query::AbstractString)
    bset = Set(_parse_itag(base))
    return all(t -> t ∈ bset, _parse_itag(query))
end

_matches_itag_selector(base::AbstractString, query::AbstractString) = _has_itag(base, query)
_matches_itag_selector(base::AbstractString, queries::Tuple{Vararg{AbstractString}}) =
    any(query -> _has_itag(base, query), queries)
_matches_itag_selector(base::AbstractString, queries::AbstractVector{<:AbstractString}) =
    any(query -> _has_itag(base, query), queries)

# Add tags from `newtags` to `base`; result is sorted and deduplicated.
_add_itag(base::AbstractString, newtags::AbstractString) =
    Itag(join(sort!(unique!(vcat(_parse_itag(base), _parse_itag(newtags)))), ','))

# Remove every tag listed in `rmtags` from `base`.
function _remove_itag(base::AbstractString, rmtags::AbstractString)
    rm = Set(_parse_itag(rmtags))
    return Itag(join(filter(t -> t ∉ rm, _parse_itag(base)), ','))
end

# Remove tags from `base` using one or more ITensor-style tag queries.
# Each query is applied independently, and its tags are removed only if all
# tags in that query are present in `base`.
function _remove_itag(base::AbstractString,
                      rmtags::Union{Tuple{Vararg{AbstractString}},
                                    AbstractVector{<:AbstractString}})
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    rm = Set{String}()
    for query in rmtags
        parsed = _parse_itag(query)
        all(tag -> tag ∈ base_set, parsed) || continue
        union!(rm, parsed)
    end
    return Itag(join(filter(tag -> tag ∉ rm, base_tags), ','))
end

function _replace_itag(base::AbstractString, replacements)
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    replaced = Set{String}()
    new_tags = String[]

    for (from, to) in replacements
        from_tags = _parse_itag(from)
        all(tag -> tag ∈ base_set, from_tags) || continue
        union!(replaced, from_tags)
        append!(new_tags, _parse_itag(to))
    end

    append!(new_tags, (tag for tag in base_tags if tag ∉ replaced))
    return Itag(join(sort!(unique!(new_tags)), ','))
end

struct TLIndex
    # Tags associated with the leg, similar to ITensor
    itags::Itag
    # Direction of the leg, '+' for incoming, '-' for outgoing
    dir::Char
    # Prime level similar to ITensor
    plev::Int
    # Lock level. Cannot cntracted if lock > 0. 
    # Decrease lock level by 1 after contraction.
    lock::Int
    # If true, this leg represents the dual space.
    dual::Bool

    TLIndex(itags::AbstractString, dir::Char, plev::Int=0, lock::Int=0, dual::Bool=false) = new(_normalize_itag(itags), dir, plev, lock, dual)
end

TLIndex(dir::Char, plev::Int=0, lock::Int=0) = TLIndex("", dir, plev, lock)

# Two TLIndex objects are equal if they share the same itags, dir, plev, and dual
# (lock is intentionally ignored — it is a transient contraction counter).
Base.:(==)(a::TLIndex, b::TLIndex) = a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.dual == b.dual
Base.isequal(a::TLIndex, b::TLIndex) = (a == b)
Base.hash(a::TLIndex, h::UInt) = hash((a.itags, a.dir, a.plev, a.dual), h)

to_incoming(idx::TLIndex) = TLIndex(idx.itags, '+', idx.plev, idx.lock, idx.dual)
to_outgoing(idx::TLIndex) = TLIndex(idx.itags, '-', idx.plev, idx.lock, idx.dual)
change_dir(idx::TLIndex)  = TLIndex(idx.itags, idx.dir == '+' ? '-' : '+', idx.plev, idx.lock, idx.dual)
dual(idx::TLIndex) = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, true)
change_dual(idx::TLIndex) = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, !idx.dual)
green(idx::TLIndex) = dual(idx)
change_green(idx::TLIndex) = change_dual(idx)

# Format a scalar RMT value as a string with consistent width:
# integers print without decimal point; floats use %#.7g which always
# shows 7 significant digits including trailing zeros (e.g. 3.46410 not 3.4641).
function _fmt_scalar_str(v::Real)
    return @sprintf("%#.7g", v)
end

function _localspace_cgt_fields(data::Vector{Tuple{NTuple{QD, NTuple{N, Tuple{Vararg{Int}}}}, Array{T, RD}}},
                                symm::NTuple{N, Any},
                                spaces::Tuple{Vararg{<:AbstractVector, QD}}) where {T, QD, N, RD}

    @assert RD == QD + N; @assert QD == 2 || QD == 3
    QT = qlabeltype(symm)
    qlabels = Matrix{QT}(undef, QD, length(data))
    wmats = _wmat_matrix(symm, length(data))
    RMTs = LurTensor{T, RD, Array{T, RD}}[]

    for (sector_index, (sector_qlabels, block0)) in pairs(data)
        block = block0
        sector_wmats = Vector{LurTensor{Float64, 2, Matrix{Float64}}}(undef, N)
        for i in 1:N
            wmat, block, _ = svd_leg(block, QD + i)
            sector_wmats[i] = LurTensor(wmat)
        end
        RMT = LurTensor(block)
        for leg in 1:QD
            qlabels[leg, sector_index] = sector_qlabels[leg]
        end
        cgt_metadata = ntuple(N) do i
            qforsymm = Tuple(sector_qlabels[j][i] for j in 1:QD)
            if QD == 2
                cgp = (1, 2); legdir = (1, 1)
            elseif QD == 3
                cgp = qforsymm[2] <= qforsymm[3] ? (1, 2, 3) : (1, 3, 2); legdir = (1, 2)
            else error("Invalid TLArray tensor dimension for getLocalSpace: $QD") end

            if QD == 3 && qforsymm[2] > qforsymm[3]
                qforsymm = (qforsymm[1], qforsymm[3], qforsymm[2])
            end
            (qforsymm, cgp, legdir)
        end

        if QD == 2
            for i in 1:N
                S = symm[i]
                q1, q2 = cgt_metadata[i][1]
                @assert q2 == q1 || q2 == get_dualq(S, q1)
                dim = dimension(S, q1); @assert dim == dimension(S, q2)
                @assert size(sector_wmats[i]) == (1, 1)
                w_factor = sqrt(dim) / sector_wmats[i][1]
                sector_wmats[i][:] *= w_factor
                RMT[:] /= w_factor
            end
        end

        for i in 1:N
            if length(sector_wmats[i].data) == 1 && sector_wmats[i].data[1] < 0
                sector_wmats[i][:] .*= -1
                RMT[:] .*= -one(T)
            end
        end

        for i in 1:N
            _set_sector_wmat!(wmats, symm, sector_index, i, sector_wmats[i])
        end
        push!(RMTs, RMT)
    end

    q = _field_tlarray(symm, qlabels, wmats, RMTs,
                       ntuple(_ -> TLIndex('+'), Val(QD)), spaces)
    return _tlarray_fields(_drop_small_sectors!(q))
end

# ─── TLArray invariant checkers ─────────────────────────────────────────────

# Condition 1: an index with empty itags must have lock == 0
function _check_empty_tag_lock(inds::NTuple{QD, TLIndex}) where QD
    for idx in inds
        if isempty(idx.itags)
            @assert idx.lock == 0 "TLArray: index with empty itag has nonzero lock=$(idx.lock)"
        end
    end
end

# Condition 2: no two TLIndex objects with non-empty itags in the inds tuple
# may be equal (as determined by ==, which compares itags, dir, plev, dual).
function _check_unique_inds(inds::NTuple{QD, TLIndex}) where QD
    tagged = [idx for idx in inds if !isempty(idx.itags)]
    for i in 1:length(tagged), j in i+1:length(tagged)
        @assert tagged[i] != tagged[j] begin
            "Duplicate TLIndex with non-empty itag in TLArray.inds: $(tagged[i])"
        end
    end
end

const QSPACE_SECTOR_CUTOFF = 1e-14

# T: type of element in the RMT array, can be Float64, ComplexF64, etc.
# QD: The rank of tensor (# of legs), N: The number of symmetries
# RD: The rank of RMT array, which is equal to QD + N
# QT: The qlabel type for one leg sector, inferred from the symmetries
struct TLArray{T, QD, N, RD, QT, PS<:ProductSymm, WMATS, RMTS}
    qlabels::Matrix{QT}
    wmats::WMATS
    RMTs::RMTS
    inds::NTuple{QD, TLIndex}
    # Space list for each leg: vector of (qlabels, RMT_dim) pairs
    # Similar to leginfo.splist but precomputed for all legs
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}

    function TLArray(symm::NTuple{N, Any},
        qlabels::Matrix{QT},
        wmats::WMATS,
        RMTs::RMTS,
        inds::NTuple{QD, TLIndex},
        spaces::Tuple{Vararg{<:AbstractVector, QD}}) where {
            N, QD, QT, WMATS<:AbstractMatrix, T, RD,
            RMTS<:AbstractVector{<:LurTensor{T, RD}}}

        size(qlabels) == (QD, length(RMTs)) ||
            throw(ArgumentError("qlabels must have size (number of TLArray legs, number of sectors)"))

        PS = productsymm(symm)
        size(wmats) == (n_nonabelian_symmetries(PS), length(RMTs)) ||
            throw(ArgumentError("w-matrix table must have size (number of non-Abelian symmetries, number of sectors)"))
        typed_spaces = ntuple(l -> convert(Vector{Tuple{QT, Int}}, spaces[l]), QD)
        q = new{T, QD, N, RD, QT, PS, WMATS, RMTS}(
            qlabels, wmats, RMTs, inds, typed_spaces)
        _check_unique_inds(q.inds)
        _check_empty_tag_lock(q.inds)
        return q
    end
end

_compute_spaces(q::TLArray) = q.spaces

function _raw_field_tlarray(symm::NTuple{N, Any},
                            qlabels::Matrix{QT},
                            wmats::AbstractMatrix,
                            RMTs::RMTS,
                            inds::NTuple{QD, TLIndex},
                            spaces::Tuple{Vararg{<:AbstractVector, QD}}) where {N, QD, QT, RMTS}
    return TLArray(symm, qlabels, wmats, RMTs, inds, spaces)
end

function _normalize_wmats!(q::TLArray{T, QD, N}) where {T, QD, N}
    if QD == 2
        for sector_index in 1:nsectors(q)
            for i in 1:N
                S = symm(q)[i]
                isabelian(S) && continue
                q1 = sector_qlabel(q, sector_index, 1)[i]
                q2 = sector_qlabel(q, sector_index, 2)[i]
                @assert q2 == q1 || q2 == get_dualq(S, q1)
                dim = dimension(S, q1); @assert dim == dimension(S, q2)
                wmat = sector_wmat(q, sector_index, i)
                @assert size(wmat) == (1, 1)
                w_factor = sqrt(dim) / wmat[1]
                wmat[:] *= w_factor
                sector_rmt(q, sector_index)[:] /= w_factor
            end
        end
    elseif QD == 0
        for sector_index in 1:nsectors(q)
            for i in 1:N
                isabelian(symm(q)[i]) && continue
                wmat = sector_wmat(q, sector_index, i)
                @assert size(wmat) == (1, 1) "0D TLArray must have 1x1 w-matrices"
                w_val = wmat[1]
                if w_val != 1.0
                    wmat[:] .= 1.0
                    sector_rmt(q, sector_index)[:] .*= w_val
                end
            end
        end
    end
    return q
end

function _field_tlarray(symm::NTuple{N, Any},
                        qlabels::Matrix{QT},
                        wmats::WMATS,
                        RMTs::RMTS,
                        inds::NTuple{QD, TLIndex},
                        spaces::Tuple{Vararg{<:AbstractVector, QD}}) where {N, QD, QT, WMATS, RMTS}
    q = _raw_field_tlarray(symm, qlabels, wmats, RMTs, inds, spaces)
    _normalize_wmats!(q)
    _orient_wmats!(q)
    return _drop_small_sectors!(q)
end

productsymm(::TLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} = PS
product_symms(q::TLArray) = product_symms(productsymm(q))
@inline symm(::TLArray{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    product_symms(PS)
nsymms(q::TLArray) = nsymms(productsymm(q))

Base.propertynames(q::TLArray, private::Bool=false) =
    (:qlabels, :wmats, :RMTs, :inds, :spaces)

@inline nsectors(q::TLArray) = length(q.RMTs)
@inline sector_qlabel(q::TLArray, sector::Int, leg::Int) = q.qlabels[leg, sector]
@inline sector_qlabel(::Type{QT}, q::TLArray, sector::Int, leg::Int) where {QT} =
    q.qlabels[leg, sector]::QT
@inline function sector_wmat(q::TLArray{T, QD, N, RD, QT, PS}, sector::Int, n::Int) where {T, QD, N, RD, QT, PS}
    isabelian(symm(q)[n]) && return _trivial_wmat()
    return q.wmats[nonabelian_wmat_slot(PS, n), sector]
end
@inline function sector_wmat(q::TLArray{T, QD, N, RD, QT, PS}, sector::Int, ::Val{n}) where {T, QD, N, RD, QT, PS, n}
    is_stored_wmat_symmetry(PS, Val(n)) || return _trivial_wmat()
    return q.wmats[nonabelian_wmat_slot(PS, Val(n)), sector]
end
@inline sector_rmt(q::TLArray, sector::Int) = q.RMTs[sector]

@inline function _stored_position(stored_to_phys::NTuple{QD, Int}, phys_leg::Int) where {QD}
    @inbounds for stored_pos in 1:QD
        stored_to_phys[stored_pos] == phys_leg && return stored_pos
    end
    throw(BoundsError(stored_to_phys, phys_leg))
end

@inline _phys_to_stored_order(stored_to_phys::NTuple{QD, Int}) where {QD} =
    ntuple(phys_leg -> _stored_position(stored_to_phys, phys_leg), Val(QD))

function _stored_leg_order(q::TLArray{T, QD, N}, sector::Int, n::Int) where {T, QD, N}
    incoming = Int[l for l in 1:QD if q.inds[l].dir == '+']
    outgoing = Int[l for l in 1:QD if q.inds[l].dir == '-']

    sort!(incoming; by = l -> sector_qlabel(q, sector, l)[n], alg = MergeSort)
    sort!(outgoing; by = l -> sector_qlabel(q, sector, l)[n], alg = MergeSort)

    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD))
end

function _stored_leg_order(qlabels::AbstractMatrix,
                           inds::NTuple{QD, TLIndex},
                           sector::Int,
                           n::Int) where {QD}
    incoming = Int[l for l in 1:QD if inds[l].dir == '+']
    outgoing = Int[l for l in 1:QD if inds[l].dir == '-']

    sort!(incoming; by = l -> qlabels[l, sector][n], alg = MergeSort)
    sort!(outgoing; by = l -> qlabels[l, sector][n], alg = MergeSort)

    n_in = length(incoming)
    return ntuple(i -> i <= n_in ? incoming[i] : outgoing[i - n_in], Val(QD))
end

function _sector_cgt_metadata(q::TLArray{T, QD, N}, sector::Int, n::Int) where {T, QD, N}
    stored_to_phys = _stored_leg_order(q, sector, n)
    qlabels = ntuple(i -> sector_qlabel(q, sector, stored_to_phys[i])[n], Val(QD))

    n_in = count(l -> q.inds[l].dir == '+', 1:QD)
    return qlabels, _phys_to_stored_order(stored_to_phys), (n_in, QD - n_in)
end

function _sector_cgt_metadata(qlabels::AbstractMatrix,
                           inds::NTuple{QD, TLIndex},
                           sector::Int,
                           n::Int) where {QD}
    stored_to_phys = _stored_leg_order(qlabels, inds, sector, n)
    stored_qlabels = ntuple(i -> qlabels[stored_to_phys[i], sector][n], Val(QD))

    n_in = count(l -> inds[l].dir == '+', 1:QD)
    return stored_qlabels, _phys_to_stored_order(stored_to_phys), (n_in, QD - n_in)
end

function _sector_label_widths(q::TLArray{T, QD, N}, sector_index::Int) where {T, QD, N}
    map(1:N) do n
        vals = (v for l in 1:QD for v in sector_qlabel(q, sector_index, l)[n])
        mxabs = maximum(abs, vals, init=0)
        needs_sign = symm(q)[n] <: U1
        ndigits(max(mxabs, 1)) + (needs_sign ? 1 : 0)
    end
end

function _print_sector_qlabels(io::IO, q::TLArray{T, QD, N}, sector_index::Int, widths) where {T, QD, N}
    print(io, "[")
    for l in 1:QD
        l > 1 && print(io, " ;")
        for n in 1:N
            print(io, " ")
            for v in sector_qlabel(q, sector_index, l)[n]
                print(io, lpad(v, widths[n]))
            end
        end
    end
    print(io, " ]")
end

function _print_sector_cgt_dims(io::IO, q::TLArray{T, QD, N}, sector_index::Int) where {T, QD, N}
    first = true
    for n in 1:N
        S = symm(q)[n]
        isabelian(S) && continue
        dims = [dimension(S, sector_qlabel(q, sector_index, l)[n]) for l in 1:QD]
        first && print(io, "| ")
        first = false
        print(io, join(dims, "x"), "\t")
    end
end

function _sector_cgt_size_2d(q::TLArray{T, 2, N}, sector_index::Int) where {T, N}
    total = 1
    for n in 1:N
        S = symm(q)[n]
        isabelian(S) && continue
        total *= dimension(S, sector_qlabel(q, sector_index, 1)[n])
    end
    return total
end

_tlarray_fields(q::TLArray) = (qlabels = q.qlabels, wmats = q.wmats, RMTs = q.RMTs)

# Drop sectors whose norm² contribution is below cutoff² × total norm² (relative threshold).
# For QD == 2 the effective norm² per sector is dim_r * ‖RMT_r‖² (see normalize_qspace!);
# for all other ranks it is simply ‖RMT_r‖².
Base.ndims(q::TLArray{T, QD}) where {T, QD} = QD

function _orient_wmats!(q::TLArray{T, QD, N}) where {T, QD, N}
    for sector_index in 1:nsectors(q)
        for n in 1:N
            isabelian(symm(q)[n]) && continue
            wmat = sector_wmat(q, sector_index, n)
            if length(wmat.data) == 1 && wmat.data[1] < 0
                wmat[:] .*= -1
                sector_rmt(q, sector_index)[:] .*= -one(T)
            end
        end
    end
    return q
end

function _drop_small_sectors!(q::TLArray{T, QD, N}; cutoff::Float64 = QSPACE_SECTOR_CUTOFF) where {T, QD, N}
    nsectors(q) == 0 && return q

    total = 0.0
    for sector_index in 1:nsectors(q)
        total += QD == 2 ? _sector_cgt_size_2d(q, sector_index) * sum(abs2, sector_rmt(q, sector_index).data) :
                 sum(abs2, sector_rmt(q, sector_index).data)
    end
    iszero(total) && return q   # zero tensor: keep all

    thresh  = (cutoff^2) * total
    keep = trues(nsectors(q))
    for sector_index in 1:nsectors(q)
        sector_norm_sq = QD == 2 ? _sector_cgt_size_2d(q, sector_index) * sum(abs2, sector_rmt(q, sector_index).data) :
                         sum(abs2, sector_rmt(q, sector_index).data)
        keep[sector_index] = sector_norm_sq > thresh
    end

    if !all(keep)
        qlabels = copy(q.qlabels[:, keep])
        wmats = copy(q.wmats[:, keep])
        RMTs = q.RMTs[keep]
        return _raw_field_tlarray(symm(q), qlabels, wmats, RMTs, q.inds, _copy_spaces_tuple(q.spaces))
    end
    return q
end

# Construct a TLArray with the same sectors but with itags replaced.
# itags: Tuple{Vararg{AbstractString, QD}} — one tag per leg; all other TLIndex
# fields are preserved.
function TLArray(q::TLArray{T, QD, N, RD}, itags::Tuple{Vararg{AbstractString, QD}}) where {T, QD, N, RD}
    new_inds = ntuple(l -> TLIndex(itags[l], q.inds[l].dir, q.inds[l].plev,
                                  q.inds[l].lock, q.inds[l].dual), QD)
    return _field_tlarray(symm(q), copy(q.qlabels), deepcopy(q.wmats), deepcopy(q.RMTs),
                          new_inds, _copy_spaces_tuple(q.spaces))
end

# Construct a TLArray with the same sectors but with all TLIndex fields replaced.
# inds: NTuple{QD, TLIndex} — one full TLIndex per leg.
# Arrow directions must match the original TLArray (only itags/lock/plev/dual may differ).
function TLArray(q::TLArray{T, QD, N, RD}, inds::NTuple{QD, TLIndex}) where {T, QD, N, RD}
    @assert ntuple(l -> inds[l].dir, QD) == ntuple(l -> q.inds[l].dir, QD) "TLArray(q, inds): arrow directions must match the original TLArray on all legs"
    return _field_tlarray(symm(q), copy(q.qlabels), deepcopy(q.wmats), deepcopy(q.RMTs),
                          inds, _copy_spaces_tuple(q.spaces))
end

Base.getindex(q::TLArray, i::Int) = TLArray(q, i)
Base.length(q::TLArray) = nsectors(q)
Base.firstindex(q::TLArray) = 1
Base.lastindex(q::TLArray) = nsectors(q)

function _normalize_qspace_sector_index(i::Int, nsectors::Int)
    i == 0 && throw(BoundsError(1:nsectors, i))
    idx = i < 0 ? nsectors + i + 1 : i
    1 <= idx <= nsectors || throw(BoundsError(1:nsectors, i))
    return idx
end

function _normalize_qspace_sector_selector(selector, nsectors::Int)
    if selector isa Colon
        return collect(1:nsectors)
    elseif selector isa Integer
        return Int[_normalize_qspace_sector_index(Int(selector), nsectors)]
    elseif selector isa AbstractVector{Bool}
        length(selector) == nsectors || throw(DimensionMismatch(
            "sector selector of length $(length(selector)) does not match number of sectors $nsectors"))
        return findall(selector)
    elseif selector isa AbstractRange{<:Integer}
        return _normalize_qspace_sector_selector(collect(selector), nsectors)
    elseif selector isa AbstractVector{<:Integer}
        inds = Int[_normalize_qspace_sector_index(Int(i), nsectors) for i in selector]
        length(unique(inds)) == length(inds) || throw(ArgumentError(
            "sector selector must not contain duplicate indices"))
        return inds
    else
        throw(ArgumentError(
            "sector selector must be :, Int, AbstractRange{<:Integer}, AbstractVector{<:Integer}, or AbstractVector{Bool}"))
    end
end

"""
    TLArray(q::TLArray, selector) -> TLArray

Create a new `TLArray` from a subset of sectors, preserving the original
symmetry tuple, leg indices, and cached leg-space metadata in `q.spaces`.

`selector` may be `:`, an `Int`, an integer range, an integer vector, or a
boolean mask. Negative integer indices count from the end.
"""
function TLArray(q::TLArray{T, QD, N, RD}, selector) where {T, QD, N, RD}
    inds = _normalize_qspace_sector_selector(selector, nsectors(q))
    qlabels = copy(q.qlabels[:, inds])
    wmats = deepcopy(q.wmats[:, inds])
    RMTs = deepcopy(q.RMTs[inds])
    return _field_tlarray(symm(q), qlabels, wmats, RMTs, q.inds, _copy_spaces_tuple(q.spaces))
end

Base.getindex(q::TLArray,
              selector::Union{Colon, AbstractRange{<:Integer},
                              AbstractVector{<:Integer}, AbstractVector{Bool}}) = TLArray(q, selector)

# For 0-dimensional TLArray (scalar), q[] returns the unique RMT element.
function Base.getindex(q::TLArray{T, 0, N, N}) where {T, N}
    @assert nsectors(q) <= 1 "0D TLArray must have zero or one sector"
    if nsectors(q) == 1
        @assert length(sector_rmt(q, 1).data) == 1 "0D TLArray RMT must be a scalar"
        return only(sector_rmt(q, 1).data)
    else return zero(T) end
end

# ─── Leg selection utilities ─────────────────────────────────────────────────

"""
    findlegs(q::TLArray, pred::Function) -> Vector{Int}

Find all leg indices where `pred(qindex)` returns true.

# Examples
```julia
findlegs(q, idx -> idx.dir == '-')                    # all outgoing legs
findlegs(q, idx -> occursin("site", idx.itags))       # legs with "site" in tag
findlegs(q, idx -> idx.dir == '-' && idx.plev == 0)   # outgoing, unprimed
```
"""
findlegs(q::TLArray{T, QD}, pred::Function) where {T, QD} = [i for i in 1:QD if pred(q.inds[i])]

"""
    findleg(q::TLArray, pred::Function) -> Int

Find the first leg index where `pred(qindex)` returns true.
Throws an error if no leg matches.

# Examples
```julia
findleg(q, idx -> idx.itags == "bond")   # first leg with exact tag "bond"
findleg(q, idx -> idx.dir == '+')        # first incoming leg
```
"""
function findleg(q::TLArray{T, QD}, pred::Function) where {T, QD}
    for i in 1:QD pred(q.inds[i]) && return i end
    return nothing
end

# Internal: check if a TLIndex matches all specified criteria
function _matches_criteria(idx::TLIndex; dir=nothing, itag=nothing, plev=nothing, lock=nothing)
    (!isnothing(dir)  && idx.dir != dir)                                 && return false
    (!isnothing(itag) && !_matches_itag_selector(idx.itags, itag))       && return false
    (!isnothing(plev) && idx.plev != plev)                               && return false
    (!isnothing(lock) && idx.lock != lock)                               && return false
    return true
end

"""
    findlegs(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Find all leg indices matching the specified criteria. Unspecified criteria match any value.
If `rev=true`, the selection is reversed: legs that do *not* match are returned.

# Arguments
- `dir`: Match direction ('+' for incoming, '-' for outgoing)
- `itag`: Match exact tags. A single string like `"a,b"` means both `a` and `b`
  must be present; a tuple/vector of strings means any one of those tag-sets may match
- `plev`: Match exact prime level
- `lock`: Match exact lock level
- `rev`: If `true`, return legs that do *not* satisfy the criteria (default `false`)

# Examples
```julia
findlegs(q; dir='-')                    # all outgoing legs
findlegs(q; itag="site")                # legs carrying tag "site"
findlegs(q; itag=("a,b", "a,c"))        # legs with tags a+b, or a+c
findlegs(q; dir='+', plev=0)            # incoming, unprimed legs
findlegs(q; lock=0)                     # non-locked legs
findlegs(q; dir='-', rev=true)          # all legs that are NOT outgoing
```
"""
function findlegs(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    return [i for i in 1:QD if _matches_criteria(q.inds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev]
end

"""
    findleg(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Find the first leg index matching the specified criteria.
Returns `nothing` if no leg matches.
If `rev=true`, returns the first leg that does *not* match the criteria.

# Examples
```julia
findleg(q; itag="bond")           # first leg with "bond" in tag
findleg(q; dir='+')               # first incoming leg
findleg(q; lock=0)                # first non-locked leg
findleg(q; dir='+', rev=true)     # first leg that is NOT incoming
```
"""
function findleg(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    for i in 1:QD
        _matches_criteria(q.inds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

function _matching_targets(q::TLArray; require_unlocked::Bool=false)
    return Set(change_dir(idx) for idx in q.inds
               if !require_unlocked || idx.lock == 0)
end

_has_target_match(idx::TLIndex, targets; require_unlocked::Bool=false) =
    (!require_unlocked || idx.lock == 0) && (idx in targets)

function _find_matching_legs(a::TLArray{T, QD}, b::TLArray;
                             dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             matched::Bool=true,
                             require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    return [i for i in 1:QD
            if (_has_target_match(a.inds[i], targets; require_unlocked=require_unlocked) == matched) &&
               (_matches_criteria(a.inds[i]; dir=dir, itag=itag,
                                  plev=plev, lock=lock) ⊻ rev)]
end

function _find_matching_leg(a::TLArray{T, QD}, b::TLArray;
                            dir=nothing, itag=nothing, plev=nothing,
                            lock=nothing, rev::Bool=false,
                            matched::Bool=true,
                            require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    for i in 1:QD
        (_has_target_match(a.inds[i], targets; require_unlocked=require_unlocked) == matched) || continue
        _matches_criteria(a.inds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

"""
    matchings(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one matching leg in `b`.

A leg matches when it has the same `itags`, `plev`, and `dual` flag as a leg
of `b`, but with opposite direction. Lock is ignored for the cross-tensor
match test. Keyword arguments filter the returned legs of `a` using the same
rules as `findlegs`.
"""
function matchings(a::TLArray{T, QD}, b::TLArray;
                   dir=nothing, itag=nothing, plev=nothing,
                   lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true)
end

"""
    matching(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a matching leg in `b`, or `nothing`
if no such leg exists.

Matching ignores lock between tensors; keyword arguments filter the returned
leg of `a` using the same rules as `findleg`.
"""
function matching(a::TLArray{T, QD}, b::TLArray;
                  dir=nothing, itag=nothing, plev=nothing,
                  lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true)
end

"""
    unmatchings(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any matching leg in `b`.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned legs of `a` using the same rules as `findlegs`.
"""
function unmatchings(a::TLArray{T, QD}, b::TLArray;
                     dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false)
end

"""
    unmatching(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any matching leg in `b`,
or `nothing` if every leg is matched.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned leg of `a` using the same rules as `findleg`.
"""
function unmatching(a::TLArray{T, QD}, b::TLArray;
                    dir=nothing, itag=nothing, plev=nothing,
                    lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false)
end

"""
    contractables(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one contractable leg in `b`.

Two legs are contractable when they satisfy the same cross-tensor match rule as
`matchings` and both legs have `lock == 0`. Keyword arguments filter the
returned legs of `a` using the same rules as `findlegs`.
"""
function contractables(a::TLArray{T, QD}, b::TLArray;
                       dir=nothing, itag=nothing, plev=nothing,
                       lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true, require_unlocked=true)
end

"""
    contractable(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a contractable leg in `b`, or
`nothing` if no such leg exists.

Contractable legs satisfy the same cross-tensor match rule as `matchings`, but
both tensors must have `lock == 0` on the matched legs.
"""
function contractable(a::TLArray{T, QD}, b::TLArray;
                      dir=nothing, itag=nothing, plev=nothing,
                      lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true, require_unlocked=true)
end

"""
    uncontractables(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any contractable leg in `b`.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs. Keyword arguments filter the returned legs of
`a` using the same rules as `findlegs`.
"""
function uncontractables(a::TLArray{T, QD}, b::TLArray;
                         dir=nothing, itag=nothing, plev=nothing,
                         lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false, require_unlocked=true)
end

"""
    uncontractable(a::TLArray, b::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any contractable leg in
`b`, or `nothing` if every eligible leg is contractable.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs.
"""
function uncontractable(a::TLArray{T, QD}, b::TLArray;
                        dir=nothing, itag=nothing, plev=nothing,
                        lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false, require_unlocked=true)
end

# ─── Lock utilities ──────────────────────────────────────────────────────────
#
# Lock level semantics:
#   lock > 0  : Cannot be contracted. Decreases by 1 after each contraction.
#   lock == 0 : Normal leg, can be contracted.
#   lock == -1: Permanently locked, never changes and cannot be contracted.
#
# ─────────────────────────────────────────────────────────────────────────────

const LegList = Union{AbstractVector{<:Integer}, Tuple{Vararg{<:Integer}}}

# Internal: apply a lock modification function to selected leg indices.
# legs can be any iterable of integers (Int, Vector, UnitRange, Tuple, etc.)
function _modify_lock(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_lock = modify_fn(idx.lock)
        new_inds[i] = TLIndex(idx.itags, idx.dir, idx.plev, new_lock, idx.dual)
    end
    return _field_tlarray(symm(q), copy(q.qlabels), deepcopy(q.wmats), deepcopy(q.RMTs),
                          Tuple(new_inds), _copy_spaces_tuple(q.spaces))
end

# Lock increase function (respects permanent lock)
_lock_inc(current_lock, inc) = current_lock == -1 ? -1 : current_lock + inc

"""
    lock(q::TLArray, leg::Integer; inc::Int=1)

Increase lock level of a single specified leg by `inc` (default 1).
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::TLArray, leg::Integer; inc::Int=1)
    return _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray, legs::LegList; inc::Int=1)

Increase lock level of the specified legs by `inc` (default 1).
`legs` can be any vector, range, or tuple of integers, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::TLArray, legs::LegList; inc::Int=1)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray, pred::Function; inc::Int=1)

Increase lock level of legs satisfying predicate by `inc` (default 1).
"""
function Base.lock(q::TLArray, pred::Function; inc::Int=1)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::TLArray; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Increase lock level of legs matching criteria by `inc`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lock(q; dir='-')                  # lock all outgoing legs by 1
lock(q; inc=2, itag="bond")       # lock legs with "bond" in tag by 2
lock(q; lock=0)                   # lock all currently-unlocked legs by 1
lock(q; dir='-', rev=true)        # lock all legs that are NOT outgoing
```
"""
function Base.lock(q::TLArray; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lockp(q::TLArray, leg::Integer)

Permanently lock a single specified leg (set lock=-1).
"""
function lockp(q::TLArray, leg::Integer)
    return _modify_lock(q, (leg,), _ -> -1)
end

"""
    lockp(q::TLArray, legs::LegList)

Permanently lock the specified legs (set lock=-1).
`legs` can be any vector, range, or tuple of integers.
"""
function lockp(q::TLArray, legs::LegList)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::TLArray, pred::Function)

Permanently lock legs satisfying predicate.
"""
function lockp(q::TLArray, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Permanently lock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lockp(q; itag="phys")          # permanently lock physical legs
lockp(q; lock=0)               # permanently lock all currently-unlocked legs
lockp(q; itag="phys", rev=true)   # permanently lock all legs except "phys"
```
"""
function lockp(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    unlock(q::TLArray, leg::Integer)

Unlock a single specified leg (set lock=0). Also removes permanent lock.
"""
function Base.unlock(q::TLArray, leg::Integer)
    return _modify_lock(q, (leg,), _ -> 0)
end

"""
    unlock(q::TLArray, legs::LegList)

Unlock the specified legs (set lock=0).
`legs` can be any vector, range, or tuple of integers.
"""
function Base.unlock(q::TLArray, legs::LegList)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::TLArray, pred::Function)

Unlock legs satisfying predicate.
"""
function Base.unlock(q::TLArray, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Unlock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
unlock(q; dir='-')             # unlock all outgoing legs
unlock(q; lock=1)              # unlock all legs currently at lock=1
unlock(q; dir='-', rev=true)   # unlock all legs that are NOT outgoing
```
"""
function Base.unlock(q::TLArray; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> 0)
end

# ─── Prime level modification ────────────────────────────────────────────────
#
# ITensor-style prime level modification. Prime level is always non-negative.
# All increment forms use `inc` as an optional keyword argument (default 1).
#
#   prime(q; inc=1, kw...)         – increase plev of selected legs by inc
#   prime(q, leg; inc=1)           – increase plev of a single leg by inc
#   prime(q, legs; inc=1)          – increase plev of specified legs by inc
#   prime(q, pred; inc=1)          – increase plev of legs matching predicate
#   setprime(q, n; kw...)          – set plev of selected legs to n (≥0)
#   setprime(q, legs, n)           – set plev of specified legs to n
#   noprime(q; kw...)              – set plev of selected legs to 0
#   noprime(q, leg)                – set plev of a single leg to 0
#   noprime(q, legs)               – set plev of specified legs to 0
#   noprime(q, pred)               – set plev of legs matching predicate to 0
#
# Keyword selectors for criteria forms (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

# legs can be any iterable of integers
function _modify_plev(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.dual)
    end
    return _field_tlarray(symm(q), copy(q.qlabels), deepcopy(q.wmats), deepcopy(q.RMTs),
                          Tuple(new_inds), _copy_spaces_tuple(q.spaces))
end

"""
    prime(q::TLArray; inc::Int=1, dir, itag, plev, lock, rev=false)

Increase the prime level of matching legs by `inc` (default 1).
Prime level is clamped to 0 from below.
With no keyword arguments, all legs are affected.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
prime(q)                        # all legs: plev += 1
prime(q; inc=2)                 # all legs: plev += 2
prime(q; inc=-1)                # all legs: plev -= 1, clamped to 0
prime(q; dir='+')               # incoming legs only
prime(q; itag="site")           # legs whose tag contains "site"
prime(q; plev=0)                # only currently unprimed legs
prime(q; dir='+', rev=true)     # all legs that are NOT incoming
```
"""
function prime(q::TLArray{T, QD}; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::TLArray, leg::Integer; inc::Int=1)

Increase the prime level of a single specified leg by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, 2)                     # leg 2: plev += 1
prime(q, 2; inc=3)              # leg 2: plev += 3
```
"""
function prime(q::TLArray{T, QD}, leg::Integer; inc::Int=1) where {T, QD}
    return _modify_plev(q, (leg,), p -> max(0, p + inc))
end

"""
    prime(q::TLArray, legs::LegList; inc::Int=1)

Increase the prime level of the specified legs by `inc` (default 1).
Prime level is clamped to 0 from below.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.

# Examples
```julia
prime(q, [1, 3])                # legs 1 and 3: plev += 1
prime(q, 1:2; inc=2)            # legs 1–2: plev += 2
prime(q, (1, 3))                # same as prime(q, [1, 3])
```
"""
function prime(q::TLArray{T, QD}, legs::LegList; inc::Int=1) where {T, QD}
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::TLArray, pred::Function; inc::Int=1)

Increase the prime level of legs satisfying predicate by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, idx -> idx.dir == '+')          # incoming legs: plev += 1
prime(q, idx -> idx.plev == 0; inc=2)   # unprimed legs: plev += 2
```
"""
function prime(q::TLArray{T, QD}, pred::Function; inc::Int=1) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    setprime(q::TLArray, n::Int; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to `n`. `n` must be non-negative.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setprime(q, 2)                  # set all legs to plev=2
setprime(q, 1; dir='-')         # set outgoing legs to plev=1
setprime(q, 0; itag="bond")     # same as noprime(q; itag="bond")
setprime(q, 0; dir='-', rev=true)  # clear prime on all non-outgoing legs
```
"""
function setprime(q::TLArray{T, QD}, n::Int; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> n)
end

"""
    setprime(q::TLArray, legs, n::Int)

Set the prime level of the specified legs to `n`. `n` must be non-negative.
`legs` can be any iterable of leg indices.

# Examples
```julia
setprime(q, [1, 3], 2)          # set legs 1 and 3 to plev=2
setprime(q, 1:2, 0)             # same as noprime(q, 1:2)
```
"""
function setprime(q::TLArray{T, QD}, legs, n::Int) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, legs, _ -> n)
end

"""
    noprime(q::TLArray; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to 0.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
noprime(q)                      # clear prime on all legs
noprime(q; dir='+')             # clear prime on incoming legs only
noprime(q; plev=1)              # clear all legs currently at plev=1
noprime(q; dir='+', rev=true)   # clear prime on all non-incoming legs
```
"""
function noprime(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::TLArray, leg::Integer)

Set the prime level of a single specified leg to 0.
"""
function noprime(q::TLArray{T, QD}, leg::Integer) where {T, QD}
    return _modify_plev(q, (leg,), _ -> 0)
end

"""
    noprime(q::TLArray, legs::LegList)

Set the prime level of the specified legs to 0.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
"""
function noprime(q::TLArray{T, QD}, legs::LegList) where {T, QD}
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::TLArray, pred::Function)

Set the prime level of legs satisfying predicate to 0.
"""
function noprime(q::TLArray{T, QD}, pred::Function) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, _ -> 0)
end

# ─── Tag modification ────────────────────────────────────────────────────────
#
# ITensor-style tag manipulation on TLArray legs.
# Tags are stored as sorted, comma-separated strings (e.g. "bond,site").
#
#   additag(q, newtags; kw...)     – add tag(s) to matching legs
#   removeitag(q, tags; kw...)     – remove tag(s) from matching legs
#   replaceitag(q, old=>new...)    – replace tag queries on matching legs
#   setitag(q, tags; kw...)        – replace entire tag string of matching legs
#
# Keyword selectors (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

const ITagQuerySpec = Union{AbstractString,
                            Tuple{Vararg{AbstractString}},
                            AbstractVector{<:AbstractString}}
const ITagReplacementPair = Pair{<:AbstractString, <:AbstractString}
const ITagReplacementDict = AbstractDict{<:AbstractString, <:AbstractString}

function _modify_itag(q::TLArray{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = TLIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.dual)
    end
    return _field_tlarray(symm(q), copy(q.qlabels), deepcopy(q.wmats), deepcopy(q.RMTs),
                          Tuple(new_inds), _copy_spaces_tuple(q.spaces))
end

"""
    additag(q::TLArray, newtags; dir, itag, plev, lock, rev=false)

Add one or more tags to matching legs. `newtags` may be a comma-separated
string of tags (e.g. `"bond,u1"`). The result is always sorted.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
additag(q, "site")              # add "site" to all legs
additag(q, "phys"; dir='+')     # add "phys" to incoming legs only
additag(q, "u1"; itag="bond")   # add "u1" to legs that already have "bond"
additag(q, "aux"; dir='+', rev=true)  # add "aux" to all non-incoming legs
```
"""
function additag(q::TLArray{T, QD}, newtags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, leg::Integer, newtags)

Add tags to a single specified leg.
"""
function additag(q::TLArray{T, QD}, leg::Integer, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, (leg,), base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, legs::LegList, newtags)

Add tags to the specified legs. `legs` can be any vector, range, or tuple.
"""
function additag(q::TLArray{T, QD}, legs::LegList, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::TLArray, pred::Function, newtags)

Add tags to legs satisfying predicate.
"""
function additag(q::TLArray{T, QD}, pred::Function, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _add_itag(base, newtags))
end

"""
    removeitag(q::TLArray, tags; dir, itag, plev, lock, rev=false)

Remove one or more tags from matching legs.

`tags` may be either a single comma-separated tag string or a tuple/vector of
such strings. For tuple/vector input, each query is applied independently and
its tags are removed only from legs that contain all tags in that query.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
removeitag(q, "site")                     # remove "site" from all legs
removeitag(q, ("aaa,bbb", "ccc"))       # remove aaa+bbb together, and ccc independently
removeitag(q, "phys"; dir='-')           # remove "phys" from outgoing legs
removeitag(q, "aux"; dir='-', rev=true)  # remove "aux" from all non-outgoing legs
```
"""
function removeitag(q::TLArray{T, QD}, tags::ITagQuerySpec; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, leg::Integer, tags)

Remove tags from a single specified leg.
"""
function removeitag(q::TLArray{T, QD}, leg::Integer, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, (leg,), base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, legs::LegList, tags)

Remove tags from the specified legs. `legs` can be any vector, range, or tuple.
"""
function removeitag(q::TLArray{T, QD}, legs::LegList, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::TLArray, pred::Function, tags)

Remove tags from legs satisfying predicate.
"""
function removeitag(q::TLArray{T, QD}, pred::Function, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _remove_itag(base, tags))
end

"""
    replaceitag(q::TLArray, replacements::Pair...; dir, itag, plev, lock, rev=false)
    replaceitag(q::TLArray, replacements::AbstractDict; dir, itag, plev, lock, rev=false)

Replace tags on matching legs. Each replacement removes the tags in the source
query and adds the destination tags when the leg contains all tags in the source
query. For example, `"aaa,bbb"=>"ccc"` removes both `aaa` and `bbb` and adds
`ccc` only on legs containing both source tags.

Dictionary input applies each `key => value` replacement in iteration order.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
replaceitag(q, "site"=>"phys")
replaceitag(q, "aaa,bbb"=>"ccc")
replaceitag(q, "site1"=>"left", "site2"=>"right"; dir='-')
replaceitag(q, Dict("site"=>"phys"))
```
"""
function replaceitag(q::TLArray{T, QD}, replacements::ITagReplacementPair...;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, replacements::ITagReplacementDict;
                     dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                     rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, leg::Integer, replacements)

Replace tags on a single specified leg.
"""
function replaceitag(q::TLArray{T, QD}, leg::Integer,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, leg::Integer,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, (leg,), base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, legs::LegList, replacements)

Replace tags on the specified legs. `legs` can be any vector, range, or tuple.
"""
function replaceitag(q::TLArray{T, QD}, legs::LegList,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, legs::LegList,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, legs, base -> _replace_itag(base, replacements))
end

"""    replaceitag(q::TLArray, pred::Function, replacements)

Replace tags on legs satisfying predicate.
"""
function replaceitag(q::TLArray{T, QD}, pred::Function,
                     replacements::ITagReplacementPair...) where {T, QD}
    isempty(replacements) &&
        throw(ArgumentError("replaceitag requires at least one replacement pair"))
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end

function replaceitag(q::TLArray{T, QD}, pred::Function,
                     replacements::ITagReplacementDict) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _replace_itag(base, replacements))
end

"""
    setitag(q::TLArray, tags; dir, itag, plev, lock, rev=false)

Replace the entire tag string of matching legs with `tags`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setitag(q, "bond")              # set all legs to tag "bond"
setitag(q, ""; dir='+')         # clear tags on incoming legs
setitag(q, "phys"; itag="site") # rename "site" → "phys" (full replacement)
setitag(q, "aux"; dir='+', rev=true)  # set tag on all non-incoming legs
```
"""
function setitag(q::TLArray{T, QD}, tags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::TLArray, leg::Integer, tags)

Set the entire tag string of a single specified leg.
"""
function setitag(q::TLArray{T, QD}, leg::Integer, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, (leg,), _ -> norm)
end

"""    setitag(q::TLArray, legs::LegList, tags)

Set the entire tag string of the specified legs. `legs` can be any vector, range, or tuple.
"""
function setitag(q::TLArray{T, QD}, legs::LegList, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::TLArray, pred::Function, tags)

Set the entire tag string of legs satisfying predicate.
"""
function setitag(q::TLArray{T, QD}, pred::Function, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, findlegs(q, pred), _ -> norm)
end

# ─── Pretty-printing for TLArray ──────────────────────────────────────────────
#
# Format:
#
#   TLArray{T}  [Symm1, Symm2, ...]
#     leg 1:  dir  'tag'  (plev=k)
#     leg 2:  dir  'tag'
#     ...
#     1.  <sector inline>
#     2.  <sector inline>
#     ...
#
# When the number of sectors exceeds QSPACE_DISPLAY_HEAD + QSPACE_DISPLAY_TAIL,
# only the first QSPACE_DISPLAY_HEAD and last QSPACE_DISPLAY_TAIL sectors are shown.
# Set these globals to control the truncation behaviour.
# ─────────────────────────────────────────────────────────────────────────────

const QSPACE_DISPLAY_HEAD = Ref(5)   # number of first sectors to show
const QSPACE_DISPLAY_TAIL = Ref(5)   # number of last sectors to show

Base.show(io::IO, qs::TLArray) = show(io, MIME"text/plain"(), qs)

_qindex_plev_string(plev::Int) =
    plev == 0 ? "" : "p$(plev)"

_qindex_lock_string(lock::Int) = lock == 0 ? "" : "🔒$(lock)"

function _format_qindex(idx::TLIndex)
    return "\"$(idx.itags)$(idx.dir)\"$(_qindex_plev_string(idx.plev))$(_qindex_lock_string(idx.lock))"
end

# Special pretty-printing for 0-dimensional TLArray (scalar result of full contraction).
function Base.show(io::IO, ::MIME"text/plain", qs::TLArray{T, 0, N, N}) where {T, N}
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "0D TLArray{$T}, $N symmetries [$symm_names]: ", _fmt_scalar_str(qs[]))
end

function Base.show(io::IO, ::MIME"text/plain", qs::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    # --- Header: symmetries and leg dirs/tags on one line ---
    # Format:  TLArray{...}  [Sym1, Sym2]  ["tag1"+, "tag2"-', ...]
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "$(QD)D TLArray, $N symmetries [$symm_names]")
    leg_strs = map(qs.inds) do idx
        raw = _format_qindex(idx)
        idx.dual ? "\e[32m$(raw)\e[0m" : raw
    end
    println(io, "  [", join(leg_strs, ", "), "]")

    # --- Sectors: one per line with global label width for cross-sector alignment ---
    nr = nsectors(qs)
    if nr == 0
        print(io, "  (empty)")
        return
    end

    # Determine which sector indices to display.
    head = QSPACE_DISPLAY_HEAD[]
    tail = QSPACE_DISPLAY_TAIL[]
    truncate = nr > head + tail
    display_indices = if truncate
        vcat(1:head, nr-tail+1:nr)
    else
        1:nr
    end

    # Compute per-symmetry widths globally across displayed sectors only.
    widths = map(1:N) do n
        maximum(display_indices) do i
            _sector_label_widths(qs, i)[n]
        end
    end
    # Pre-compute scalar width for alignment (only sectors with scalar RMT).
    scalar_width = 0
    for i in display_indices
        rmt = sector_rmt(qs, i)
        length(rmt.data) == 1 || continue
        scalar_width = max(scalar_width, length(_fmt_scalar_str(only(rmt.data))))
    end

    for (k, i) in enumerate(display_indices)
        # Print ellipsis between head and tail blocks.
        if truncate && k == head + 1
            println(io)
            println(io, "  ⋮  ($(nr - head - tail) sectors omitted)")
        end
        rmt = sector_rmt(qs, i)
        om_dim = prod(size(rmt.data)[QD+1:end]; init=1)
        phys_str = join(size(rmt.data)[1:QD], "x")
        om_str   = om_dim > 1 ? " @$om_dim" : ""
        print(io, "  $i.\t", phys_str, om_str, "\t")
        _print_sector_cgt_dims(io, qs, i)
        _print_sector_qlabels(io, qs, i, widths)
        length(rmt.data) == 1 && print(io, "\t", lpad(_fmt_scalar_str(only(rmt.data)), scalar_width))
        QD == 2 && print(io, "\t√", _sector_cgt_size_2d(qs, i))
        k < length(display_indices) && println(io)
    end
end

# Normalize TLArray w-matrices in-place.
# - For 2D: set each wmat[1] = sqrt(dim) so the CGT contributes identity scaling.
# - For 0D: set all wmat elements to 1.0, absorbing factors into RMT.
# - For other dimensions: no-op.
function normalize_qspace!(q::TLArray{T, QD, N}) where {T, QD, N}
    if QD == 2
        for sector_index in 1:nsectors(q)
            for i in 1:N
                S = symm(q)[i]
                isabelian(S) && continue
                qlabels, _, _ = _sector_cgt_metadata(q, sector_index, i)
                q1, q2 = qlabels
                wmat = sector_wmat(q, sector_index, i)
                @assert q2 == q1 || q2 == get_dualq(S, q1)
                dim = dimension(S, q1); @assert dim == dimension(S, q2)
                @assert size(wmat) == (1, 1) # No outer multiplicity
                w_factor = sqrt(dim) / wmat[1]
                wmat[:] *= w_factor
                sector_rmt(q, sector_index)[:] /= w_factor
            end
        end
    elseif QD == 0
        for sector_index in 1:nsectors(q)
            for i in 1:N
                isabelian(symm(q)[i]) && continue
                wmat = sector_wmat(q, sector_index, i)
                @assert size(wmat) == (1, 1) "0D TLArray must have 1x1 w-matrices"
                w_val = wmat[1]
                if w_val != 1.0
                    wmat[:] .= 1.0
                    sector_rmt(q, sector_index)[:] .*= w_val
                end
            end
        end
    end
    # For other dimensions (QD != 0 and QD != 2), do nothing.
end

# Sort sectors of a TLArray in-place in dictionary order by physical leg qlabels.
# For each sector the sort key is built leg by leg (1 → QD): at leg l, the key
# is the tuple of qlabels across all symmetries, i.e.
#   (CGT metadata[1].qlabels[cgp₁[l]], CGT metadata[2].qlabels[cgp₂[l]], ...)
# Comparison is then lexicographic over (leg 1 key, leg 2 key, ..., leg QD key).
function sort_sectors!(q::TLArray{T, QD, N}) where {T, QD, N}
    perm = sortperm(collect(1:nsectors(q)); by = sector_index -> Tuple(
            sector_qlabel(q, sector_index, l)
        for l in QD:-1:1)
    )
    q.qlabels[:, :] = q.qlabels[:, perm]
    q.wmats[:, :] = q.wmats[:, perm]
    q.RMTs[:] = q.RMTs[perm]
    return q
end

# Scalar multiplication and division: only the RMT arrays are scaled.
# CGT metadata (w-matrices, qlabels) are left untouched.
function Base.:*(qs::TLArray, fac::Number)
    result = deepcopy(qs)
    for rmt in result.RMTs
        rmt.data .*= fac
    end
    return result
end
Base.:*(fac::Number, qs::TLArray) = qs * fac
Base.:/(qs::TLArray, fac::Number) = qs * (1 / fac)
Base.:-(qs::TLArray) = qs * -1

# Return a deep copy of a TLArray (CGT metadata, RMTs, indices, spaces all copied).
Base.copy(q::TLArray) = deepcopy(q)

function _identity_on_qspace(q::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    @assert QD == 2 "Scalar add/subtract is only defined for rank-2 TLArray objects"

    in_legs  = findlegs(q; dir='+')
    out_legs = findlegs(q; dir='-')
    @assert length(in_legs) == 1 && length(out_legs) == 1 "Scalar add/subtract requires exactly one incoming and one outgoing leg"

    in_leg  = only(in_legs)
    out_leg = only(out_legs)
    @assert q.spaces[in_leg] == q.spaces[out_leg] "Scalar add/subtract requires matching incoming and outgoing spaces"

    id_q = getIdentity((q, out_leg); itag=q.inds[out_leg].itags)
    return TLArray(id_q, (q.inds[in_leg], q.inds[out_leg]))
end

# ─── TLArray norm ─────────────────────────────────────────────────────────────
#
# Exploits the Wigner-Eckart decomposition to compute the Frobenius norm
# directly from the reduced matrix elements (RMTs) without building the full
# dense tensor.
#
# For QD ≠ 2 (including QD = 0 and QD ≥ 3):
#   Sectors from different q-label sectors are orthogonal by symmetry.  Within
#   each sector, the CGT canonical basis elements are orthonormal and the
#   w-matrix is left-orthogonal (Uᵀ·U = I), so the M columns of cgt_block
#   are also orthonormal.  Therefore all CGT cross-terms vanish and:
#       ‖A‖² = Σ_r ‖RMT_r‖²
#
# For QD = 2 (2-leg operators / matrices):
#   normalize_qspace! sets wmat[1,1] = √dim for each sector, turning the 2D CGT
#   block into a (dim × dim) identity matrix  (‖Id_dim‖² = dim).
#   Consequently:
#       ‖A‖² = Σ_r dim_r · ‖RMT_r‖²
#   where dim_r = _cgt_size_2d(r.CGT metadata, symm(q)) = ∏_{non-abelian n} d_leg1^(n).
#
# ─────────────────────────────────────────────────────────────────────────────
function LinearAlgebra.norm(q::TLArray{T, QD, N}) where {T, QD, N}
    s = zero(Float64)
    if QD == 2
        for sector_index in 1:nsectors(q)
            d = _sector_cgt_size_2d(q, sector_index)
            s += d * sum(abs2, sector_rmt(q, sector_index).data)
        end
    else
        for sector_index in 1:nsectors(q)
            s += sum(abs2, sector_rmt(q, sector_index).data)
        end
    end
    return sqrt(s)
end

# ─── TLArray addition ─────────────────────────────────────────────────────────
#
# Each sector is a TLArray record uniquely identified by its physical q-labels
# (CGT metadata[n].qlabels with cgp applied) across all symmetries.
#
# Per the Wigner-Eckart decomposition (paper Eq. 22):
#   X = ⊕_q [ ‖X‖_{qμ} ⊗ (w_{μμ'} C_{qμ'}) ]
#
# For a sector present in both operands, both sectors reference the same underlying
# sorted CGT.  The sum pools the two (w, RMT) contributions and finds a minimal
# new w-matrix via SVD using _compress_sector / merge_new_sector (same routine as
# used for contractions).  A sector present in only one operand is copied as-is.
#
# If the two Telum share the same indices but in different order, the second
# operand is permuted to match the first before addition.
# ─────────────────────────────────────────────────────────────────────────────

# Find the unique permutation perm such that qs2.inds[perm[i]] == inds1[i]
# and qs2.spaces[perm[i]] == spaces1[i] for all i.  Raises an error if no
# such bijection exists or if it is not unique.
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
            "No leg in second TLArray matches leg $i of first TLArray " *
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

function Base.:+(qs1::TLArray{T1, QD, N, RD, QT, PS},
                 qs2::TLArray{T2, QD, N, RD, QT, PS}) where {T1, T2, QD, N, RD, QT, PS}
    @assert symm(qs1) == symm(qs2) "TLArray objects must share the same symmetry tuple"

    if qs1.inds != qs2.inds || qs1.spaces != qs2.spaces
        perm = _find_leg_permutation(qs1.inds, qs1.spaces, qs2.inds, qs2.spaces)
        qs2  = permutedims(qs2, perm)
    end

    T = promote_type(T1, T2)

    sector_key(q, sector_index) = ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD))

    dict1 = Dict(sector_key(qs1, sector_index) => sector_index for sector_index in 1:nsectors(qs1))
    dict2 = Dict(sector_key(qs2, sector_index) => sector_index for sector_index in 1:nsectors(qs2))

    result_qlabel_sectors = Vector{NTuple{QD, QT}}()
    result_wmat_buffers = _wmat_buffers(PS)
    result_RMTs = LurTensor{T, RD, Array{T, RD}}[]

    for key in union(keys(dict1), keys(dict2))
        in1 = haskey(dict1, key)
        in2 = haskey(dict2, key)

        if in1 && !in2
            # Sector exists only in qs1 — copy with promoted element type.
            sector_index = dict1[key]
            push!(result_qlabel_sectors, ntuple(l -> sector_qlabel(QT, qs1, sector_index, l), Val(QD)))
            for n in 1:N
                _push_wmat!(result_wmat_buffers, PS, n, deepcopy(sector_wmat(qs1, sector_index, n)))
            end
            push!(result_RMTs, LurTensor(T.(sector_rmt(qs1, sector_index).data)))

        elseif in2 && !in1
            # Sector exists only in qs2 — copy with promoted element type.
            sector_index = dict2[key]
            push!(result_qlabel_sectors, ntuple(l -> sector_qlabel(QT, qs2, sector_index, l), Val(QD)))
            for n in 1:N
                _push_wmat!(result_wmat_buffers, PS, n, deepcopy(sector_wmat(qs2, sector_index, n)))
            end
            push!(result_RMTs, LurTensor(T.(sector_rmt(qs2, sector_index).data)))

        else
            # Sector exists in both — merge the two (w, RMT) contributions.
            # Both sectors reference the same underlying sorted CGT, so the
            # stored qlabels and cgp (which encode the permutation to that
            # sorted CGT) must agree.
            i1, i2 = dict1[key], dict2[key]
            for n in 1:N
                qlabels1, cgp1, _ = _sector_cgt_metadata(qs1, i1, n)
                qlabels2, cgp2, _ = _sector_cgt_metadata(qs2, i2, n)
                @assert qlabels1 == qlabels2 "qlabels mismatch at symmetry $n for sector $key"
                @assert cgp1     == cgp2     "cgp mismatch at symmetry $n for sector $key"
            end

            # Pool w-matrices and RMTs as two contributions, then compress.
            # TODO: When GPU support is added, 'Array{Float64, 2}' here can be generalized
            new_wmats = ntuple(Val(N)) do n
                LurTensor{Float64, 2, Array{Float64, 2}}[sector_wmat(qs1, i1, n), sector_wmat(qs2, i2, n)]
            end
            new_RMTs  = LurTensor{T, RD, Array{T, RD}}[LurTensor(T.(sector_rmt(qs1, i1).data)),
                                                       LurTensor(T.(sector_rmt(qs2, i2).data))]
            new_qlabs = ntuple(Val(N)) do n
                _sector_cgt_metadata(qs1, i1, n)
            end

            compressed = _compress_sector(new_wmats, new_RMTs, QD)
            if !isnothing(compressed)
                U_mats, result_RMT = compressed
                push!(result_qlabel_sectors,
                      _physical_qlabels_from_cgt_metadata(QT, new_qlabs, Val(QD), Val(N)))
                for n in 1:N
                    _push_wmat!(result_wmat_buffers, PS, n, U_mats[n])
                end
                push!(result_RMTs, result_RMT)
            end
        end
    end

    result_qlabels = Matrix{QT}(undef, QD, length(result_qlabel_sectors))
    for sector_index in eachindex(result_qlabel_sectors), leg in 1:QD
        result_qlabels[leg, sector_index] = result_qlabel_sectors[sector_index][leg]
    end

    result_wmats = _wmat_matrix_from_buffers(PS, result_wmat_buffers, length(result_RMTs))
    return _field_tlarray(symm(qs1), result_qlabels, result_wmats, result_RMTs,
                          qs1.inds, qs1.spaces)
end

Base.:-(qs1::TLArray, qs2::TLArray) = qs1 + (-1 * qs2)
Base.:+(q::TLArray, x::Number) = q + x * _identity_on_qspace(q)
Base.:+(x::Number, q::TLArray) = q + x
Base.:-(q::TLArray, x::Number) = q + (-x) * _identity_on_qspace(q)
Base.:-(x::Number, q::TLArray) = x * _identity_on_qspace(q) + (-q)

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
_qspace_eltype(::TLArray{T}) where {T} = T

function _oplus_pad_qspace(q::TLArray{T, QD, N, RD},
                           result_spaces,
                           dims_tuple,
                           start_dim_maps,
                           result_dim_maps) where {T, QD, N, RD}
    dims_set = Set(dims_tuple)
    qlabels = copy(q.qlabels)
    wmats = deepcopy(q.wmats)
    RMTs = similar(q.RMTs, nsectors(q))
    for sector_index in 1:nsectors(q)
        old_sizes = size(sector_rmt(q, sector_index).data)
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
        new_data[fill_inds...] = sector_rmt(q, sector_index).data
        RMTs[sector_index] = LurTensor(new_data)
    end
    return _field_tlarray(symm(q), qlabels, wmats, RMTs, q.inds, _copy_spaces_tuple(result_spaces))
end

function _zero_qspace_with_spaces(symm::NTuple{N, Any},
                                  inds::NTuple{QD, TLIndex},
                                  spaces::NTuple{QD, Vector};
                                  T::Type=Float64) where {N, QD}
    QT = qlabeltype(symm)
    qlabels = Matrix{QT}(undef, QD, 0)
    wmats = _wmat_matrix(symm, 0)
    RMTs = LurTensor{T, QD + N, Array{T, QD + N}}[]
    return _field_tlarray(symm, qlabels, wmats, RMTs, inds, _copy_spaces_tuple(spaces))
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
            "(same itag, direction, prime level, and lock required; green ignored)"))
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
    first(qs) isa TLArray || throw(ArgumentError("oplus entry 1 is not a TLArray"))

    ref = first(qs)
    aligned = Vector{TLArray}(undef, length(qs))
    aligned[1] = ref
    for i in 2:length(qs)
        q = qs[i]
        q isa TLArray || throw(ArgumentError("oplus entry $i is not a TLArray"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "TLArray entry $i has a different symmetry tuple"))
        perm = _find_oplus_leg_permutation(ref.inds, q.inds, i)
        aligned[i] = perm == ntuple(identity, length(ref.inds)) ? q : permutedims(q, perm)
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
            "(same itag, direction, prime level, and lock required; green ignored)"))
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
    T = promote_type((_qspace_eltype(q) for q in qs)...)

    acc = _zero_qspace_with_spaces(symm(ref), ref.inds, result_spaces; T=T)
    for (q, qstarts) in zip(qs, start_maps)
        padded = _oplus_pad_qspace(q, result_spaces, dims_tuple, qstarts, result_dim_maps)
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
    val isa TLArray || throw(ArgumentError(
        "matrix oplus entry ($i, $j) is neither a TLArray nor an undefined entry"))
    return val
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

function oplus(q1::TLArray, q2::TLArray, dimensions)
    return oplus(TLArray[q1, q2], dimensions)
end

function oplus(q1::TLArray, q2::TLArray; dir=nothing, itag=nothing,
               plev=nothing, lock=nothing, rev::Bool=false)
    return oplus(TLArray[q1, q2]; dir=dir, itag=itag, plev=plev,
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

    T = promote_type((_qspace_eltype(q) for q in defined_qs)...)
    filled = Matrix{TLArray}(undef, size(mat, 1), size(mat, 2))
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        if q === nothing
            spaces = _infer_zero_matrix_spaces(first_axis_sources, second_axis_sources, i, j, length(ref.inds))
            filled[i, j] = _zero_qspace_with_spaces(symm(ref), ref.inds, spaces; T=T)
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



# ─── conj / adjoint ──────────────────────────────────────────────────────────
#
# conj(q): conjugate a TLArray object.
#   1. Every leg direction is reversed ('+' ↔ '-').
#   2. Every RMT entry is complex-conjugated.
#   3. When the old incoming qlabels equal the old outgoing qlabels
#      (self-conjugate CGT), the canonical OM ordering of the conjugated CGT
#      differs from the old one by a transposition within each central-space
#      block. This is applied as a sector permutation on the first dimension of
#      the w-matrix.
#
# adjoint(q): for TLArray tensors, defined as conj(q).
# ─────────────────────────────────────────────────────────────────────────────
function Base.conj(q::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    new_inds = ntuple(l -> change_dir(q.inds[l]), QD)

    qlabels = copy(q.qlabels)
    wmats = _wmat_matrix(productsymm(q), nsectors(q))
    RMTs = similar(q.RMTs, nsectors(q))
    for sector_index in 1:nsectors(q)
        RMTs[sector_index] = LurTensor(conj.(sector_rmt(q, sector_index).data))

        for n in 1:N
            ordered_qlabels, _, legdir = _sector_cgt_metadata(q, sector_index, n)
            m, k  = legdir

            # 3. Permute the OM (first) axis of the w-matrix when the old
            #    incoming and outgoing qlabels are identical tuples.  In that
            #    case the new CGT references the same canonical CGTom but with
            #    the flat OM ordering transposed within every central-space block:
            #      old flat = start + (upidx-1) + (dnidx-1)*om_up   [upidx fast]
            #      new flat = start + (dnidx-1) + (upidx-1)*om_dn   [dnidx fast]
            S = symm(q)[n]
            ins, outs = ordered_qlabels[1:m], ordered_qlabels[m+1:m+k]
            ins_, _ = remove_zeros(S, ins)
            outs_, _ = remove_zeros(S, outs)
            new_wmat =
                if !isabelian(S) && ins_ == outs_
                    is1j = detect_1j(S, ins_, outs_)
                    cgt_oms  = get_CGTom(S, ins_, outs_, is1j)
                    perm_vec = get_conj_perm(cgt_oms)
                    LurTensor(sector_wmat(q, sector_index, n).data[perm_vec, :])
                else
                    deepcopy(sector_wmat(q, sector_index, n))
                end

            _set_sector_wmat!(wmats, productsymm(q), sector_index, n, new_wmat)
        end
    end

    # spaces remain the same: physical qlabels at each leg don't change in conj,
    # only the CGT internal structure (incoming/outgoing) changes
    return _field_tlarray(symm(q), qlabels, wmats, RMTs, new_inds, _copy_spaces_tuple(q.spaces))
end

Base.adjoint(q::TLArray) = conj(q)

getsub(q::TLArray, selector) = TLArray(q, selector)

function _normalize_getsub_index(i::Int, dim::Int, sector, leg::Int)
    i == 0 && throw(ArgumentError(
        "selector for sector $sector on leg $leg cannot contain 0"))
    return i < 0 ? dim + i + 1 : i
end

function _normalize_getsub_indices(raw, dim::Int, sector, leg::Int)
    inds = if raw isa Integer
        [Int(raw)]
    elseif raw isa Tuple && all(i -> i isa Integer, raw)
        Int[Int(i) for i in raw]
    elseif raw isa AbstractRange{<:Integer}
        Int[i for i in raw]
    elseif raw isa AbstractVector{<:Integer}
        Int[i for i in raw]
    else
        throw(ArgumentError(
            "selector for sector $sector on leg $leg must be :, Int, integer tuple, AbstractRange, or AbstractVector{<:Integer}"))
    end

    isempty(inds) && throw(ArgumentError(
        "selector for sector $sector on leg $leg must not be empty"))
    inds = [_normalize_getsub_index(i, dim, sector, leg) for i in inds]
    length(unique(inds)) == length(inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains duplicate indices"))
    all(1 <= i <= dim for i in inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains out-of-bounds indices for dimension $dim"))
    return inds
end
function _normalize_getsub_predicate_pick(raw, dim::Int, sector, leg::Int)
    raw isa Bool && throw(ArgumentError("getsub predicate for sector $sector on leg $leg must not return Bool; use Colon() or nothing explicitly"))
    raw === nothing && return nothing
    raw isa Colon && return Colon()
    return _normalize_getsub_indices(raw, dim, sector, leg)
end

function _collect_getsub_predicate_picks(q::TLArray, positions, pred::Function)
    selected_picks = Dict{Int, Dict{Any, Any}}()
    for leg in positions
        picks = Dict{Any, Any}()
        for (sector, dim) in q.spaces[leg]
            pick = _normalize_getsub_predicate_pick(pred(sector), dim, sector, leg)
            isnothing(pick) && continue
            picks[sector] = pick
        end
        selected_picks[leg] = picks
    end
    return selected_picks
end

function _apply_getsub_picks(q::TLArray{T, QD, N, RD, QT},
                             positions,
                             selected_picks::Dict{Int, Dict{Any, Any}};
                             preserve_space::Bool=false) where {T, QD, N, RD, QT}
    if preserve_space
        for leg in positions
            for (sector, pick) in selected_picks[leg]
                pick isa Colon && continue
                throw(ArgumentError(
                    "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
            end
        end
    end

    selected_leg_set = Set(positions)
    qlabels_out = NTuple{QD, QT}[]
    wmats_out = _wmat_buffers(productsymm(q))
    RMTs_out = LurTensor{T, RD, Array{T, RD}}[]
    for sector_index in 1:nsectors(q)
        picks_by_leg = Dict{Int, Any}()
        keep = true
        for leg in positions
            sector = sector_qlabel(q, sector_index, leg)
            picks = selected_picks[leg]
            haskey(picks, sector) || (keep = false; break)
            picks_by_leg[leg] = picks[sector]
        end
        keep || continue
        selectors = ntuple(d -> get(picks_by_leg, d, Colon()), RD)
        push!(qlabels_out, ntuple(l -> sector_qlabel(q, sector_index, l), Val(QD)))
        for n in 1:N
            _push_wmat!(wmats_out, productsymm(q), n, sector_wmat(q, sector_index, n))
        end
        push!(RMTs_out, LurTensor(sector_rmt(q, sector_index).data[selectors...]))
    end

    spaces_out = if preserve_space
        _copy_spaces_tuple(q.spaces)
    else
        ntuple(l -> begin
            if l in selected_leg_set
                out = eltype(q.spaces[l])[]
                picks = selected_picks[l]
                for (sector, dim) in q.spaces[l]
                    haskey(picks, sector) || continue
                    pick = picks[sector]
                    push!(out, (sector, pick isa Colon ? dim : length(pick)))
                end
                out
            else
                copy(q.spaces[l])
            end
        end, QD)
    end

    qlabels = Matrix{QT}(undef, QD, length(qlabels_out))
    for (sector_index, sector) in enumerate(qlabels_out), leg in 1:QD
        qlabels[leg, sector_index] = sector[leg]
    end
    wmats = _wmat_matrix_from_buffers(productsymm(q), wmats_out, length(RMTs_out))
    return _field_tlarray(symm(q), qlabels, wmats, RMTs_out, q.inds, spaces_out)
end

function _normalize_getsub_predicate_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("getsub requires at least one leg"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "getsub legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "getsub legs must be unique, got $positions"))
    return positions
end

"""
    getsub(q::TLArray, leg::Integer, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sector on `leg` satisfies
`pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), only `q.spaces[leg]` is truncated to
the retained sectors and all other leg-space lists are copied unchanged. If
`preserve_space=true`, all cached leg-space lists are preserved exactly and only
the sectors are filtered. This requires `pred` to keep whole sectors, so any
index-selection return value is rejected when `preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, leg::Integer, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    return getsub(q, (Int(leg),), pred; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, legs, pred::Function; preserve_space::Bool=false) -> TLArray

Return a new `TLArray` containing only sectors whose sectors on every selected leg
satisfy `pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), each selected leg keeps only the
matching entries in its space list, while unselected legs keep copies of their
original space lists. If `preserve_space=true`, all cached leg-space lists are
preserved exactly and only the sectors are filtered. This requires `pred` to keep
whole sectors, so any index-selection return value is rejected when
`preserve_space=true`.
"""
function getsub(q::TLArray{T, QD, N, RD}, legs::LegList, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    positions = _normalize_getsub_predicate_legs(q, legs)
    selected_picks = _collect_getsub_predicate_picks(q, positions, pred)
    return _apply_getsub_picks(q, positions, selected_picks; preserve_space=preserve_space)
end

"""
    getsub(q::TLArray, pred::Function; preserve_space::Bool=false, dir=nothing,
           itag=nothing, plev=nothing, lock=nothing, rev=false) -> TLArray

Apply predicate-based `getsub` to every leg selected by the keyword criteria.
The leg selection follows the same matching rules as `findlegs`.
"""
function getsub(q::TLArray{T, QD, N, RD}, pred::Function; preserve_space::Bool=false,
                dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                rev::Bool=false) where {T, QD, N, RD}
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="getsub")
    return getsub(q, legs, pred; preserve_space=preserve_space)
end

"""
    empty_qspace(symm::NTuple{N, Any}, inds::NTuple{QD, TLIndex}; T::Type=Float64) where {N, QD}

Create an empty rank-`QD` (zero-sector) TLArray over the given symmetries.

`symm` is an `N`-tuple of symmetry types (e.g. `(SU{2}, U1)`); `inds` is a
`QD`-tuple of `TLIndex` objects describing the leg directions, tags, and prime
levels.  All `TLIndex` entries with non-empty tags must be pairwise distinct.

The element type of future sector data defaults to `Float64`; pass `T=ComplexF64`
(or another concrete `<:Number` type) to use a different element type.
"""
function empty_qspace(symm::NTuple{N, Any}, inds::NTuple{QD, TLIndex};
                      T::Type=Float64) where {N, QD}
    RD = QD + N
    QT = qlabeltype(symm)
    spaces = ntuple(_ -> Vector{Tuple{QT, Int}}(), QD)
    qlabels = Matrix{QT}(undef, QD, 0)
    wmats = _wmat_matrix(symm, 0)
    RMTs = LurTensor{T, RD, Array{T, RD}}[]
    return _field_tlarray(symm, qlabels, wmats, RMTs, inds, spaces)
end

function empty_qspace(q::TLArray; T::Type=Float64)
    return empty_qspace(symm(q), q.inds; T=T)
end

function Base.zero(q::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    qlabels = Matrix{qlabeltype(q)}(undef, QD, 0)
    wmats = _wmat_matrix(symm(q), 0)
    RMTs = LurTensor{T, RD, Array{T, RD}}[]
    return _field_tlarray(symm(q), qlabels, wmats, RMTs, q.inds, _copy_spaces_tuple(q.spaces))
end

"""
    qlabeltype(symm::NTuple{N, Any}) where {N}
    qlabeltype(q::TLArray)

Return the qlabel type for one leg sector over the symmetries in `symm` or `q`.

For example, `(U1, SU{3})` returns `Tuple{Tuple{Int}, NTuple{2, Int}}`.
"""
function qlabeltype(symm::NTuple{N, Any}) where {N}
    return Tuple{ntuple(n -> NTuple{nzops(symm[n]), Int}, N)...}
end

qlabeltype(::Type{<:ProductSymm{Syms}}) where {Syms} =
    Tuple{(NTuple{nzops(S), Int} for S in Syms.parameters)...}

qlabeltype(::TLArray{T, QD, N, RD, QT}) where {T, QD, N, RD, QT} = QT

"""
    zero_qlabels(symm::NTuple{N, Any}) where {N}
    zero_qlabels(q::TLArray)

Return the trivial qlabel for each symmetry in `symm` or `q`.

For example, `(SU{2}, SU{3})` returns `((0,), (0, 0))`.
"""
function zero_qlabels(symm::NTuple{N, Any}) where {N}
    return ntuple(n -> Tuple(0 for _ in 1:nzops(symm[n])), N)
end

zero_qlabels(q::TLArray) = zero_qlabels(symm(q))

function _is_singleton_leg(q::TLArray{T, QD, N}, leg::Int) where {T, QD, N}
    1 <= leg <= QD || throw(ArgumentError("leg must lie in 1:$QD, got $leg"))
    return length(q.spaces[leg]) == 1 && only(q.spaces[leg]) == (zero_qlabels(q), 1)
end

_singleton_legs(q::TLArray{T, QD}) where {T, QD} = [leg for leg in 1:QD if _is_singleton_leg(q, leg)]

function _normalize_delete_singleton_legs(q::TLArray{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one deletion leg must be specified"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "singleton deletion legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "singleton deletion legs must be unique, got $positions"))
    sort!(positions)
    return positions
end


function _delete_singleton_rmt(rmt::LurTensor{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    selectors = ntuple(axis -> axis <= qd && axis ∈ positions ? 1 : Colon(), qd + n_symm)
    return LurTensor(copy(rmt.data[selectors...]))
end

function _delete_singleton_impl(q::TLArray{T, QD, N, RD}, positions) where {T, QD, N, RD}
    new_qd = QD - length(positions)
    new_rd = RD - length(positions)

    keep_inds = [q.inds[leg] for leg in 1:QD if leg ∉ positions]
    QT = qlabeltype(q)
    keep_spaces = Vector{Tuple{QT, Int}}[q.spaces[leg] for leg in 1:QD if leg ∉ positions]

    qlabels = Matrix{QT}(undef, new_qd, nsectors(q))
    wmats = _wmat_matrix(productsymm(q), nsectors(q))
    RMTs = Vector{LurTensor{T, new_rd, Array{T, new_rd}}}(undef, nsectors(q))
    keep_legs = [leg for leg in 1:QD if leg ∉ positions]
    for sector_index in 1:nsectors(q)
        for (new_leg, old_leg) in enumerate(keep_legs)
            qlabels[new_leg, sector_index] = sector_qlabel(q, sector_index, old_leg)
        end
        for n in 1:N
            _set_sector_wmat!(wmats, productsymm(q), sector_index, n,
                              LurTensor(copy(sector_wmat(q, sector_index, n).data)))
        end
        RMTs[sector_index] = _delete_singleton_rmt(sector_rmt(q, sector_index), positions, QD, N)
    end

    return _field_tlarray(symm(q), qlabels, wmats, RMTs, Tuple(keep_inds), Tuple(keep_spaces))
end

"""
    deleteSingleton(q::TLArray; dir=nothing, itag=nothing, plev=nothing) -> TLArray

Delete singleton legs matching the supplied criteria.

With no keyword arguments, all singleton legs are deleted.
Only singleton legs are eligible for deletion. If none match, a warning is
emitted and `q` is returned unchanged.
"""
function deleteSingleton(q::TLArray{T, QD}; dir=nothing, itag=nothing, plev=nothing) where {T, QD}
    singleton_legs = if isnothing(dir) && isnothing(itag) && isnothing(plev)
        _singleton_legs(q)
    else
        candidate_legs = findlegs(q; dir=dir, itag=itag, plev=plev)
        [leg for leg in candidate_legs if _is_singleton_leg(q, leg)]
    end

    if isempty(singleton_legs)
        if isnothing(dir) && isnothing(itag) && isnothing(plev)
            @warn "deleteSingleton: no singleton legs found"
        else
            @warn "deleteSingleton: no singleton legs matched the requested criteria"
        end
        return q
    end
    return _delete_singleton_impl(q, singleton_legs)
end

"""
    deleteSingleton(q::TLArray, leg::Integer) -> TLArray
    deleteSingleton(q::TLArray, legs::LegList) -> TLArray

Delete the specified singleton legs from `q`.

Every selected leg must be singleton. Otherwise an `ArgumentError` is thrown.
"""
function deleteSingleton(q::TLArray, leg::Integer)
    return deleteSingleton(q, (leg,))
end

function deleteSingleton(q::TLArray{T, QD}, legs::LegList) where {T, QD}
    positions = _normalize_delete_singleton_legs(q, legs)
    bad = [leg for leg in positions if !_is_singleton_leg(q, leg)]
    isempty(bad) || throw(ArgumentError(
        "deleteSingleton requires singleton legs, but legs $bad are not singleton"))
    return _delete_singleton_impl(q, positions)
end

function _expand_singleton_kw(values, count::Int, name::AbstractString)
    if values isa AbstractString || values isa Char || values isa Integer
        return fill(values, count)
    end
    collected = collect(values)
    length(collected) == count || throw(ArgumentError(
        "$name must have length $count, got $(length(collected))"))
    return collected
end

function _singleton_insert_spec(q::TLArray{T, QD}, legs;
                                itag="", plev=0, lock=0, dir='+') where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one insertion leg must be specified"))

    count = length(positions)
    itag_vec = _expand_singleton_kw(itag, count, "itag")
    plev_vec = _expand_singleton_kw(plev, count, "plev")
    lock_vec = _expand_singleton_kw(lock, count, "lock")
    dir_vec  = _expand_singleton_kw(dir,  count, "dir")

    perm = sortperm(positions)
    positions = positions[perm]
    itag_vec = itag_vec[perm]
    plev_vec = plev_vec[perm]
    lock_vec = lock_vec[perm]
    dir_vec  = dir_vec[perm]

    final_qd = QD + count
    all(p -> 1 <= p <= final_qd, positions) || throw(ArgumentError(
        "singleton insertion legs must lie in 1:$final_qd, got $positions"))
    length(unique(positions)) == count || throw(ArgumentError(
        "singleton insertion legs must be unique, got $positions"))

    for direction in dir_vec
        direction in ('+', '-') || throw(ArgumentError(
            "added leg directions must be '+' or '-', got '$direction'"))
    end

    return positions, itag_vec, plev_vec, lock_vec, dir_vec
end

function _insert_singleton_rmt(rmt::LurTensor{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    old_phys = size(rmt.data)[1:qd]
    om_dims = size(rmt.data)[qd+1:qd+n_symm]

    new_phys = Int[]
    sizehint!(new_phys, qd + length(positions))

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:(qd + length(positions))
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            push!(new_phys, 1)
            insert_idx += 1
        else
            push!(new_phys, old_phys[old_leg])
            old_leg += 1
        end
    end

    new_data = copy(reshape(rmt.data, Tuple(vcat(new_phys, collect(om_dims)))))
    return LurTensor(new_data)
end

function _convert_rank2_singleton_normalization!(new_wmats, new_rmt::LurTensor, old_wmats)
    for n in eachindex(new_wmats)
        w_val = old_wmats[n][1]
        new_wmats[n][:] .= 1.0
        new_rmt[:] .*= w_val
    end
    return nothing
end

"""
    addSingleton(q::TLArray; nlegs=1, itag="", plev=0, lock=0, dir='+')

Append one or more singleton trivial legs to `q`.

The `nlegs` keyword controls how many singleton legs are created. New legs are
added to the end of the leg list. `itag`, `plev`, `lock`, and `dir` may each be
either a scalar applied to every added leg or an iterable with one value per
inserted leg.
"""
function addSingleton(q::TLArray{T, QD}; nlegs::Integer=1,
                      itag="", plev=0, lock=0, dir='+') where {T, QD}
    count = Int(nlegs)
    count >= 1 || throw(ArgumentError("nlegs must be at least 1, got $nlegs"))
    positions = (QD + 1):(QD + count)
    return addSingleton(q, positions; itag=itag, plev=plev, lock=lock, dir=dir)
end

"""
    addSingleton(q::TLArray, legs; itag="", plev=0, lock=0, dir='+')

Insert one or more singleton trivial legs into `q`.

`legs` may be a single integer or any iterable of integers. Each value is the
position of an added leg in the output tensor, whose rank is `ndims(q) +
length(legs)`. The original legs keep their relative order.

`itag`, `plev`, `lock`, and `dir` may each be either a scalar applied to
every added leg or an iterable with one value per inserted leg.
"""
function addSingleton(q::TLArray{T, QD, N, RD}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD, N, RD}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)

    new_qd = QD + length(positions)
    new_rd = RD + length(positions)
    trivial_qlabels = zero_qlabels(q)

    new_inds = Vector{TLIndex}(undef, new_qd)
    QT = qlabeltype(q)
    new_spaces = Vector{Vector{Tuple{QT, Int}}}(undef, new_qd)
    singleton_space = [(trivial_qlabels, 1)]

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            new_inds[new_leg] = TLIndex(String(itag_vec[insert_idx]), dir_vec[insert_idx],
                                       plev_vec[insert_idx], lock_vec[insert_idx])
            new_spaces[new_leg] = copy(singleton_space)
            insert_idx += 1
        else
            new_inds[new_leg] = q.inds[old_leg]
            new_spaces[new_leg] = q.spaces[old_leg]
            old_leg += 1
        end
    end

    qlabels = Matrix{QT}(undef, new_qd, nsectors(q))
    wmats = _wmat_matrix(productsymm(q), nsectors(q))
    RMTs = Vector{LurTensor{T, new_rd, Array{T, new_rd}}}(undef, nsectors(q))
    for sector_index in 1:nsectors(q)
        old_wmats = ntuple(n -> sector_wmat(q, sector_index, n), Val(N))
        new_wmats = ntuple(n -> LurTensor(copy(old_wmats[n].data)), Val(N))
        new_rmt = _insert_singleton_rmt(sector_rmt(q, sector_index), positions, QD, N)
        if QD == 2 && new_qd > 2
            _convert_rank2_singleton_normalization!(new_wmats, new_rmt, old_wmats)
        end
        old_leg = 1
        insert_idx = 1
        for new_leg in 1:new_qd
            if insert_idx <= length(positions) && positions[insert_idx] == new_leg
                qlabels[new_leg, sector_index] = trivial_qlabels
                insert_idx += 1
            else
                qlabels[new_leg, sector_index] = sector_qlabel(q, sector_index, old_leg)
                old_leg += 1
            end
        end
        for n in 1:N
            _set_sector_wmat!(wmats, productsymm(q), sector_index, n, new_wmats[n])
        end
        RMTs[sector_index] = new_rmt
    end

    return _field_tlarray(symm(q), qlabels, wmats, RMTs, Tuple(new_inds), Tuple(new_spaces))
end

"""
    getvac(q::TLArray, itags::Tuple{Vararg{AbstractString, 2}}=("", "")) -> TLArray

Build the rank-2 vacuum TLArray associated with `q`.

The result keeps the same symmetry tuple as `q`, has one incoming leg and one
outgoing leg, and contains exactly one trivial sector with RMT dimension 1 on
each leg. If `itags` is provided, it is used as the tags of the two legs.
"""
function getvac(q::TLArray{T, QD, N, RD},
                itags::Tuple{Vararg{AbstractString, 2}}=("", "")) where {T, QD, N, RD}
    trivial_qlabels = zero_qlabels(q)
    space_entry = (trivial_qlabels, 1)

    qlabels = Matrix{qlabeltype(q)}(undef, 2, 1)
    qlabels[1, 1] = trivial_qlabels
    qlabels[2, 1] = trivial_qlabels
    wmats = _wmat_matrix(symm(q), 1)
    for n in 1:N
        _set_sector_wmat!(wmats, productsymm(q), 1, n, LurTensor([1.0;;]))
    end
    rmt_data = fill(one(T), ntuple(_ -> 1, N + 2))
    RMTs = LurTensor{T, N + 2, Array{T, N + 2}}[LurTensor(rmt_data)]
    inds = (TLIndex(itags[1], '+'), TLIndex(itags[2], '-'))
    QT = qlabeltype(q)
    space_template = Vector{Tuple{QT, Int}}([space_entry])
    spaces = (copy(space_template), copy(space_template))

    return _field_tlarray(symm(q), qlabels, wmats, RMTs, inds, spaces)
end

function ⊗(q1::TLArray{T1, QD1, N, RD1},
           q2::TLArray{T2, QD2, N, RD2}) where {T1, T2, QD1, QD2, N, RD1, RD2}
    @assert symm(q1) == symm(q2) "TLArray objects must share the same symmetry tuple"

    q1_ext = addSingleton(q1, QD1 + 1; dir='-')
    q2_ext = addSingleton(q2, 1; dir='+')
    return contract(q1_ext, (QD1 + 1,), q2_ext, (1,))
end

LinearAlgebra.kron(q1::TLArray, q2::TLArray) = q1 ⊗ q2

include("getLocalSpace.jl")
include("getIdentity.jl")
include("contract.jl")
include("get1jtensor.jl")
include("svd.jl")
include("eig.jl")
include("permute.jl")
