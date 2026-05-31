using LinearAlgebra
using SparseArrayKit
using Test
const _compress_sector = Telum._compress_sector
const _contract_om_axis = Telum._contract_om_axis
const _qr_shared_isometry = Telum._qr_shared_isometry
const change_dir = Telum.change_dir
⊗(a, b) = kron(b, a)



# ─── _leg_kron ────────────────────────────────────────────────────────────────
# Kronecker product of two tensors, taken leg-by-leg.
#   A: (d_1a, ..., d_QDa, Ma)
#   B: (d_1b, ..., d_QDb, Mb)
#   Result: (d_1a*d_1b, ..., d_QDa*d_QDb, Ma*Mb)
# The last axis in each operand is treated as a "bond/weight" index and is
# also Kronecker-products so the result carries the full combined bond.
function _leg_kron(A::AbstractArray, B::AbstractArray, QD::Int)
    da = ntuple(l -> size(A, l), QD)
    db = ntuple(l -> size(B, l), QD)
    Ma, Mb = size(A, QD+1), size(B, QD+1)

    # Expand A: insert QD singleton dims between physical and bond, plus a 1 at end
    #   (da..., 1..._QD, Ma, 1)
    Ar = reshape(A, da..., ones(Int, QD)..., Ma, 1)
    # Expand B: insert QD singleton dims at the front, plus a 1 before Mb
    #   (1..._QD, db..., 1, Mb)
    Br = reshape(B, ones(Int, QD)..., db..., 1, Mb)

    # Broadcast multiply → (da..., db..., Ma, Mb)
    outer = Ar .* Br

    # Permute so matching legs are adjacent: (d1a,d1b, d2a,d2b, ..., Ma,Mb)
    perm = vcat([[l, QD+l] for l in 1:QD]..., 2QD+1, 2QD+2)
    outer_p = permutedims(outer, perm)

    # Reshape interleaved pairs into single legs
    return reshape(outer_p, ntuple(l -> da[l]*db[l], QD)..., Ma*Mb)
end

# ─── _legwise_kron ───────────────────────────────────────────────────────────
# Leg-wise Kronecker product of two QD-dimensional tensors (no bond leg).
#   A: (d_1, ..., d_QD),  B: (s_1, ..., s_QD)
#   Result: (d_1*s_1, ..., d_QD*s_QD)
# Convention: kron(A_l, B_l) per leg — A (CGT) is the slow index, B (RMT) fast.
function _legwise_kron(A::AbstractArray, B::AbstractArray)
    QD = ndims(A)
    da = ntuple(l -> size(A, l), QD)
    db = ntuple(l -> size(B, l), QD)
    # Expand A → (d1,1,d2,1,...), B → (1,s1,1,s2,...)
    Ar = reshape(A, Tuple(Iterators.flatten((da[l], 1) for l in 1:QD)))
    Br = reshape(B, Tuple(Iterators.flatten((1, db[l]) for l in 1:QD)))
    outer = Ar .* Br   # shape (d1, s1, d2, s2, ..., dQD, sQD)
    # Keep pair order (dl, sl) so that dl (A) is fast in the reshape.
    # Combined index per leg = (sl-1)*dl + rl  ↔  kron(B_l, A_l) w/ A fast
    perm = Tuple(Iterators.flatten((2l-1, 2l) for l in 1:QD))
    outer_p = permutedims(outer, perm)  # (d1, s1, d2, s2, ...)
    return reshape(outer_p, ntuple(l -> da[l]*db[l], QD)...)
end

# ─── contract_sparse ─────────────────────────────────────────────────────────
# Arbitrary contraction of two arrays over matching leg pairs.
# Remaining legs keep their original relative order:
#   result legs = (free legs of A in original order, free legs of B in original order)
#
#   A     : any AbstractArray  (works with SparseArray, plain Array, …)
#   B     : any AbstractArray
#   legs_A: legs of A to contract (1-based, length M)
#   legs_B: corresponding legs of B (must match sizes pairwise)
function contract_sparse(A::AbstractArray, B::AbstractArray,
                         legs_A::NTuple{M, Int},
                         legs_B::NTuple{M, Int}) where M
    keep_A = [i for i in 1:ndims(A) if !(i in legs_A)]
    keep_B = [i for i in 1:ndims(B) if !(i in legs_B)]

    @assert all(size(A, legs_A[k]) == size(B, legs_B[k]) for k in 1:M) "contracted leg size mismatch"

    # Permute A → (keep_A..., legs_A...)  and reshape to matrix (free × contr)
    Ap = permutedims(A, [keep_A; collect(legs_A)])
    sz_fA = [size(A, i) for i in keep_A]
    Amat  = reshape(Ap, prod(sz_fA; init=1), :)

    # Permute B → (legs_B..., keep_B...)  and reshape to matrix (contr × free)
    Bp = permutedims(B, [collect(legs_B); keep_B])
    sz_fB = [size(B, i) for i in keep_B]
    Bmat  = reshape(Bp, :, prod(sz_fB; init=1))

    # Contract via matrix multiply, then restore leg shapes
    return reshape(Amat * Bmat, sz_fA..., sz_fB...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Compute offset map from a splist (space list) for a given symmetry tuple.
# Returns (leg_offset::Dict{qlabel => UnitRange}, leg_total::Int)
function get_offset(symm::Tuple, splist::Vector)
    N = length(symm)
    leg_offset = Dict{Any, UnitRange{Int}}()
    sector_size = Dict{Any, Int}()
    
    for (qlabels, RMTd) in splist
        irep_dim = prod(
            isabelian(symm[n]) ? 1 :
            dimension(getNsave_irep(symm[n], BigInt, qlabels[n]))
            for n in 1:N)
        sz = irep_dim * RMTd
        sector_size[qlabels] = sz
    end

    # Assign offsets in sorted qlabel order.
    off = 1
    for ql in sort(collect(keys(sector_size)))
        sz = sector_size[ql]
        leg_offset[ql] = off:off+sz-1
        off += sz
    end
    leg_total = off - 1
    return (leg_offset, leg_total)
end


# ─── to_sparse_array ───────────────────────────────────────────────────────────
# Convert a TLArray to a sparse array using its spaces field for offset computation.
function to_sparse_array(q::TLArray{T, QD, N, RD},
    ::Type{FT} = Float64) where {T, QD, N, RD, FT}

    symmetries = symm(q)
    # ── Step 1: offset map ──────────────────────────────────────────────────
    # Use spaces field directly for each leg's offset computation.
    leg_info = [get_offset(symmetries, q.spaces[l]) for l in 1:QD]
    leg_offsets = [li[1] for li in leg_info]
    leg_total = [li[2] for li in leg_info]

    # ── Step 2: allocate output array ───────────────────────────────────────
    result = SparseArray(zeros(FT, leg_total...))

    # ── Step 3: accumulate each sector's contribution ────────────────────────
    for sector_index in 1:Telum.nsectors(q)
        # For each symmetry n, build the CGT contracted with its w-matrix.
        # After contracting, cgt_wmats[n] has shape
        #   (d_phys_1^(n), ..., d_phys_QD^(n), M_n)
        # where d_phys_l^(n) = irrep dim at physical leg l for symmetry n,
        # and M_n = size(wmat, 2) (compressed bond dimension after SVD).
        cgt_wmats = Vector{Array{FT}}(undef, N)
        for n in 1:N
            S   = symmetries[n]
            qlabels, cgp, legdir = Telum._sector_cgt_metadata(q, sector_index, n)
            wmat = Telum.sector_wmat(q, sector_index, n)
            M   = size(wmat, 2)

            if isabelian(S)
                # Abelian symmetry: irrep dim = 1 at every leg, outer multiplicity = 1.
                # CGT is trivially the scalar 1; the contracted result is just the
                # w-matrix sector reshaped to (1,...,1, M).
                @assert M == 1 "Unexpected bond dimension for abelian symmetry $n"
                cgt_wmats[n] = reshape(FT.(wmat), ones(Int, QD)..., M)
            else
                # a. Extract CGT qlabels from this CGR.
                nin, nout = legdir
                insp  = Tuple(qlabels[i] for i in 1:nin)
                outsp = Tuple(qlabels[i] for i in nin+1:QD)
                insp_, _ = remove_zeros(S, insp)
                outsp_, _ = remove_zeros(S, outsp)

                CGTom = get_CGTom(S, insp_, outsp_)
                om    = CGTom.totalOM
                @assert om == size(wmat, 1) "outer multiplicity mismatch for symmetry $n"

                canbasis = get_canonical_basis(S, insp, outsp, CGTom)

                # b. Allocate (CGT_shape..., om).
                cgt_shape = size(Array(canbasis[1]))   # (d_in_1,...,d_out_NO) in canonical order
                cgt_arr   = zeros(FT, cgt_shape..., om)

                # c. Fill each om-slice with the corresponding canonical basis element.
                for i in 1:om cgt_arr[fill(:, QD)..., i] .= Array(canbasis[i]) end

                # d. Contract last (om) leg of cgt_arr with first (om) leg of wmat.
                #    cgt_arr: (...cgt_shape..., om)  →  flatten to (prod_shape, om)
                #    wmat:    (om, M)
                #    result:  (prod_shape, M)  →  reshaped to (cgt_shape..., M)
                wmat_data   = wmat                                 # (om, M)
                cgt_flat    = reshape(cgt_arr, :, om)              # (prod(cgt_shape), om)
                result_flat = cgt_flat * wmat_data                 # (prod(cgt_shape), M)
                cgt_wmat_canon = reshape(result_flat, cgt_shape..., M)

                # Permute from canonical leg order to physical leg order using cgp.
                # cgp[l] = canonical axis that corresponds to physical leg l,
                # so permutedims with perm = (cgp..., QD+1) maps canonical → physical.
                perm = (cgp..., QD+1)
                cgt_wmats[n] = permutedims(cgt_wmat_canon, perm)
            end
        end

        # e. Kronecker product over symmetries to get the combined CGT block.
        #    cgt_wmats[n]: (d_1^(n), ..., d_QD^(n), M_n)
        #    cgt_block:    (irep_dim_1, ..., irep_dim_QD, M_total)
        cgt_block = cgt_wmats[1]
        for n in 2:N
            cgt_block = _leg_kron(cgt_block, cgt_wmats[n], QD)
        end
        # cgt_block has shape (irep_dim_1, ..., irep_dim_QD, M_total)
        # where irep_dim_l = prod_n d_l^(n)  and  M_total = prod_n M_n.
        chi = size(cgt_block, QD + 1)   # = M_total

        # ── Step 3a: Merge all OM (non-physical) legs of RMT ─────────────────
        # RMT shape: (s_1, ..., s_QD, M_1, ..., M_N)  →  (s_1, ..., s_QD, chi)
        # Julia column-major reshape: M_1 varies fastest, consistent with how
        # _leg_kron built up the bond dimension in cgt_block.
        rmt = Array(Telum.sector_rmt(q, sector_index))
        rmt_merged = reshape(rmt, size(rmt)[1:QD]..., chi)

        # ── Step 3b: Σ_i kron(CGT[:,...,:,i], RMT[:,...,:,i]) ────────────────
        d_phys = ntuple(l -> size(cgt_block,  l), QD)   # irep dims per leg
        s_phys = ntuple(l -> size(rmt_merged, l), QD)   # RMT free dims per leg
        block  = zeros(FT, ntuple(l -> d_phys[l] * s_phys[l], QD))
        col    = fill(:, QD)   # helper for slicing all physical legs
        for i in 1:chi
            block .+= _legwise_kron(cgt_block[col..., i], rmt_merged[col..., i])
        end

        # ── Step 3c: Scatter block into result at the correct qlabel offsets ─
        ranges = Tuple(leg_offsets[l][Telum.sector_qlabel(q, sector_index, l)] for l in 1:QD)
        result[ranges...] .+= block
    end

    return result
end

# ─── test_compress_sector ────────────────────────────────────────────────────
# Invariant: the effective tensor is preserved by the SVD compression (tol=0).
#   Direct:        D[..., om3_1,...,om3_N] = Σ_p Σ_{om12} (⊗_n W[n][p]) * RMT[p][..., om12]
#   Reconstructed: contract result_RMT with each U[n] along axis QD_out+n
#
# Parameters:
#   N        : number of symmetries
#   K        : number of matched pairs (= length of new_wmats[n] and new_RMTs)
#   QD_out   : number of free (non-OM) axes in each RMT
#   free_sizes : sizes of the QD_out free axes  (default: random 2–5)
#   OM3_sizes  : output OM size per symmetry    (default: random 1–4)
#   om12_sizes : input OM size [n, p]           (default: random 1–4)
function test_compress_sector(N::Int = 2, K::Int = 2, QD_out::Int = 2;
                               seed::Int       = 420,
                               free_sizes      = rand(2:5, QD_out),
                               OM3_sizes       = rand(1:4, N),
                               om12_sizes      = rand(1:4, N, K),
                               verbose::Bool   = false)
    Random.seed!(seed)

    # W[n][p] : (OM3_sizes[n], om12_sizes[n,p])
    W    = [[randn(OM3_sizes[n], om12_sizes[n, p]) for p in 1:K] for n in 1:N]
    # RMT[p] : (free_sizes..., om12_sizes[1,p], ..., om12_sizes[N,p])
    RMTs = [randn(free_sizes..., [om12_sizes[n, p] for n in 1:N]...) for p in 1:K]

    if verbose
        println("  Parameters: N=$N, K=$K, QD_out=$QD_out")
        println("  free_sizes  = $free_sizes")
        println("  OM3_sizes   = $OM3_sizes")
        println("  om12_sizes  = $om12_sizes  (sectors=symmetry, cols=pair)")
        for n in 1:N, p in 1:K
            println("  W[$n][$p]   size = $(size(W[n][p]))")
        end
        for p in 1:K
            println("  RMTs[$p]    size = $(size(RMTs[p]))")
        end
    end

    new_wmats = Tuple([LurTensor(W[n][p]) for p in 1:K]
                      for n in 1:N)
    new_RMTs  = [LurTensor(RMTs[p]) for p in 1:K]

    U_mats, result_RMT = _compress_sector(new_wmats, new_RMTs, QD_out, 0.0)

    if verbose
        for n in 1:N
            println("  U_mats[$n]  size = $(size(U_mats[n]))")
        end
        println("  result_RMT  size = $(size(result_RMT))")
    end

    # Reconstruct: contract result_RMT with U[n] along axis QD_out+n for each n.
    reconstructed = result_RMT
    for n in 1:N
        reconstructed = _contract_om_axis(reconstructed, U_mats[n], QD_out + n)
    end
    # shape: (free_sizes..., OM3_sizes...)

    # Direct: for each pair p, contract W[n][p] along axis QD_out+n, then sum over p.
    direct = similar(reconstructed)
    fill!(direct, 0.0)
    for p in 1:K
        contrib = new_RMTs[p]
        for n in 1:N
            contrib = _contract_om_axis(contrib, new_wmats[n][p], QD_out + n)
        end
        direct .+= contrib
    end

    if verbose
        println("  reconstructed size = $(size(reconstructed))")
        println("  direct        size = $(size(direct))")
    end

    max_diff = maximum(abs, reconstructed .- direct)
    @assert max_diff < 1e-10 "test_compress_sector FAILED: max diff = $(max_diff)"
    println("test_compress_sector passed (N=$N, K=$K, QD_out=$QD_out).")
end

function test_compress_sector_zero_wmat_shortcircuits(; N::Int = 3,
                                                      K::Int = 4,
                                                      QD_out::Int = 2,
                                                      zero_symmetry::Int = 2,
                                                      seed::Int = 421)
    Random.seed!(seed)

    free_sizes = rand(2:5, QD_out)
    OM3_sizes = rand(1:4, N)
    om12_sizes = rand(1:4, N, K)

    W = [[randn(OM3_sizes[n], om12_sizes[n, p]) for p in 1:K] for n in 1:N]
    for p in 1:K
        W[zero_symmetry][p] .= 0.0
    end
    RMTs = [randn(free_sizes..., [om12_sizes[n, p] for n in 1:N]...) for p in 1:K]

    new_wmats = Tuple([LurTensor(W[n][p]) for p in 1:K]
                      for n in 1:N)
    new_RMTs  = [LurTensor(RMTs[p]) for p in 1:K]

    @test isnothing(_compress_sector(new_wmats, new_RMTs, QD_out, 0.0))
    dummy_qlabels = ntuple(_ -> (ntuple(_ -> (0,), QD_out), ntuple(identity, QD_out), (QD_out, 0)), N)
    dummy_ps = ProductSymm((ntuple(_ -> U1, N))...)
    @test isnothing(Telum.merge_new_sector(new_wmats, new_RMTs, dummy_qlabels,
                                          dummy_ps, QD_out, 0.0))

    direct = nothing
    for p in 1:K
        contrib = new_RMTs[p]
        for n in 1:N
            contrib = _contract_om_axis(contrib, new_wmats[n][p], QD_out + n)
        end
        if isnothing(direct)
            direct = similar(contrib)
            fill!(direct, 0.0)
        end
        direct .+= contrib
    end
    @test !isnothing(direct)
    @test all(iszero, direct)

    println("test_compress_sector_zero_wmat_shortcircuits passed.")
end

function test_qr_shared_isometry_rank1_fastpath()
    mats = [
        LurTensor(reshape([1.0, -2.0, 3.5], 1, :)),
        LurTensor(reshape([0.25, 4.0], 1, :)),
        LurTensor(reshape([-1.5], 1, :)),
    ]

    Q, factors = _qr_shared_isometry(mats; tol=0.0)

    @test Q isa LurTensor{Float64, 2}
    @test size(Q) == (1, 1)
    @test Q[1, 1] == 1.0
    @test length(factors) == length(mats)
    @test all(factors[i] === mats[i] for i in eachindex(mats))

    println("test_qr_shared_isometry_rank1_fastpath passed.")
end

function test_contract_xsym_wmat_tullio()
    Random.seed!(423)

    function check_case(OM1::Int, OM2::Int, OM3::Int, d1::Int, d2::Int)
        wm1 = randn(OM1, d1)
        wm2 = randn(OM2, d2)
        xarr = randn(OM1, OM2, OM3)

        result = Telum._contract_xsym_wmat(wm1, xarr, wm2)
        reference = zeros(Float64, OM3, d1, d2)
        @inbounds for a in 1:OM3, b in 1:d1, c in 1:d2
            acc = 0.0
            for cc in 1:OM2, bb in 1:OM1
                acc += xarr[bb, cc, a] * wm1[bb, b] * wm2[cc, c]
            end
            reference[a, b, c] = acc
        end

        @test result.data ≈ reference
    end

    check_case(1, 1, 1, 1, 1)
    check_case(2, 3, 4, 5, 2)
    check_case(4, 2, 3, 1, 6)

    @test_throws DimensionMismatch Telum._contract_xsym_wmat(
        randn(3, 2), randn(2, 2, 1), randn(2, 2))
    @test_throws DimensionMismatch Telum._contract_xsym_wmat(
        randn(2, 2), randn(2, 2, 1), randn(3, 2))

    println("test_contract_xsym_wmat_tullio passed.")
end

function test_accumulate_mkl_matches_small_all_orders()
    Random.seed!(424)

    function check_case(; fdim::Int, gdim::Int, cdim::Int,
                         o1dim::Int, o2dim::Int, rdim::Int)
        A = randn(fdim, cdim, o1dim)
        B = randn(gdim, cdim, o2dim)
        K = randn(rdim, o1dim, o2dim)
        initial = randn(fdim, gdim, rdim)

        mkl_results = Dict{Symbol, Array{Float64, 3}}()
        small_results = Dict{Symbol, Array{Float64, 3}}()
        for order in (:AB, :AK, :BK)
            temp_len = Telum._rmt_contract_temp_len(
                fdim, gdim, cdim, o1dim, o2dim, rdim, order)

            out_mkl = copy(initial)
            Telum._accumulate_mkl!(
                out_mkl, A, B, K, order, Vector{Float64}(undef, temp_len))

            out_small = copy(initial)
            Telum._accumulate_small!(
                out_small, A, B, K, order, Vector{Float64}(undef, temp_len))

            @test out_mkl ≈ out_small atol=1e-10 rtol=1e-10
            mkl_results[order] = out_mkl
            small_results[order] = out_small
        end

        @test mkl_results[:AB] ≈ mkl_results[:AK] atol=1e-10 rtol=1e-10
        @test mkl_results[:AB] ≈ mkl_results[:BK] atol=1e-10 rtol=1e-10
        @test small_results[:AB] ≈ small_results[:AK] atol=1e-10 rtol=1e-10
        @test small_results[:AB] ≈ small_results[:BK] atol=1e-10 rtol=1e-10
    end

    check_case(fdim=3, gdim=4, cdim=5, o1dim=2, o2dim=3, rdim=4)
    check_case(fdim=1, gdim=5, cdim=3, o1dim=4, o2dim=2, rdim=3)
    check_case(fdim=6, gdim=2, cdim=4, o1dim=3, o2dim=1, rdim=5)

    println("test_accumulate_mkl_matches_small_all_orders passed.")
end

function test_qr_shared_isometry_rank3_splits_factors()
    tensors = [
        LurTensor(reshape(collect(1.0:24.0), 3, 2, 4)),
        LurTensor(reshape(collect(25.0:42.0), 3, 3, 2)),
    ]
    mats = [LurTensor(reshape(t.data, size(t, 1), size(t, 2) * size(t, 3))) for t in tensors]

    Q3, factors3 = _qr_shared_isometry(tensors; tol=0.0)
    Q2, factors2 = _qr_shared_isometry(mats; tol=0.0)

    @test Q3.data ≈ Q2.data
    @test length(factors3) == length(tensors)
    for i in eachindex(tensors)
        @test size(factors3[i]) == (size(factors2[i], 1), size(tensors[i], 2), size(tensors[i], 3))
        @test reshape(factors3[i].data, size(factors2[i])) ≈ factors2[i].data
        @test Q3.data * reshape(factors3[i].data, size(factors2[i])) ≈ mats[i].data
    end

    println("test_qr_shared_isometry_rank3_splits_factors passed.")
end

function test_contract_compress_sector_rmt_optimizer()
    Random.seed!(422)

    function check_case(; f1::Int, f2::Int, c::Int,
                         o1::Int, o2::Int, om3::Int,
                         perm1::NTuple{3, Int}, perm2::NTuple{3, Int})
        nf1 = 1
        nf2 = 1
        cn = 1
        N = 1
        QD_out = 2

        dims1 = Vector{Int}(undef, 3)
        dims1[perm1[1]] = f1
        dims1[perm1[2]] = c
        dims1[perm1[3]] = o1

        dims2 = Vector{Int}(undef, 3)
        dims2[perm2[1]] = f2
        dims2[perm2[2]] = c
        dims2[perm2[3]] = o2

        rmt1 = LurTensor(randn(dims1...))
        rmt2 = LurTensor(randn(dims2...))
        wmat = LurTensor(randn(om3, o1, o2))

        new_wmats = (LurTensor{Float64, 3, Array{Float64, 3}}[wmat],)
        sector_pairs = [(1, 1)]
        work1 = Dict(1 => Telum._permuted_rmt_data(rmt1, perm1, nf1, cn, N))
        work2 = Dict(1 => Telum._permuted_rmt_data(rmt2, perm2, nf2, cn, N))
        kept_sizes1 = (f1,)
        kept_sizes2 = (f2,)

        U_mats, result_RMT = _compress_sector(new_wmats, sector_pairs,
                                              work1, work2,
                                              kept_sizes1, kept_sizes2, 0.0)
        reconstructed = _contract_om_axis(result_RMT, U_mats[1], QD_out + 1)
        P1 = permutedims(rmt1.data, perm1)
        P2 = permutedims(rmt2.data, perm2)
        reference = zeros(Float64, f1, f2, om3)
        @inbounds for r in 1:om3, g in 1:f2, f in 1:f1
            acc = 0.0
            for o2idx in 1:o2, o1idx in 1:o1, cidx in 1:c
                acc += P1[f, cidx, o1idx] * P2[g, cidx, o2idx] * wmat[r, o1idx, o2idx]
            end
            reference[f, g, r] = acc
        end

        @test maximum(abs, reconstructed .- reference) < 1e-10
    end

    check_case(f1=3, f2=4, c=2, o1=3, o2=2, om3=5,
               perm1=(2, 1, 3), perm2=(1, 2, 3))
    check_case(f1=12, f2=12, c=10, o1=10, o2=10, om3=8,
               perm1=(1, 2, 3), perm2=(1, 2, 3))
    check_case(f1=4, f2=3, c=5, o1=1, o2=1, om3=1,
               perm1=(3, 1, 2), perm2=(2, 3, 1))

    function check_multi_pair_reuses_temp()
        f1 = 3
        f2 = 2
        c = 4
        om3 = 5
        QD_out = 2

        work1 = Dict(
            1 => randn(f1, c, 2),
            2 => randn(f1, c, 3),
        )
        work2 = Dict(
            1 => randn(f2, c, 4),
            2 => randn(f2, c, 1),
        )
        sector_pairs = [(1, 1), (2, 1), (1, 2)]
        wmats = LurTensor{Float64, 3, Array{Float64, 3}}[
            LurTensor(randn(om3, size(work1[idx1], 3), size(work2[idx2], 3)))
            for (idx1, idx2) in sector_pairs
        ]
        new_wmats = (wmats,)

        U_mats, result_RMT = _compress_sector(new_wmats, sector_pairs,
                                              work1, work2, (f1,), (f2,), 0.0)
        reconstructed = _contract_om_axis(result_RMT, U_mats[1], QD_out + 1)

        reference = zeros(Float64, f1, f2, om3)
        @inbounds for (p, (idx1, idx2)) in pairs(sector_pairs)
            A = work1[idx1]
            B = work2[idx2]
            W = wmats[p].data
            for r in 1:om3, g in 1:f2, f in 1:f1
                acc = 0.0
                for o2idx in 1:size(B, 3), o1idx in 1:size(A, 3), cidx in 1:c
                    acc += A[f, cidx, o1idx] * B[g, cidx, o2idx] * W[r, o1idx, o2idx]
                end
                reference[f, g, r] += acc
            end
        end

        @test maximum(abs, reconstructed .- reference) < 1e-10
    end

    check_multi_pair_reuses_temp()

    println("test_contract_compress_sector_rmt_optimizer passed.")
end

function test_diag_rmt_storage_and_prepared_cache()
    r = DiagRMT([2.0, -3.0], Val(3), (1, 2))
    @test size(r) == (2, 2, 1)
    @test r[1, 1, 1] == 2.0
    @test r[2, 2, 1] == -3.0
    @test r[1, 2, 1] == 0.0
    @test Array(r) == reshape([2.0, 0.0, 0.0, -3.0], 2, 2, 1)

    rv = DiagRMT([2.0, -3.0], Val(3), (2, 0))
    @test size(rv) == (1, 4, 1)
    @test rv[1, 1, 1] == 2.0
    @test rv[1, 4, 1] == -3.0
    @test rv[1, 2, 1] == 0.0

    @test Telum._permuted_rmt_type(Array{Float64, 4}) == Array{Float64, 3}
    @test Telum._permuted_rmt_type(DiagRMT{Float64, 4}) == DiagRMT{Float64, 3}

    r4 = DiagRMT([2.0, -3.0], Val(4), (1, 2))
    mixed = Telum._permuted_rmt_data(r4, (1, 2, 3, 4), Val(1), Val(2), Val(1))
    @test mixed isa DiagRMT{Float64, 3}
    @test mixed.axis == (1, 2)
    @test size(mixed) == (2, 2, 1)

    free_vectorized = Telum._permuted_rmt_data(r4, (1, 2, 3, 4), Val(2), Val(1), Val(1))
    @test free_vectorized isa DiagRMT{Float64, 3}
    @test free_vectorized.axis == (1, 0)
    @test size(free_vectorized) == (4, 1, 1)

    contracted_vectorized = Telum._permuted_rmt_data(r4, (3, 1, 2, 4), Val(1), Val(2), Val(1))
    @test contracted_vectorized isa DiagRMT{Float64, 3}
    @test contracted_vectorized.axis == (2, 0)
    @test size(contracted_vectorized) == (1, 4, 1)

    symm = (U1,)
    qlabels = [(((0,),), ((0,),))]
    wmats = Vector{NTuple{0, Matrix{Float64}}}(undef, 1)
    RMTs = [DiagRMT([1.0, 2.0], Val(3), (1, 2))]
    spaces = ([(((0,),), 2)], [(((0,),), 2)])
    @test TLArray(symm, qlabels, wmats, RMTs,
                  (TLIndex('+'), TLIndex('-')), spaces) isa TLArray

    bad_RMTs = [DiagRMT([1.0, 2.0], Val(3), (1, 0))]
    @test_throws ArgumentError TLArray(symm, qlabels, wmats, bad_RMTs,
                                       (TLIndex('+'), TLIndex('-')), spaces)

    println("test_diag_rmt_storage_and_prepared_cache passed.")
end

function test_contract_compress_sector_diag_rmt()
    Random.seed!(423)

    function reference_contract(A, B, W)
        Ad = Array(A)
        Bd = Array(B)
        fdim, cdim, o1dim = size(Ad)
        gdim, cdim2, o2dim = size(Bd)
        @assert cdim == cdim2
        rdim = size(W, 1)
        reference = zeros(promote_type(eltype(Ad), eltype(Bd), eltype(W)), fdim, gdim, rdim)
        @inbounds for r in 1:rdim, g in 1:gdim, f in 1:fdim
            acc = 0.0
            for o2 in 1:o2dim, o1 in 1:o1dim, c in 1:cdim
                acc += Ad[f, c, o1] * Bd[g, c, o2] * W[r, o1, o2]
            end
            reference[f, g, r] = acc
        end
        return reference
    end

    function check_pair(A, B)
        W = randn(4, size(A, 3), size(B, 3))
        new_wmats = (Array{Float64, 3}[W],)
        sector_pairs = [(1, 1)]
        work1 = Dict(1 => A)
        work2 = Dict(1 => B)
        kept_sizes1 = (size(A, 1),)
        kept_sizes2 = (size(B, 1),)
        U_mats, result_RMT = _compress_sector(new_wmats, sector_pairs,
                                              work1, work2,
                                              kept_sizes1, kept_sizes2, 0.0)
        reconstructed = _contract_om_axis(result_RMT, U_mats[1], 3)
        @test reconstructed ≈ reference_contract(A, B, W) atol=1e-10 rtol=1e-10
    end

    Dleft = DiagRMT([1.5, -2.0], Val(3), (1, 2))
    Bright = randn(3, 2, 5)
    Wleft = randn(4, size(Dleft, 3), size(Bright, 3))
    @test Telum._rmt_contract_temp_len(typeof(Dleft), typeof(Bright),
                                       size(Dleft, 1), size(Bright, 1),
                                       size(Dleft, 2), size(Dleft, 3),
                                       size(Bright, 3), Wleft) == 3 * 2 * 4
    @test Telum._rmt_contract_temp_len(Dleft, Bright, Wleft) == 3 * 2 * 4

    Aleft = randn(3, 2, 5)
    Dright = DiagRMT([0.5, 4.0], Val(3), (1, 2))
    Wright = randn(4, size(Aleft, 3), size(Dright, 3))
    @test Telum._rmt_contract_temp_len(typeof(Aleft), typeof(Dright),
                                       size(Aleft, 1), size(Dright, 1),
                                       size(Aleft, 2), size(Aleft, 3),
                                       size(Dright, 3), Wright) == 0
    @test Telum._rmt_contract_temp_len(Aleft, Dright, Wright) == 0

    check_pair(DiagRMT([1.5, -2.0], Val(3), (1, 2)), randn(3, 2, 1))
    check_pair(Dleft, Bright)
    check_pair(randn(3, 2, 1), DiagRMT([0.5, 4.0], Val(3), (1, 2)))
    check_pair(Aleft, Dright)
    check_pair(DiagRMT([1.5, -2.0], Val(3), (2, 0)), randn(3, 4, 1))
    check_pair(randn(3, 4, 1), DiagRMT([0.5, 4.0], Val(3), (2, 0)))
    check_pair(DiagRMT([1.5, -2.0], Val(3), (1, 2)),
               DiagRMT([0.5, 4.0], Val(3), (1, 2)))

    println("test_contract_compress_sector_diag_rmt passed.")
end

function test_contract_diag_rmt_tlarray()
    symm = (U1,)
    qlabels = [(((0,),), ((0,),))]
    wmats = Vector{NTuple{0, Matrix{Float64}}}(undef, 1)
    spaces = ([(((0,),), 2)], [(((0,),), 2)])

    q1 = TLArray(symm, qlabels, wmats,
                 [DiagRMT([1.0, 2.0], Val(3), (1, 2))],
                 (TLIndex("x", '+'), TLIndex("c", '-')), spaces)
    q2 = TLArray(symm, qlabels, wmats,
                 [reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)],
                 (TLIndex("c", '+'), TLIndex("y", '-')), spaces)

    result = contract(q1, (2,), q2, (1,))
    @test typeof(result.RMTs) == Vector{Array{Float64, 3}}
    @test only(result.RMTs)[:, :, 1] == [1.0 3.0; 4.0 8.0]

    qlabels3 = [(((0,),), ((0,),), ((0,),))]
    spaces3 = ([(((0,),), 2)], [(((0,),), 2)], [(((0,),), 1)])
    q3 = TLArray(symm, qlabels3, wmats,
                 [DiagRMT([1.0, 2.0], Val(4), (1, 2))],
                 (TLIndex("x2", '+'), TLIndex("y2", '+'), TLIndex("c2", '-')),
                 spaces3)
    q4 = TLArray(symm, [(((0,),),)], wmats,
                 [reshape([1.0], 1, 1)],
                 (TLIndex("c2", '+'),), ([(((0,),), 1)],))

    result_free = contract(q3, (3,), q4, (1,))
    @test typeof(result_free.RMTs) == Vector{Array{Float64, 3}}
    @test only(result_free.RMTs)[:, :, 1] == [1.0 0.0; 0.0 2.0]

    println("test_contract_diag_rmt_tlarray passed.")
end

function test_diag_rmt_producers_and_metadata_ops()
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    j = get1jtensor(q.I, 1)
    @test typeof(j.RMTs) <: Vector{<:DiagRMT}

    jp = permutedims(j, (2, 1))
    @test typeof(jp.RMTs) == typeof(j.RMTs)
    @test all(Telum.sector_rmt(jp, i).axis == (1, 2) for i in Telum.sector_slots(jp) if !jp.iszero[i])

    jc = conj(j)
    @test typeof(jc.RMTs) == typeof(j.RMTs)

    js = 2.0 * j
    @test typeof(js.RMTs) == typeof(j.RMTs)
    for i in Telum.sector_slots(j)
        j.iszero[i] && continue
        @test Telum.sector_rmt(js, i).diag == 2.0 .* Telum.sector_rmt(j, i).diag
    end

    jzero = 0.0 * j
    @test all(jzero.iszero)
    @test !any(jzero.isdefined)

    ja = addSingleton(j, 3)
    @test typeof(ja.RMTs) <: Vector{<:DiagRMT}
    jd = deleteSingleton(ja, 3)
    @test typeof(jd.RMTs) == typeof(j.RMTs)

    er = eigen(q.I; hermitian=true)
    @test typeof(er.D.RMTs) <: Vector{<:DiagRMT}

    println("test_diag_rmt_producers_and_metadata_ops passed.")
end

function test_FAcont(option::LocalSpaceOptions)
    q = getLocalSpace(option);
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2));

    ct = qf * a
    Farr = to_sparse_array(q.F)
    Aarr = to_sparse_array(a)

    ctarr1 = contract_sparse(Farr, Aarr, (2,), (2,))
    ctarr2 = to_sparse_array(ct)

    println(norm(ctarr1 - ctarr2))
    @test norm(ctarr1 - ctarr2) < 1e-10

    return q, a, ct, ctarr1, ctarr2
end

function test_contract_abelian_wmats_are_unit(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2))

    ct = qf * a
    @test !isempty(ct.sectors)
    for r in ct.sectors
        @test size(r.cgrs[1].wmat.data) == (1, 1)
        @test r.cgrs[1].wmat.data ≈ [1.0;;] atol=1e-12 rtol=1e-12
    end
end

function test_getIdentity_direct_contract(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    qi = TLArray(q.I, ("lur", "lur"))
    a_pairs = getIdentity((qi, 1); itag="fused")
    a = getIdentity(qi, 1; itag="fused")

    @test a.inds == a_pairs.inds
    @test a.spaces == a_pairs.spaces
    @test _rows_equal(a.sectors, a_pairs.sectors)
    @test a.inds[1] == TLIndex(qi.inds[1].itags, '-', qi.inds[1].plev, qi.inds[1].lock, qi.inds[1].dual)
    @test a.spaces[1] == qi.spaces[1]
    @test a.inds[2] == TLIndex("fused", '-')

    ct = qi * a
    @test length(ct.inds) == 2
    @test ct.inds[1] == qi.inds[2]
    @test ct.inds[2] == TLIndex("fused", '-')
    @test ct.spaces[1] == qi.spaces[2]
    @test !isempty(ct.sectors)
end

function test_spin_local_space()
    q = getLocalSpace(SpinOptions(SU{2}, 1//2))

    @test hasproperty(q, :S)
    @test hasproperty(q, :I)
    @test length(q.I.inds) == 2
    @test length(q.S.inds) == 3

    iarr = Array(to_sparse_array(q.I))
    @test norm(iarr - Matrix{Float64}(I, 2, 2)) < 1e-10
end

function test_1jpair(option::LocalSpaceOptions)
    q = getLocalSpace(option);
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))

    q1 = get1jtensor(qi1, 2)
    q2 = permutedims(q1', (2, 1))
    mult = contract(q1, 2, q2, 1)
    arr1 = to_sparse_array(q.I)
    arr2 = to_sparse_array(mult)

    println(norm(arr1 - arr2))
    @test norm(arr1 - arr2) < 1e-10

    a = getIdentity((qi1, 2), (qi2, 2))
    q1 = get1jtensor(a, 3)
    q2 = permutedims(q1', (2, 1))
    mult = contract(q1, 2, q2, 1)

    arr1 = to_sparse_array(mult)
    sz = size(arr1)[1]
    arr2 = SparseArray(Matrix(I, sz, sz))

    println(norm(arr1 - arr2))
    @test norm(arr1 - arr2) < 1e-10

end

function test_get1jtensor_and_legflip_keywords(option::LocalSpaceOptions)
    q0 = getLocalSpace(option, ("site1", "site2", "op"))
    q = q0.F

    function test_same_qspace_structure(q1::TLArray, q2::TLArray)
        @test q1.inds == q2.inds
        @test q1.spaces == q2.spaces
        @test length(q1.sectors) == length(q2.sectors)
        for (row1, row2) in zip(q1.sectors, q2.sectors)
            @test row1.RMT.data == row2.RMT.data
            @test length(row1.cgrs) == length(row2.cgrs)
            for n in eachindex(row1.cgrs)
                @test row1.cgrs[n].qlabels == row2.cgrs[n].qlabels
                @test row1.cgrs[n].wmat.data == row2.cgrs[n].wmat.data
                @test row1.cgrs[n].cgp == row2.cgrs[n].cgp
                @test row1.cgrs[n].legdir == row2.cgrs[n].legdir
            end
        end
    end

    j_explicit = get1jtensor(q, 2)
    j_kw = get1jtensor(q; itag="site2")
    test_same_qspace_structure(j_kw, j_explicit)

    @test_throws ArgumentError get1jtensor(q; plev=0)
    @test_throws ArgumentError get1jtensor(q; itag="missing")

    flipped_explicit = legflip(q, 2)
    flipped_kw = legflip(q; itag="site2")
    test_same_qspace_structure(flipped_kw, flipped_explicit)

    @test flipped_explicit.inds[1] == q.inds[1]
    @test flipped_explicit.inds[2] == change_dir(Telum.change_green(q.inds[2]))
    @test flipped_explicit.inds[3] == q.inds[3]

    roundtrip = legflip(flipped_explicit, 2)
    @test roundtrip.inds == q.inds
    @test roundtrip.spaces == q.spaces

    flipped_multi_tuple = legflip(q, (2, 3))
    flipped_multi_vector = legflip(q, [2, 3])
    flipped_multi_kw = legflip(q; dir='-')

    test_same_qspace_structure(flipped_multi_vector, flipped_multi_tuple)
    test_same_qspace_structure(flipped_multi_kw, flipped_multi_tuple)

    @test flipped_multi_tuple.inds[1] == q.inds[1]
    @test flipped_multi_tuple.inds[2] == change_dir(Telum.change_green(q.inds[2]))
    @test flipped_multi_tuple.inds[3] == change_dir(Telum.change_green(q.inds[3]))

    roundtrip_multi = legflip(flipped_multi_tuple, (2, 3))
    @test roundtrip_multi.inds == q.inds
    @test roundtrip_multi.spaces == q.spaces

    @test_throws ArgumentError legflip(q, Int[])
    @test_throws ArgumentError legflip(q, (2, 2))
    @test_throws ArgumentError legflip(q; itag="missing")
end

function test_contract_verify_legs_checks_green(option::LocalSpaceOptions)
    q0 = getLocalSpace(option, ("site1", "site2", "op"))
    q = q0.F
    j = get1jtensor(q, 2)

    @test contract(q, (2,), j, (1,); reduce_lock=false) isa TLArray
    @test_throws AssertionError contract(q, (2,), j, (2,); reduce_lock=false)
    @test contract(q, (2,), j, (2,); reduce_lock=false, verify_legs=false) isa TLArray
end

# ─── test_conj ───────────────────────────────────────────────────────────────
# Invariant: conj(q) represented as a dense/sparse array equals the
# elementwise complex conjugate of q's sparse array.
#
# Tested TLArray objects (drawn from existing test helpers):
#   1. q.I       — 2-leg identity operator
#   2. q.F       — 3-leg creation/annihilation operator
#   3. a         — 4-leg identity from getIdentity (tensor product of two I legs)
#   4. ct        — 4-leg contraction result F ⊗ a from test_FAcont
#
# For each, we verify:
#   norm( to_sparse_array(qs)  −  conj(to_sparse_array(conj(qs))) )  < tol
#
# Physical qlabels do not change under conjugation (only stored CGR ordering
# flips), so both sparse arrays have the same shape and qlabel offsets.
# ─────────────────────────────────────────────────────────────────────────────
function test_conj(option::LocalSpaceOptions)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2))
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    # Pairs (label, TLArray) to test
    cases = [
        ("I",  q.I),
        ("F",  q.F),
        ("a",  a),
        ("ct", ct),
    ]

    zero_qlabels = copy(q.F.qlabels)
    push!(zero_qlabels, first(q.F.qlabels))
    zero_wmats = similar(q.F.wmats, length(q.F.wmats) + 1)
    zero_RMTs = similar(q.F.RMTs, length(q.F.RMTs) + 1)
    for sector_index in Telum.sector_slots(q.F)
        q.F.iszero[sector_index] && continue
        zero_wmats[sector_index] = q.F.wmats[sector_index]
        zero_RMTs[sector_index] = q.F.RMTs[sector_index]
    end
    q_with_zero_sector = TLArray(symm(q.F), zero_qlabels, zero_wmats, zero_RMTs,
                                 q.F.inds, q.F.spaces)
    qzc = conj(q_with_zero_sector)
    @test qzc.iszero[end]
    @test !isassigned(qzc.RMTs, length(qzc.RMTs))
    @test !isassigned(qzc.wmats, length(qzc.wmats))

    for (label, qs) in cases
        qc   = conj(qs)

        # Convert both to plain dense arrays for comparison.
        arr  = Array(to_sparse_array(qs))
        arrc = Array(to_sparse_array(qc))

        # conj(q) as a sparse array should equal conj(q as sparse array).
        diff = norm(arr .- conj.(arrc))
        println("test_conj [$label]: ‖arr − conj(arrc)‖ = $diff")
        @test diff < 1e-10
    end
end

# ─── test_svdQS ────────────────────────────────────────────────────────────────
# Test SVD by performing decomposition and reconstructing the original TLArray.
#
# Arguments:
#   q         : TLArray to decompose
#   left_legs : legs to group on the "U" side (same as svd)
#   cutoff    : singular value cutoff (default 1e-12)
#   tol       : tolerance for reconstruction error (default 1e-9)
#
# Returns: (diff_norm, max_rmt_diff) where
#   diff_norm     : norm of (original - reconstructed) as sparse array
#   max_rmt_diff  : maximum per-RMT norm of difference (after matching sectors)
#
# Algorithm:
#   1. Perform svd(q, left_legs; cutoff)
#   2. Contract U * S to get US
#   3. Contract US * Vd to get reconstructed TLArray
#   4. Permute reconstructed to match original leg order
#   5. Convert both to sparse arrays and compute norm of difference
# ─────────────────────────────────────────────────────────────────────────────
function test_svdQS(q::TLArray{T, QD, N, RD},
                    left_legs;
                    cutoff::Float64 = 1e-12,
                    tol::Float64 = 1e-9,
                    verbose::Bool = true) where {T, QD, N, RD}
    
    left_legs = collect(Int, left_legs)
    right_legs = [l for l in 1:QD if l ∉ left_legs]
    NL, NR = length(left_legs), length(right_legs)
    
    # Step 1: Perform SVD
    result = svd(q, left_legs; cutoff=cutoff)
    U, S, Vd = result.U, result.S, result.Vd
    
    if verbose
        println("SVD completed:")
        println("  U  : $(length(U.sectors)) sectors, $(NL+1) legs")
        println("  S  : $(length(S.sectors)) sectors, 2 legs")
        println("  Vd : $(length(Vd.sectors)) sectors, $(NR+1) legs")
    end
    
    # Step 2: Contract U * S
    # U has legs (left_legs..., bond_left'+'), bond leg is last with direction '+'
    # S has legs (left_tag'-', right_tag'-'), both directions '-'
    # U's bond leg (+) should contract with S's first leg (-)
    US = U * S
    
    if verbose
        println("  US : $(length(US.sectors)) sectors after U*S")
    end
    
    # Step 3: Contract US * Vd
    # US has legs (left_legs..., right_tag'-')
    # Vd has legs (bond'+', right_legs...)
    # US's last leg (-) should contract with Vd's first leg (+)
    rec = US * Vd
    
    if verbose
        println("  rec: $(length(rec.sectors)) sectors after (U*S)*Vd")
    end
    
    # Step 4: Permute reconstructed to match original leg order
    # After contraction, rec has legs in order (left_legs..., right_legs...)
    # We need to permute back to original order (1, 2, ..., QD)
    # Build inverse permutation: orig_order[i] tells where leg i should go
    combined_order = vcat(left_legs, right_legs)
    inv_perm = zeros(Int, QD)
    for (new_pos, orig_leg) in enumerate(combined_order)
        inv_perm[orig_leg] = new_pos
    end
    # Now inv_perm[orig_leg] = current_pos, so we need perm where perm[final_pos] = current_pos
    # final_pos = orig_leg, so perm[orig_leg] = inv_perm[orig_leg] is wrong
    # Actually: combined_order[new_pos] = orig_leg means leg orig_leg is at new_pos
    # We want permutation p such that after permute, leg in position i came from combined_order[p[i]]
    # i.e., we want result[i] = orig[p[i]] where orig has legs in combined_order
    # Since combined_order[new_pos] = orig_leg, we want p[orig_leg] = new_pos
    # That's what inv_perm computes: inv_perm[orig_leg] = new_pos
    # So we apply permutation inv_perm to get legs in order (1, 2, ..., QD)
    
    # Wait, let me think again:
    # rec has legs ordered as combined_order = (left_legs..., right_legs...)
    # rec.legs[new_pos] corresponds to original leg combined_order[new_pos]
    # We want final.legs[orig_leg] = rec.legs[pos_in_rec_of_orig_leg]
    # pos_in_rec_of_orig_leg = inv_perm[orig_leg] where inv_perm[combined_order[i]] = i
    # So perm for permutedims: perm[final_pos] = old_pos means final.leg[final_pos] = rec.leg[old_pos]
    # We want final.leg[orig_leg] = rec.leg[inv_perm[orig_leg]]
    # So perm[orig_leg] = inv_perm[orig_leg]? No, that's circular.
    # 
    # Let's be concrete: if left_legs = [1,3], right_legs = [2,4]
    # combined_order = [1,3,2,4]
    # rec has legs: (orig_leg1, orig_leg3, orig_leg2, orig_leg4)
    # We want: (orig_leg1, orig_leg2, orig_leg3, orig_leg4)
    # So perm = [1, 3, 2, 4] meaning final[1]=rec[1], final[2]=rec[3], final[3]=rec[2], final[4]=rec[4]
    # In general: perm[orig_leg] = position_of_orig_leg_in_combined_order
    # inv_perm[combined_order[i]] = i, so inv_perm[orig_leg] = position_of_orig_leg_in_combined_order
    # So perm = inv_perm, and perm[orig_leg] = inv_perm[orig_leg]
    
    # Actually, permutedims semantics: perm[new_pos] = old_pos
    # result.leg[new_pos] = input.leg[perm[new_pos]]
    # We want result.leg[orig_leg] = rec.leg[inv_perm[orig_leg]]
    # So perm[orig_leg] = inv_perm[orig_leg], i.e., perm = inv_perm
    
    perm = Tuple(inv_perm)
    rec_permuted = permutedims(rec, perm)
    
    if verbose
        println("  rec_permuted: legs permuted to original order")
    end
    
    # Step 5: Convert to sparse arrays and compare
    arr_orig = Array(to_sparse_array(q))
    arr_rec  = Array(to_sparse_array(rec_permuted))
    
    diff_arr = arr_orig .- arr_rec
    diff_norm = norm(diff_arr)
    max_diff = maximum(abs, diff_arr)
    
    if verbose
        println("Reconstruction error:")
        println("  ‖orig - rec‖₂ = $diff_norm")
        println("  max|orig - rec| = $max_diff")
    end
    
    @assert diff_norm < tol "SVD reconstruction error ‖orig - rec‖ = $diff_norm exceeds tolerance $tol"
    
    return diff_norm, max_diff
end

# ─── test_norm ───────────────────────────────────────────────────────────────
# Verify norm(q) against two independent reference values:
#
#   (a) norm(to_sparse_array(q))  – dense/sparse Frobenius norm
#   (b) sqrt(abs((q * q')[]))    – scalar obtained by contracting q with conj(q)
#                                   over all legs (Wigner-Eckart inner product)
#
# Cases tested per call:
#   q.I  – rank-2 identity (QD = 2, exercises the dim·‖RMT‖² formula)
#   q.F  – rank-3 operator (QD = 3, standard CGT-orthonormal formula)
#   ct   – rank-4 tensor   (QD = 4, after F * identity contraction)
#
# For (b) to work every pair (q leg l, conj(q) leg l) must form a valid
# contraction pair: same itags, opposite directions.  conj() flips all
# directions, so the pairing is always leg-for-leg given that all legs of q
# already carry distinct TLIndex values (full-rank contraction → 0D scalar).
# ─────────────────────────────────────────────────────────────────────────────
function test_norm(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a   # rank-4: legs (lur1_in, lur2_in, lur2_out, op)

    cases = [
        ("I",  q.I),
        ("F",  q.F),
        ("ct", ct),
    ]

    for (label, qs) in cases
        # ── (a) compare with sparse array norm ─────────────────────────────
        norm_qs     = norm(qs)
        norm_sparse = norm(Array(to_sparse_array(qs)))
        diff_a = abs(norm_qs - norm_sparse)
        println("test_norm [$label]: norm_qs=$norm_qs  norm_sparse=$norm_sparse  Δ=$diff_a")
        @test diff_a < tol

        # ── (b) compare norm² with scalar from qs * qs' ────────────────────
        #   qs' = conj(qs) flips all leg directions; contraction over all
        #   matching legs yields a 0D TLArray whose single element is ‖qs‖².
        scalar_qs     = qs * qs'
        println(qs)
        norm_sq_contr = abs(scalar_qs[])
        diff_b = abs(norm_qs^2 - norm_sq_contr)
        ref    = max(norm_sq_contr, 1.0)
        println("test_norm [$label]: ‖qs‖²=$(norm_qs^2)  (qs·qs')=$norm_sq_contr  Δ=$diff_b  (rel=$(diff_b/ref))")
        @test diff_b < tol * ref
    end
end

# ─── test_eigen ──────────────────────────────────────────────────────────────
# Build a non-trivial Hermitian test input by copying the structure of q.I and
# replacing each block's RMT with a random real symmetric matrix of the same
# size.  This avoids testing only the trivial identity (eigenvalues all 1) and
# exercises the decomposition with general positive-definite blocks.
#
# Checks:
#   (a) Reconstruction  : ‖V * D * V' - A‖ / ‖A‖ < tol  (relative)
#   (b) Orthonormality  : ‖V' * V - I‖ < tol
#   (c) eig_list size   : total count == sum of RMT sector block sizes
#   (d) eig_list order  : sorted ascending by eigenvalue
#   (e) eig_list space  : each entry carries its sector and in-sector index
# ─────────────────────────────────────────────────────────────────────────────
function test_eigen(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    # Copy q.I (keeps CGRs/inds/spaces intact) then overwrite each RMT block
    # with a random real-symmetric (= Hermitian) matrix of the same block size.
    # Using A = Mᵀ * M + εI guarantees positive-definite eigenvalues so the
    # sort-descending check is unambiguous.
    A = copy(q.I)
    rng = Random.MersenneTwister(42)
    for r in A.sectors
        sz = size(r.RMT.data)   # (n, n, 1, …, 1)
        n  = sz[1]
        M  = randn(rng, n, n)
        H  = M' * M + I(n) * 0.1   # symmetric, positive-definite
        r.RMT.data .= reshape(Float64.(H), sz)
    end

    result = eigen(A; hermitian = true)
    @test isnothing(result.V_inv)
    D = result.D
    V = result.V
    eig_list = result.eig_list
    println("test_eigen: $(length(eig_list)) eigenvalues, $(length(D.sectors)) D-sectors, $(length(V.sectors)) V-sectors")

    # ── (a) Reconstruction: V * D * V' ≈ A ──────────────────────────────────
    rec      = lock(V, 1) * (D * V')
    arr_A    = Array(to_sparse_array(A))
    arr_rec  = Array(to_sparse_array(rec))
    ref_norm = max(norm(arr_A), 1.0)
    diff_a   = norm(arr_A - arr_rec) / ref_norm
    println("  ‖V*D*V' - A‖/‖A‖ = $diff_a")
    @test diff_a < tol

    # ── (b) Orthonormality: V' * V ≈ I ──────────────────────────────────────
    VtV     = V' * lock(V, 2)
    arr_VtV = Array(to_sparse_array(VtV))
    n_bond  = size(arr_VtV, 1)
    diff_b  = norm(arr_VtV - Matrix(I, n_bond, n_bond))
    println("  ‖V'V - I‖ = $diff_b")
    @test diff_b < tol

    # ── (c) eig_list size matches total RMT dimension ────────────────────────
    total_eig_count = sum(size(r.RMT.data, 1) for r in A.sectors)
    @test length(eig_list) == total_eig_count

    # ── (d) eig_list is sorted ascending ─────────────────────────────────────
    if length(eig_list) > 1
        @test all(real(eig_list[i][1]) <= real(eig_list[i+1][1]) for i in 1:length(eig_list)-1)
    end

    # ── (e) eig_list entries include sector metadata ────────────────────────
    sector_dims = Dict(
        Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(symm(A))) => size(r.RMT.data, 1)
        for r in D.sectors
    )
    for entry in eig_list
        _, deg, sector, idx = entry
        @test deg >= 1
        @test idx >= 1
        if haskey(sector_dims, sector)
            @test idx <= sector_dims[sector]
        end
    end
end

function test_eigen_autodetect(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    A = copy(q.I)
    rng = Random.MersenneTwister(7)
    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n)
        H  = M' * M + I(n) * 0.1
        r.RMT.data .= reshape(Float64.(H), sz)
    end

    hermitian_result = eigen(A)
    @test isnothing(hermitian_result.V_inv)
    hermitian_rec = lock(hermitian_result.V, 1) * (hermitian_result.D * hermitian_result.V')
    arr_A = Array(to_sparse_array(A))
    arr_hermitian_rec = Array(to_sparse_array(hermitian_rec))
    hermitian_diff = norm(arr_A - arr_hermitian_rec) / max(norm(arr_A), 1.0)
    @test hermitian_diff < tol

    B = copy(q.I)
    rng = Random.MersenneTwister(8)
    made_nonsymmetric = false
    for r in B.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n)
        if n > 1
            M[1, 2] += 1.0
            M[2, 1] -= 0.5
            made_nonsymmetric = true
        end
        r.RMT.data .= reshape(Float64.(M + I(n) * 0.1), sz)
    end
    made_nonsymmetric || return

    general_result = eigen(B)
    @test !isnothing(general_result.V_inv)
    general_rec = lock(general_result.V, 1) * (general_result.D * general_result.V_inv)
    arr_B = Array(to_sparse_array(B))
    arr_general_rec = Array(to_sparse_array(general_rec))
    general_diff = norm(arr_B - arr_general_rec) / max(norm(arr_B), 1.0)
    @test general_diff < tol
end

function test_eigen_permuted_input(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    A = copy(q.I)
    rng = Random.MersenneTwister(11)
    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n)
        H  = M' * M + I(n) * 0.1
        r.RMT.data .= reshape(Float64.(H), sz)
    end

    A_perm = permutedims(A, (2, 1))
    result = eigen(A_perm; hermitian = true)

    @test (result.D.inds[1].dir, result.D.inds[2].dir) == ('+', '-')
    @test (result.V.inds[1].dir, result.V.inds[2].dir) == ('+', '-')
    @test isnothing(result.V_inv)

    rec = lock(result.V, 1) * (result.D * result.V')
    arr_A = Array(to_sparse_array(A))
    arr_rec = Array(to_sparse_array(rec))
    diff = norm(arr_A - arr_rec) / max(norm(arr_A), 1.0)
    @test diff < tol
end

function test_eigen_hermitian_leg_guard(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    A = copy(q.I)
    rng = Random.MersenneTwister(13)
    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n)
        H  = M' * M + I(n) * 0.1
        r.RMT.data .= reshape(Float64.(H), sz)
    end

    idx1, idx2 = A.inds
    bad_inds = (
        TLIndex("eigL", idx1.dir, idx1.plev, idx1.lock, idx1.dual),
        TLIndex("eigR", idx2.dir, idx2.plev, idx2.lock, idx2.dual),
    )
    A_bad = TLArray(A, bad_inds)

    @test_throws AssertionError eigen(A_bad; hermitian = true)
    @test_throws AssertionError eigen(A_bad)
end

# ─── test_spaces_eigen ───────────────────────────────────────────────────────
# Verify that spaces of D and V returned by eigen are consistent:
#
#   (a) Input assertion: q.I has equal spaces on both legs
#   (b) D.spaces[1] == D.spaces[2]  (same bond sector list on both sides)
#   (c) V.spaces[1] == V.spaces[2]
#   (d) D and V share the same bond space: D.spaces[1] == V.spaces[2]
#   (e) Bond qlabels of D (and V) are a subset of input qlabels
# ─────────────────────────────────────────────────────────────────────────────
function test_spaces_eigen(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)
    rng = Random.MersenneTwister(0)
    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n)
        r.RMT.data .= reshape(Float64.(M' * M + I(n) * 0.1), sz)
    end

    # (a) Input spaces are equal on both legs
    @test q.I.spaces[1] == q.I.spaces[2]

    result = eigen(A; hermitian = true)
    D = result.D
    V = result.V

    println("test_spaces_eigen: D.spaces=$(length(D.spaces[1])) sectors, V.spaces=$(length(V.spaces[1])) sectors")

    # (b) D legs share the same space
    @test D.spaces[1] == D.spaces[2]

    # (c) V legs share the same space
    @test V.spaces[1] == V.spaces[2]

    # (d) D and V share the same bond space
    @test D.spaces[1] == V.spaces[2]

    # (e) Bond qlabels ⊆ input qlabels
    input_qls = Set(ql for (ql, _) in q.I.spaces[1])
    bond_qls  = Set(ql for (ql, _) in D.spaces[1])
    @test issubset(bond_qls, input_qls)
end

# ─── test_missing_spaces_eigen ───────────────────────────────────────────────
# Verify that sectors present only in the space list are treated as zero blocks:
# zero eigenvalues appear in eig_list and identity sectors are inserted in V.
# ─────────────────────────────────────────────────────────────────────────────
function test_missing_spaces_eigen(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    @test !isempty(q.I.sectors)

    removed_row = q.I.sectors[1]
    removed_sector = Tuple(removed_row.cgrs[n].qlabels[removed_row.cgrs[n].cgp[1]] for n in 1:length(symm(q.I)))
    removed_dim = size(removed_row.RMT.data, 1)

    kept_rows = copy(q.I.sectors[2:end])
    A = TLArray(symm(q.I), kept_rows, q.I.inds, q.I.spaces)

    result = eigen(A; hermitian = true)
    D = result.D
    V = result.V
    expected_cgp = (D.inds[1].dir, D.inds[2].dir) == ('+', '-') ? (1, 2) : (2, 1)

    zero_entries = [entry for entry in result.eig_list if entry[3] == removed_sector]
    @test length(zero_entries) == removed_dim
    @test all(iszero(entry[1]) for entry in zero_entries)

    v_row = only([r for r in V.sectors if Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(symm(V))) == removed_sector])
    v_mat = reshape(v_row.RMT.data, size(v_row.RMT.data, 1), size(v_row.RMT.data, 2))
    @test v_mat ≈ Matrix(I, removed_dim, removed_dim)
    @test all(cgr.cgp == expected_cgp for cgr in v_row.cgrs)

    d_rows = [r for r in D.sectors if Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(symm(D))) == removed_sector]
    for d_row in d_rows
        @test all(cgr.cgp == expected_cgp for cgr in d_row.cgrs)
        d_mat = reshape(d_row.RMT.data, size(d_row.RMT.data, 1), size(d_row.RMT.data, 2))
        @test iszero(d_mat)
    end

    @test any(ql == removed_sector for (ql, _) in D.spaces[1])

    VtV = V' * lock(V, 2)
    arr_VtV = Array(to_sparse_array(VtV))
    @test norm(arr_VtV - Matrix(I, size(arr_VtV, 1), size(arr_VtV, 2))) < tol
end

function test_missing_spaces_eigen_zero_diagonal()
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("lur", "lur", "op"))
    result = eigen(q.I + q.Z; hermitian = true)

    vals = Float64[]
    for sector_index in 1:result.D.nsectors
        rmt = result.D.RMTs[sector_index].data
        mat = reshape(rmt, size(rmt, 1), size(rmt, 2))
        append!(vals, diag(mat))
    end
    sort!(vals)
    @test vals == [2.0, 2.0]
    @test 1.0 ∉ vals
    @test sort(first.(result.eig_list)) == [0.0, 2.0, 2.0]
end

# ─── test_truncate_missing_zero_spaces_eigen ────────────────────────────────
# Verify that truncation preserves sectors selected only through zero
# eigenvalues, even when the corresponding D sectors are absent.
# ─────────────────────────────────────────────────────────────────────────────
function test_truncate_missing_zero_spaces_eigen(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    @test !isempty(q.I.sectors)

    removed_row = q.I.sectors[1]
    removed_sector = Tuple(removed_row.cgrs[n].qlabels[removed_row.cgrs[n].cgp[1]] for n in 1:length(symm(q.I)))
    removed_dim = size(removed_row.RMT.data, 1)

    kept_rows = copy(q.I.sectors[2:end])
    A = TLArray(symm(q.I), kept_rows, q.I.inds, q.I.spaces)

    result = eigen(A; hermitian = true)
    kept, discarded = discard_eigen(result, removed_dim, 0.0, "eigK", "eigD"; hermitian = true)
    expected_cgp = (result.D.inds[1].dir, result.D.inds[2].dir) == ('+', '-') ? (1, 2) : (2, 1)
    eig_tag = result.D.inds[1].itags
    v_orig_leg = only(findall(i -> result.V.inds[i].itags != eig_tag, 1:2))
    v_eig_leg = only(findall(i -> result.V.inds[i].itags == eig_tag, 1:2))

    @test length(kept.eig_list) == removed_dim
    @test all(entry[3] == removed_sector for entry in kept.eig_list)
    @test all(iszero(entry[1]) for entry in kept.eig_list)

    @test isempty(kept.D.sectors)
    @test kept.D.spaces[1] == [(removed_sector, removed_dim)]
    @test kept.D.spaces[2] == [(removed_sector, removed_dim)]

    @test kept.V.spaces[v_orig_leg] == result.V.spaces[v_orig_leg]
    @test discarded.V.spaces[v_orig_leg] == result.V.spaces[v_orig_leg]
    @test kept.V.spaces[v_eig_leg] == [(removed_sector, removed_dim)]
    @test all(ql != removed_sector for (ql, _) in discarded.V.spaces[v_eig_leg])
    kept_v_rows = [r for r in kept.V.sectors if Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(symm(kept.V))) == removed_sector]
    for r in kept_v_rows
        @test all(cgr.cgp == expected_cgp for cgr in r.cgrs)
    end

    @test any(ql == removed_sector for (ql, _) in kept.V.spaces[1])
    @test any(ql == removed_sector for (ql, _) in kept.V.spaces[2])
    @test all(ql != removed_sector for (ql, _) in discarded.D.spaces[1])

    result_full = Telum._eigen_general(A)
    kept_full, discarded_full = discard_eigen(result_full, removed_dim, 0.0, "eigKf", "eigDf"; hermitian = false)
    eig_tag_full = result_full.D.inds[1].itags
    v_full_orig_leg = only(findall(i -> result_full.V.inds[i].itags != eig_tag_full, 1:2))
    v_full_eig_leg = only(findall(i -> result_full.V.inds[i].itags == eig_tag_full, 1:2))
    vinv_orig_leg = only(findall(i -> result_full.V_inv.inds[i].itags != eig_tag_full, 1:2))
    vinv_eig_leg = only(findall(i -> result_full.V_inv.inds[i].itags == eig_tag_full, 1:2))

    @test kept_full.V.spaces[v_full_orig_leg] == result_full.V.spaces[v_full_orig_leg]
    @test discarded_full.V.spaces[v_full_orig_leg] == result_full.V.spaces[v_full_orig_leg]
    @test kept_full.V.spaces[v_full_eig_leg] == [(removed_sector, removed_dim)]
    @test all(ql != removed_sector for (ql, _) in discarded_full.V.spaces[v_full_eig_leg])

    @test kept_full.V_inv.spaces[vinv_orig_leg] == result_full.V_inv.spaces[vinv_orig_leg]
    @test discarded_full.V_inv.spaces[vinv_orig_leg] == result_full.V_inv.spaces[vinv_orig_leg]
    @test kept_full.V_inv.spaces[vinv_eig_leg] == [(removed_sector, removed_dim)]
    @test all(ql != removed_sector for (ql, _) in discarded_full.V_inv.spaces[vinv_eig_leg])
end

# ─── test_discard_eigen ─────────────────────────────────────────────────────
# Verify that eigenvalue truncation keeps the Nkeep smallest eigenvalues and
# splits the eigendecomposition consistently into kept and discarded parts.
# ─────────────────────────────────────────────────────────────────────────────
function test_discard_eigen(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)

    offset = 0.0
    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        vals = collect(offset .+ (1.0:n))
        offset += n + 1.0
        r.RMT.data .= reshape(Matrix(Diagonal(vals)), sz)
    end

    result = eigen(A; hermitian = true)
    D = result.D
    V = result.V
    eig_list = result.eig_list
    Nkeep = min(3, length(eig_list))
    kept, discarded = discard_eigen(result, Nkeep, 0.0, "eigK", "eigD"; hermitian = true)
    Vkeep = kept.V
    Dkeep = kept.D
    eig_keep = kept.eig_list
    Vdiscard = discarded.V
    Ddiscard = discarded.D
    eig_discard = discarded.eig_list

    sorted_asc = sort(copy(eig_list); by = x -> real(x[1]))
    sorted_keep = sorted_asc[1:Nkeep]
    sorted_discard = sorted_asc[Nkeep+1:end]
    @test [(x[1], x[2], x[3]) for x in eig_keep] == [(x[1], x[2], x[3]) for x in sorted_keep]
    @test [(x[1], x[2], x[3]) for x in eig_discard] == [(x[1], x[2], x[3]) for x in sorted_discard]

    @test length(eig_keep) == Nkeep
    @test length(eig_keep) + length(eig_discard) == length(eig_list)

    for (entries, sorted_entries) in ((eig_keep, sorted_keep), (eig_discard, sorted_discard))
        isempty(entries) && continue
        sector_maps = Dict{typeof(entries[1][3]), Dict{Int, Int}}()
        sector_indices = Dict{typeof(entries[1][3]), Vector{Int}}()
        for entry in sorted_entries
            push!(get!(sector_indices, entry[3], Int[]), entry[4])
        end
        for (sector, idxs) in sector_indices
            sector_maps[sector] = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(sort(unique(idxs))))
        end
        for (entry, entry_orig) in zip(entries, sorted_entries)
            @test entry[4] == sector_maps[entry[3]][entry_orig[4]]
        end
    end

    keep_vals = isempty(Dkeep.sectors) ? eltype(D.sectors[1].RMT.data)[] :
        sort(vcat([diag(reshape(r.RMT.data, size(r.RMT.data, 1), size(r.RMT.data, 2))) for r in Dkeep.sectors]...))
    disc_vals = isempty(Ddiscard.sectors) ? eltype(D.sectors[1].RMT.data)[] :
        sort(vcat([diag(reshape(r.RMT.data, size(r.RMT.data, 1), size(r.RMT.data, 2))) for r in Ddiscard.sectors]...))

    @test keep_vals ≈ sort([x[1] for x in eig_keep])
    @test disc_vals ≈ sort([x[1] for x in eig_discard])

    n_keep_cols = sum((size(r.RMT.data, 2) for r in Vkeep.sectors); init = 0)
    n_disc_cols = sum((size(r.RMT.data, 2) for r in Vdiscard.sectors); init = 0)
    @test n_keep_cols == length(eig_keep)
    @test n_disc_cols == length(eig_discard)

    keep_qls = Set(ql for (ql, _) in Dkeep.spaces[1])
    discard_qls = Set(ql for (ql, _) in Ddiscard.spaces[1])
    @test issubset(keep_qls, Set(ql for (ql, _) in D.spaces[1]))
    @test issubset(discard_qls, Set(ql for (ql, _) in D.spaces[1]))

    if !isempty(Dkeep.sectors)
        rec_keep = lock(Vkeep, 1) * (Dkeep * Vkeep')
        arr_keep = Array(to_sparse_array(rec_keep))
        @test isfinite(norm(arr_keep))
    end

    if !isempty(Ddiscard.sectors)
        rec_discard = lock(Vdiscard, 1) * (Ddiscard * Vdiscard')
        arr_discard = Array(to_sparse_array(rec_discard))
        @test isfinite(norm(arr_discard))
    end

    rec_total = nothing
    if !isempty(Dkeep.sectors)
        rec_keep = lock(Vkeep, 1) * (Dkeep * Vkeep')
        rec_total = isnothing(rec_total) ? rec_keep : rec_total + rec_keep
    end
    if !isempty(Ddiscard.sectors)
        rec_discard = lock(Vdiscard, 1) * (Ddiscard * Vdiscard')
        rec_total = isnothing(rec_total) ? rec_discard : rec_total + rec_discard
    end

    @test !isnothing(rec_total)
    arr_total = Array(to_sparse_array(rec_total))
    arr_A = Array(to_sparse_array(A))
    @test norm(arr_A - arr_total) / max(norm(arr_A), 1.0) < tol
end

function test_discard_eigen_tol(option::LocalSpaceOptions)
    entries = [(v, 1, (), i) for (i, v) in enumerate([1.0, 2.0, 3.0, 3.1, 10.0])]
    @test Telum._effective_eigen_keep_count(entries, 3, 0.1; hermitian=true) == 4
    @test Telum._effective_eigen_keep_count(entries[1:3], 5, 0.1; hermitian=true) == 3

    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)

    dims = [size(r.RMT.data, 1) for r in A.sectors]
    total_dim = sum(dims)
    @assert total_dim >= 3

    vals_all = [1.0, 10.0, 10.1]
    append!(vals_all, (30.0 + i for i in 0:total_dim-length(vals_all)-1))

    offset = 1
    for r in A.sectors
        n = size(r.RMT.data, 1)
        vals = vals_all[offset:offset+n-1]
        r.RMT.data .= reshape(Matrix(Diagonal(vals)), size(r.RMT.data))
        offset += n
    end

    result = eigen(A; hermitian = true)
    kept_exact, discarded_exact = discard_eigen(result, 2, 0.0, "eigK0", "eigD0"; hermitian = true)
    kept_tol, discarded_tol = discard_eigen(result, 2, 0.5, "eigKT", "eigDT"; hermitian = true)

    @test length(kept_exact.eig_list) == 2
    @test length(discarded_exact.eig_list) + length(kept_exact.eig_list) == length(result.eig_list)

    @test length(kept_tol.eig_list) == 3
    @test length(discarded_tol.eig_list) + length(kept_tol.eig_list) == length(result.eig_list)

    @test [x[1] for x in kept_exact.eig_list] ≈ [1.0, 10.0]
    @test [x[1] for x in kept_tol.eig_list] ≈ [1.0, 10.0, 10.1]
end

# ─── test_eigen_general_discard ─────────────────────────────────────────────────
# Verify that the result struct and discard path also work for the full
# non-Hermitian eigendecomposition, including V_inv slicing.
# ─────────────────────────────────────────────────────────────────────────────
function test_eigen_general_discard(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)
    rng = Random.MersenneTwister(7)

    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n) + 0.2 * Matrix(I, n, n)
        r.RMT.data .= reshape(ComplexF64.(M), sz)
    end

    result = Telum._eigen_general(A)
    @test !isnothing(result.V_inv)

    Nkeep = min(2, length(result.eig_list))
    kept, discarded = discard_eigen(result, Nkeep, "eigK", "eigD"; hermitian = false)

    @test !isnothing(kept.V_inv)
    @test !isnothing(discarded.V_inv)
    expected_keep = Telum._effective_eigen_keep_count(result.eig_list, Nkeep, 0.1; hermitian = false)
    @test length(kept.eig_list) == expected_keep
    @test length(kept.eig_list) + length(discarded.eig_list) == length(result.eig_list)

    n_keep_v_cols = sum((size(r.RMT.data, 2) for r in kept.V.sectors); init = 0)
    n_keep_vinv_rows = isnothing(kept.V_inv) ? 0 : sum((size(r.RMT.data, 1) for r in kept.V_inv.sectors); init = 0)
    @test n_keep_v_cols == length(kept.eig_list)
    @test n_keep_vinv_rows == length(kept.eig_list)

    n_disc_v_cols = sum((size(r.RMT.data, 2) for r in discarded.V.sectors); init = 0)
    n_disc_vinv_rows = isnothing(discarded.V_inv) ? 0 : sum((size(r.RMT.data, 1) for r in discarded.V_inv.sectors); init = 0)
    @test n_disc_v_cols == length(discarded.eig_list)
    @test n_disc_vinv_rows == length(discarded.eig_list)
end

# ─── test_discard_eigen_itag ───────────────────────────────────────────────────
# Verify that discard_eigen retags the bond legs of D, V, and V_inv for kept
# and discarded eigenspaces independently.
# ─────────────────────────────────────────────────────────────────────────────
function test_discard_eigen_itag(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)
    rng = Random.MersenneTwister(17)

    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n) + 0.3 * Matrix(I, n, n)
        r.RMT.data .= reshape(ComplexF64.(M), sz)
    end

    result = Telum._eigen_general(A, "origEig")
    kept_tag = "keptEig"
    discarded_tag = "discardedEig"
    kept, discarded = discard_eigen(
        result,
        min(2, length(result.eig_list)),
        kept_tag,
        discarded_tag;
        hermitian = false,
    )

    @test kept.D.inds[1].itags == kept_tag
    @test kept.D.inds[2].itags == kept_tag
    @test kept.V.inds[2].itags == kept_tag
    @test !isnothing(kept.V_inv)
    @test kept.V_inv.inds[1].itags == kept_tag

    @test discarded.D.inds[1].itags == discarded_tag
    @test discarded.D.inds[2].itags == discarded_tag
    @test discarded.V.inds[2].itags == discarded_tag
    @test !isnothing(discarded.V_inv)
    @test discarded.V_inv.inds[1].itags == discarded_tag

    @test kept.V.inds[1] == result.V.inds[1]
    @test kept.V_inv.inds[2] == result.V_inv.inds[2]
    @test discarded.V.inds[1] == result.V.inds[1]
    @test discarded.V_inv.inds[2] == result.V_inv.inds[2]
end

# ─── test_spaces_svdQS ───────────────────────────────────────────────────────
# Verify that spaces of U, S, Vd returned by svd are consistent.
#
#   No-truncation case:
#   (a) U bond == S left
#   (b) S.spaces[2] is the exact dual splist of S.spaces[1]
#   (c) Vd bond == S right
#   (d) U non-bond leg spaces == corresponding input qf spaces
#   (e) Vd non-bond leg space == corresponding input qf space
#
#   Truncation case (Nkeep=1):
#   (f) Ut bond == St left
#   (g) St.spaces[2] is the exact dual splist of St.spaces[1]
#   (h) Vdt bond == St right
#   (i) St left ⊆ Ut bond  (S reduced; U/Vd bond inherits full pre-trunc space)
#   (j) Ut / Vdt non-bond leg spaces still match input qf spaces
# ─────────────────────────────────────────────────────────────────────────────
function test_spaces_svdQS(option::LocalSpaceOptions)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a   # rank-4: legs (lur1_in, lur2_in, lur2_out, op)

    # qlabel set from a splist
    qls(sp) = Set(ql for (ql, _) in sp)
    symmetries = symm(q.I); N = length(symmetries)

    # Compute the dual of a splist: apply get_dualq per-symmetry to every qlabel tuple.
    function dual_sp(sp)
        ET = eltype(sp)
        sort!(ET[(Tuple(get_dualq(symmetries[n], ql[n]) for n in 1:N), d)
                 for (ql, d) in sp]; by = x -> x[1])
    end

    # ── No-truncation case ──────────────────────────────────────────────────
    result = svd(ct, (1, 2))
    U, S, Vd = result.U, result.S, result.Vd
    println("test_spaces_svdQS (no trunc): U bond=$(length(U.spaces[end])), " *
            "S=$(length(S.spaces[1]))/$(length(S.spaces[2])), " *
            "Vd bond=$(length(Vd.spaces[1])) sectors")

    # (a) U bond == S left
    @test qls(U.spaces[end]) == qls(S.spaces[1])

    # (b) S.spaces[2] is the exact dual of S.spaces[1]
    @test S.spaces[2] == dual_sp(S.spaces[1])

    # (c) Vd bond == S right
    @test qls(Vd.spaces[1]) == qls(S.spaces[2])

    # (d) U non-bond legs inherit spaces from input (left_legs = 1, 2)
    @test U.spaces[1] == ct.spaces[1]
    @test U.spaces[2] == ct.spaces[2]

    # (e) Vd non-bond leg inherits space from input (right_leg = 3)
    @test Vd.spaces[2] == ct.spaces[3]

    # ── Truncation case (Nkeep=1) ───────────────────────────────────────────
    result_t1 = svd(ct, (1, 2); Nkeep = 1)
    Ut, St, Vdt = result_t1.U, result_t1.S, result_t1.Vd
    println("test_spaces_svdQS (Nkeep=1): Ut bond=$(length(Ut.spaces[end])), " *
            "St=$(length(St.spaces[1]))/$(length(St.spaces[2])) sectors")

    # (f) Ut bond == St left
    @test qls(Ut.spaces[end]) == qls(St.spaces[1])

    # (g) St.spaces[2] is the exact dual of St.spaces[1]
    @test St.spaces[2] == dual_sp(St.spaces[1])

    # (h) Vdt bond == St right
    @test qls(Vdt.spaces[1]) == qls(St.spaces[2])

    # (i) St left ⊆ Ut bond (S sectors reduced; U/Vd bond keeps full pre-trunc space)
    @test issubset(qls(St.spaces[1]), qls(Ut.spaces[end]))

    # (j) Non-bond legs still match input after truncation
    @test Ut.spaces[1] == ct.spaces[1]
    @test Ut.spaces[2] == ct.spaces[2]
    @test Vdt.spaces[2] == ct.spaces[3]

    # ── Truncation case (Nkeep=2) ───────────────────────────────────────────
    result_t2 = svd(ct, (1, 2); Nkeep = 2)
    Ut, St, Vdt = result_t2.U, result_t2.S, result_t2.Vd
    println("test_spaces_svdQS (Nkeep=2): Ut bond=$(length(Ut.spaces[end])), " *
            "St=$(length(St.spaces[1]))/$(length(St.spaces[2])) sectors")

    # (f) Ut bond == St left
    @test qls(Ut.spaces[end]) == qls(St.spaces[1])

    # (g) St.spaces[2] is the exact dual of St.spaces[1]
    @test St.spaces[2] == dual_sp(St.spaces[1])

    # (h) Vdt bond == St right
    @test qls(Vdt.spaces[1]) == qls(St.spaces[2])

    # (i) St left ⊆ Ut bond (S sectors reduced; U/Vd bond keeps full pre-trunc space)
    @test issubset(qls(St.spaces[1]), qls(Ut.spaces[end]))

    # (j) Non-bond legs still match input after truncation
    @test Ut.spaces[1] == ct.spaces[1]
    @test Ut.spaces[2] == ct.spaces[2]
    @test Vdt.spaces[2] == ct.spaces[3]

end

function _test_svd_cgtsvd_prep(q::TLArray{T, QD, N, RD},
                               left_legs;
                               tol::Float64 = 1e-12) where {T, QD, N, RD}
    left_legs_ = Telum._normalize_svd_left_legs(left_legs, QD)
    right_legs_ = [l for l in 1:QD if l ∉ left_legs_]
    left_signatures, right_signatures =
        Telum._get_svd_sector_spaces(q, left_legs_, right_legs_)

    blocks_by_symm = Telum._get_svd_cgt_split_sectors(q, left_legs_, right_legs_; tol=tol)
    Telum._share_svd_sector_isometries!(blocks_by_symm, symm(q); tol=tol)
    active_sector_indices = [ri for ri in Telum.sector_slots(q) if !q.iszero[ri]]
    intermediate_qsectors =
        Telum._get_svd_intermediate_sector_dict(blocks_by_symm, active_sector_indices)
    intermediate_qsector_classes = Telum._get_svd_intermediate_sector_equivclasses(
        left_signatures, right_signatures, intermediate_qsectors)
    block_lookup_by_symm = ntuple(n -> Telum._build_svd_block_lookup(blocks_by_symm[n]), N)
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

function test_svd_cgtsvd_preprocess(option::LocalSpaceOptions; tol::Float64 = 1e-12)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    left_legs = (1, 2)
    split_blocks = Telum._get_svd_cgt_split_sectors(ct, left_legs; tol)
    expected_blocks = Telum._get_svd_cgt_split_sectors(ct, left_legs; tol)
    Telum._share_svd_sector_isometries!(expected_blocks, symm(ct); tol=tol)
    prep = _test_svd_cgtsvd_prep(ct, left_legs; tol=tol)
    split_blocks_via_svd = prep.blocks_by_symm

    @test length(split_blocks) == length(symm(ct))
    @test length(expected_blocks) == length(symm(ct))
    @test length(split_blocks_via_svd) == length(symm(ct))

    for n in 1:length(symm(ct))
        raw_blocks = split_blocks[n]
        shared_blocks = expected_blocks[n]
        shared_blocks_via_svd = split_blocks_via_svd[n]

        @test !isempty(raw_blocks)
        @test length(shared_blocks) == length(raw_blocks)
        @test length(shared_blocks) == length(shared_blocks_via_svd)

        blockkey(block) = (block.sector_index, block.left_spaces, block.right_spaces, block.q)
        shared_map = Dict(blockkey(block) => block for block in shared_blocks)
        shared_map_via_svd = Dict(blockkey(block) => block for block in shared_blocks_via_svd)

        @test Set(keys(shared_map)) == Set(keys(shared_map_via_svd))

        for key in keys(shared_map)
            shared_block = shared_map[key]
            shared_block_via_svd = shared_map_via_svd[key]
            @test shared_block.left_iso ≈ shared_block_via_svd.left_iso atol=tol rtol=tol
            @test shared_block.right_iso ≈ shared_block_via_svd.right_iso atol=tol rtol=tol
            @test shared_block.core ≈ shared_block_via_svd.core atol=tol rtol=tol
            qlabels, cgp, _, legdir =
                Telum._svd_symmetry_stored_leg_order(qlabeltype(ct), ct, shared_block.sector_index, Val(n))
            right_legs = Telum._svd_right_legs(Val(ndims(ct)), left_legs)
            left_legs_canon = Telum._svd_to_cgtidx(cgp, left_legs)
            right_legs_canon = Telum._svd_to_cgtidx(cgp, right_legs)
            expected_left_spaces, expected_right_spaces =
                Telum._svd_cgt_split_spaces(qlabels, legdir, left_legs_canon, right_legs_canon)
            @test shared_block.left_spaces == expected_left_spaces
            @test shared_block.right_spaces == expected_right_spaces
        end
    end

    for n in 1:length(symm(ct))
        Telum.isabelian(symm(ct)[n]) && continue

        left_groups = Dict{Any, Vector{Matrix{Float64}}}()
        right_groups = Dict{Any, Vector{Matrix{Float64}}}()
        for block in split_blocks_via_svd[n]
            left_key = (block.left_spaces, block.q)
            right_key = (block.right_spaces, block.q)
            push!(get!(left_groups, left_key, Matrix{Float64}[]), block.left_iso)
            push!(get!(right_groups, right_key, Matrix{Float64}[]), block.right_iso)
        end

        for mats in values(left_groups)
            first_mat = first(mats)
            for mat in mats
                @test mat ≈ first_mat atol=tol rtol=tol
            end
        end
        for mats in values(right_groups)
            first_mat = first(mats)
            for mat in mats
                @test mat ≈ first_mat atol=tol rtol=tol
            end
        end
    end
end

function test_svd_cgtsvd_intermediate_qrows(option::LocalSpaceOptions; tol::Float64 = 1e-12)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    left_legs = (1, 2)
    prep = _test_svd_cgtsvd_prep(ct, left_legs; tol=tol)
    split_blocks = prep.blocks_by_symm
    got = prep.intermediate_qsectors

    Sector = NTuple{length(symm(ct)), Tuple{Vararg{Int}}}
    expected = Dict{Sector, Vector{Int}}()
    for ri in Telum.sector_slots(ct)
        ct.iszero[ri] && continue
        qchoices = ntuple(length(symm(ct))) do n
            qs = [block.q for block in split_blocks[n] if block.sector_index == ri]
            sort!(qs; alg=MergeSort)
            unique!(qs)
            qs
        end

        @test all(!isempty, qchoices)
        for sector in Iterators.product(qchoices...)
            push!(get!(expected, Tuple(sector), Int[]), ri)
        end
    end

    @test got == expected
    for blocks in split_blocks
        row_q_pairs = [(block.sector_index, block.q) for block in blocks]
        @test row_q_pairs == sort(copy(row_q_pairs); alg=MergeSort)
    end
end

function test_svd_cgtsvd_intermediate_qrow_equivclasses(option::LocalSpaceOptions;
                                                         tol::Float64 = 1e-12)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    left_legs = (1, 2)
    prep = _test_svd_cgtsvd_prep(ct, left_legs; tol=tol)
    qsectors = prep.intermediate_qsectors
    got_left_sigs, got_right_sigs = prep.left_signatures, prep.right_signatures
    got = prep.intermediate_qsector_classes

    right_legs = Telum._svd_right_legs(Val(ndims(ct)), left_legs)
    left_sigs, right_sigs = Telum._get_svd_sector_spaces(ct, left_legs, right_legs)

    @test got_left_sigs == left_sigs
    @test got_right_sigs == right_sigs

    Sector = NTuple{length(symm(ct)), Tuple{Vararg{Int}}}
    expected = Dict{Sector, Vector{Vector{Int}}}()
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

        classes = Vector{Vector{Int}}()
        unassigned = Set(sectors)
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
        expected[sector] = classes
    end

    @test got == expected
end

function test_svd_cgtsvd_signature_order(option::LocalSpaceOptions; tol::Float64 = 1e-12)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    left_legs = (1, 4)
    prep = _test_svd_cgtsvd_prep(ct, left_legs; tol=tol)

    expected_signatures = [begin
        per_symm = ntuple(length(symm(ct))) do n
            qlabels, cgp, _, legdir =
                Telum._svd_symmetry_stored_leg_order(qlabeltype(ct), ct, sector_index, Val(n))
            nin = legdir[1]
            left_up = Tuple(
                qlabels[cgp[leg]] for leg in 1:ndims(ct)
                if leg in left_legs && cgp[leg] <= nin
            )
            left_dn = Tuple(
                qlabels[cgp[leg]] for leg in 1:ndims(ct)
                if leg in left_legs && cgp[leg] > nin
            )
            right_up = Tuple(
                qlabels[cgp[leg]] for leg in 1:ndims(ct)
                if leg ∉ left_legs && cgp[leg] <= nin
            )
            right_dn = Tuple(
                qlabels[cgp[leg]] for leg in 1:ndims(ct)
                if leg ∉ left_legs && cgp[leg] > nin
            )
            ((left_up, left_dn), (right_up, right_dn))
        end
        (ntuple(n -> per_symm[n][1], length(symm(ct))),
         ntuple(n -> per_symm[n][2], length(symm(ct))))
    end for sector_index in Telum.sector_slots(ct) if !ct.iszero[sector_index]]

    expected_left = first.(expected_signatures)
    expected_right = last.(expected_signatures)

    @test prep.left_signatures == expected_left
    @test prep.right_signatures == expected_right
end

function test_svd_stored_leg_order_preserves_physical_leg_ties()
    symm = (U1,)
    inds = (TLIndex("a", '+'), TLIndex("b", '+'), TLIndex("c", '-'), TLIndex("d", '-'))
    qlabels = [(((10,),), ((10,),), ((20,),), ((20,),))]
    wmats = Telum._wmat_vector(Telum.productsymm(symm), 1)
    RMTs = [ones(Float64, 1, 1, 1, 1, 1)]
    spaces = ntuple(leg -> [(qlabels[1][leg], 1)], 4)
    q = TLArray(symm, qlabels, wmats, RMTs, inds, spaces)

    qlabels_stored, cgp, stored_to_phys, legdir =
        Telum._svd_symmetry_stored_leg_order(qlabeltype(q), q, 1, Val(1))

    @test stored_to_phys == (1, 2, 3, 4)
    @test cgp == (1, 2, 3, 4)
    @test qlabels_stored == ((10,), (10,), (20,), (20,))
    @test legdir == (2, 2)
end

function _svd_cgtsvd_fixture(option::LocalSpaceOptions)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a
    return ct, (1, 2)
end

function _svd_reconstruction_permutation(left_legs, rank::Int)
    right_legs = [l for l in 1:rank if l ∉ left_legs]
    combined_order = vcat(collect(Int, left_legs), right_legs)
    perm = zeros(Int, rank)
    for (new_pos, orig_leg) in enumerate(combined_order)
        perm[orig_leg] = new_pos
    end
    return Tuple(perm)
end

function _diag_singular_values(q::TLArray)
    vals = Float64[]
    for sector_index in Telum.sector_slots(q)
        q.iszero[sector_index] && continue
        rmt = Array(Telum.sector_rmt(q, sector_index))
        mat = reshape(rmt, size(rmt, 1), size(rmt, 2))
        append!(vals, diag(mat))
    end
    return vals
end

function test_svd_cgtsvd_factorization(option::LocalSpaceOptions;
                                       cutoff::Float64 = 1e-12,
                                       tol::Float64 = 1e-9)
    ct, left_legs = _svd_cgtsvd_fixture(option)

    result = svd_cgtsvd(ct, left_legs; cutoff=cutoff)
    U, S, Vd = result.U, result.S, result.Vd

    @test U isa TLArray
    @test S isa TLArray
    @test Vd isa TLArray

    @test U.inds[end].dir == '-'
    @test S.inds[1].dir == '+'
    @test S.inds[2].dir == '+'
    @test Vd.inds[1].dir == '-'

    US = U * S
    rec = US * Vd
    rec_perm = permutedims(rec, _svd_reconstruction_permutation(left_legs, ndims(ct)))

    arr_ct = Array(to_sparse_array(ct))
    arr_rec = Array(to_sparse_array(rec_perm))
    ref_norm = max(norm(arr_ct), 1.0)
    diff = norm(arr_ct - arr_rec) / ref_norm
    @test diff < tol

    Uiso = U' * lock(U, ndims(U))
    arr_Uiso = Array(to_sparse_array(Uiso))
    @test norm(arr_Uiso - Matrix(I, size(arr_Uiso, 1), size(arr_Uiso, 2))) < tol

    Vdiso = lock(Vd, 1) * Vd'
    arr_Vdiso = Array(to_sparse_array(Vdiso))
    @test norm(arr_Vdiso - Matrix(I, size(arr_Vdiso, 1), size(arr_Vdiso, 2))) < tol
end

function test_svd_cgtsvd_heterogeneous_product_qlabels()
    q = getLocalSpace(FermionSOptions(3, :U1, :SU2, :SU3))

    result = svd(q.S, (1, 2))
    U, S, Vd = result.U, result.S, result.Vd

    @test U isa TLArray
    @test S isa TLArray
    @test Vd isa TLArray
    @test qlabeltype(U) == qlabeltype(q.S)
    @test qlabeltype(Vd) == qlabeltype(q.S)
end

function test_truncate_svd_cgtsvd(option::LocalSpaceOptions)
    ct, left_legs = _svd_cgtsvd_fixture(option)

    full_result = svd_cgtsvd(ct, left_legs; cutoff=0.0, get_lists=true)
    Sfull = full_result.S
    full_vals = sort(_diag_singular_values(Sfull); rev=true)
    nkeep = min(2, length(full_vals))
    @test nkeep > 0

    default_result = svd_cgtsvd(ct, left_legs; cutoff=0.0, Nkeep=nkeep)
    @test default_result isa Telum.SVDResult
    @test default_result.kept_list === nothing
    @test default_result.trunc_list === nothing

    result = svd_cgtsvd(ct, left_legs; cutoff=0.0, Nkeep=nkeep, get_lists=true)
    @test result isa Telum.SVDResult
    U, S, Vd = result.U, result.S, result.Vd
    kept_vals = sort(_diag_singular_values(S); rev=true)

    @test length(kept_vals) == nkeep
    @test kept_vals ≈ sort(first.(result.kept_list); rev=true)
    @test U.spaces[end] == S.spaces[1]
    @test Vd.spaces[1] == S.spaces[2]
    @test length(result.kept_list) == nkeep
    @test sort(vcat(first.(result.kept_list), first.(result.trunc_list)); rev=true) ≈ full_vals
    @test all(entry -> length(entry) == 4, result.kept_list)
    @test all(entry -> entry[4] >= 1, result.kept_list)

    cutoff_result = svd_cgtsvd(ct, left_legs; cutoff=0.0, Nkeep=0, get_lists=true)
    @test isempty(cutoff_result.kept_list)
    @test sort(first.(cutoff_result.trunc_list); rev=true) ≈ full_vals
end

function test_svd_cgtsvd_block_reduction(option::LocalSpaceOptions;
                                         tol::Float64 = 1e-12,
                                         recon_tol::Float64 = 1e-10)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itag="lurlur")
    qf  = TLArray(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    left_legs = (1, 2)
    right_legs = Telum._svd_right_legs(Val(ndims(ct)), left_legs)
    for ri in Telum.sector_slots(ct)
        ct.iszero[ri] && continue
        for n in 1:length(symm(ct))
            qlabels, cgp, _, legdir =
                Telum._svd_symmetry_stored_leg_order(qlabeltype(ct), ct, ri, Val(n))
            left_legs_canon = Telum._svd_to_cgtidx(cgp, left_legs)
            right_legs_canon = Telum._svd_to_cgtidx(cgp, right_legs)
            left_spaces, right_spaces =
                Telum._svd_cgt_split_spaces(qlabels, legdir, left_legs_canon, right_legs_canon)
            raw_blocks = Telum._get_svd_cgt_split_blocks(
                symm(ct)[n], qlabels, Telum.sector_wmat(ct, ri, n),
                cgp, legdir, left_legs, right_legs)
            reduced_pairs = [(raw, Telum._reduce_svd_cgt_block(
                raw, ri, left_spaces, right_spaces; tol=tol)) for raw in raw_blocks]
            filter!(pair -> !isnothing(pair[2]), reduced_pairs)
            reduced_blocks = [pair[2] for pair in reduced_pairs]
            kept_raw_blocks = [pair[1] for pair in reduced_pairs]

            for (raw, reduced) in zip(kept_raw_blocks, reduced_blocks)
                @test reduced.sector_index == ri
                @test reduced.left_spaces == left_spaces
                @test reduced.right_spaces == right_spaces
                @test reduced.q == raw.q
                @test size(reduced.left_iso, 1) == raw.omL
                @test size(reduced.right_iso, 1) == raw.omR
                @test size(reduced.left_iso, 2) <= raw.omL
                @test size(reduced.right_iso, 2) <= raw.omR
                @test size(reduced.core) ==
                      (size(reduced.left_iso, 2), size(reduced.right_iso, 2), size(raw.coeffs, 3))

                left_gram = reduced.left_iso' * reduced.left_iso
                right_gram = reduced.right_iso' * reduced.right_iso
                @test left_gram ≈ Matrix{Float64}(I, size(left_gram, 1), size(left_gram, 2)) atol=recon_tol rtol=recon_tol
                @test right_gram ≈ Matrix{Float64}(I, size(right_gram, 1), size(right_gram, 2)) atol=recon_tol rtol=recon_tol
            end
        end
    end
end

# ─── test_truncate_svdQS ─────────────────────────────────────────────────────
# Verify that Nkeep truncation keeps the largest singular values.
# Original-leg spaces on U and Vd must stay intact.
# ─────────────────────────────────────────────────────────────────────────────
function test_truncate_svdQS(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    @test !isempty(q.I.sectors)

    kept_rows = copy(q.I.sectors)
    A = TLArray(symm(q.I), kept_rows, q.I.inds, q.I.spaces)

    offset = 0.0
    all_positive_vals = Float64[]
    for r in A.sectors
        sz = size(r.RMT.data)
        n  = sz[1]
        vals = collect(offset .+ (1.0:n))
        offset += n + 1.0
        append!(all_positive_vals, vals)
        r.RMT.data .= reshape(Matrix(Diagonal(vals)), sz)
    end

    npositive_keep = min(2, length(all_positive_vals))
    result_top = svd(A, (1,); Nkeep = npositive_keep)
    Utop, Stop, Vdtop = result_top.U, result_top.S, result_top.Vd

    @test Utop.spaces[1] == A.spaces[1]
    @test Vdtop.spaces[2] == A.spaces[2]
    @test Utop.spaces[2] == Stop.spaces[1]
    @test Vdtop.spaces[1] == Stop.spaces[2]

    kept_vals = sort(_diag_singular_values(Stop))
    expected_vals = sort(all_positive_vals; rev = true)[1:npositive_keep] |> sort
    @test kept_vals ≈ expected_vals

    result = svd(A, (1,); Nkeep = length(all_positive_vals))
    U, S, Vd = result.U, result.S, result.Vd

    @test U.spaces[1] == A.spaces[1]
    @test Vd.spaces[2] == A.spaces[2]
    @test U.spaces[2] == S.spaces[1]
    @test Vd.spaces[1] == S.spaces[2]
end

# ─── test_lock_reduce ─────────────────────────────────────────────────────────
# Verify selective lock-level reduction in contraction:
#
# Rule: a free output leg has its lock decremented by 1 only when the *other*
# tensor in the contraction has at least one leg with the same itag.
#
#   Scenario 1 — match present   : A free leg "free" lock=1 + B has "free" → lock → 0
#   Scenario 2 — no match        : A free leg "unique" lock=1, B has no "unique" → lock stays 1
#   Scenario 3 — lock already 0  : match present but lock=0 → lock stays 0
# ─────────────────────────────────────────────────────────────────────────────
function test_lock_reduce(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    # ── Scenario 1: matching leg present → lock decrements ───────────────────
    # q.I has legs ('+', '-').  A1 is built from q.I (same dirs), B1 from q.I'
    # (dirs flipped to ('-', '+')) so they can be contracted on "ct".
    # A1: ("ct" '+' lock=0,  "free" '-' lock=1)
    # B1: ("ct" '-' lock=0,  "free" '+' lock=0)
    # Contract A1*B1 on "ct"; A1's "free" lock=1 has a match in B1 → lock → 0.
    A1 = TLArray(q.I,
        (TLIndex("ct",   '+', 0, 0),
         TLIndex("free", '-', 0, 1)))
    B1 = TLArray(q.I',
        (TLIndex("ct",   '-', 0, 0),
         TLIndex("free", '+', 0, 0)))

    C1 = A1 * B1
    free_pos1 = findfirst(idx -> idx.itags == "free" && idx.dir == '-', C1.inds)
    @test C1.inds[free_pos1].lock == 0
    println("test_lock_reduce (match):    \"free\" lock=$(C1.inds[free_pos1].lock) (expected 0)")

    # ── Scenario 2: no matching leg → lock unchanged ─────────────────────────
    # A2: ("ct" '+' lock=0,  "unique" '-' lock=1)   — from q.I
    # B2: ("ct" '-' lock=0,  "other"  '+' lock=0)   — from q.I'; "unique" absent
    A2 = TLArray(q.I,
        (TLIndex("ct",     '+', 0, 0),
         TLIndex("unique", '-', 0, 1)))
    B2 = TLArray(q.I',
        (TLIndex("ct",    '-', 0, 0),
         TLIndex("other", '+', 0, 0)))

    C2 = A2 * B2
    unique_pos = findfirst(idx -> idx.itags == "unique", C2.inds)
    @test C2.inds[unique_pos].lock == 1
    println("test_lock_reduce (no match): \"unique\" lock=$(C2.inds[unique_pos].lock) (expected 1)")

    # ── Scenario 3: lock already 0 with match → stays 0 ─────────────────────
    # A3's "free" lock=0 and B1 has a matching "free" leg, but 0 cannot go lower.
    # Must use explicit contract (not *) to avoid auto-contracting both "ct" and "free".
    A3 = TLArray(q.I,
        (TLIndex("ct",   '+', 0, 0),
         TLIndex("free", '-', 0, 0)))
    C3 = contract(A3, (1,), B1, (1,))
    free_pos3 = findfirst(idx -> idx.itags == "free" && idx.dir == '-', C3.inds)
    @test C3.inds[free_pos3].lock == 0
    println("test_lock_reduce (lock=0):   \"free\" lock=$(C3.inds[free_pos3].lock) (expected 0)")
end

# ─── test_contract_requires_matching_spaces_in_star ─────────────────────────
# Verify that automatic contraction via `*` does not match tagged legs when
# their precomputed space metadata differs, even if the TLIndex fields match.
function test_contract_requires_matching_spaces_in_star(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    A = TLArray(q.I,
        (TLIndex("ct", '+', 0, 0),
         TLIndex("",   '-', 0, 0)))
    B = TLArray(q.I',
        (TLIndex("ct", '-', 0, 0),
         TLIndex("",   '+', 0, 0)))

    first_sector, first_dim = first(B.spaces[1])
    bad_leg1 = vcat([(first_sector, first_dim + 1)], B.spaces[1][2:end])
    B_bad = Telum.TLArray(symm(B), copy(B.qlabels), deepcopy(B.wmats),
                                 deepcopy(B.RMTs), B.inds, (bad_leg1, B.spaces[2]))

    @test_throws AssertionError A * B_bad
end
