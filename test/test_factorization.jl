function _factorization_svd_inputs(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a = getIdentity((qi1, 2), (qi2, 2))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    ct = qf * a

    inputs = Tuple{String, TLArray, Vector{Int}}[
        ("identity", q.I, [1]),
        ("fermion operator", q.F, [1, 2]),
        ("contracted operator [2, 4]", ct, [2, 4]),
        ("contracted operator [1, 4]", ct, [1, 4]),
        ("contracted operator [1, 2]", ct, [1, 2]),
    ]
    hasproperty(q, :S) && push!(inputs, ("spin operator", q.S, [1, 2]))
    return inputs
end

function _test_svd_truncation_matches_full(q::TLArray, left_legs)
    full = svd(q, left_legs; cutoff=0.0, get_lists=true)
    full_singular_values = sort(first.(full.kept_list); rev=true)
    @test !isempty(full_singular_values)

    for nkeep in eachindex(full_singular_values)
        @testset "Nkeep=$nkeep" begin
            truncated = svd(q, left_legs; cutoff=0.0, Nkeep=nkeep, get_lists=true)
            kept_singular_values = sort(first.(truncated.kept_list); rev=true)

            @test kept_singular_values ≈ full_singular_values[1:nkeep]
            @test length(truncated.kept_list) == nkeep
            @test length(truncated.kept_list) + length(truncated.trunc_list) == length(full.kept_list)
            @test truncated.U.spaces[end] == truncated.S.spaces[1]
            @test truncated.Vd.spaces[1] == truncated.S.spaces[2]
        end
    end
end

function _dense_svd_values(q::TLArray, left_legs)
    left = collect(Int, left_legs)
    right = [leg for leg in 1:ndims(q) if leg ∉ left]
    arr = Array(to_sparse_array(q))
    arr_perm = permutedims(arr, (left..., right...))
    left_dim = prod(size(arr, leg) for leg in left; init=1)
    right_dim = prod(size(arr, leg) for leg in right; init=1)
    return sort(LinearAlgebra.svdvals(reshape(arr_perm, left_dim, right_dim)); rev=true)
end

function _test_svd_singular_values_match_dense(q::TLArray, left_legs)
    result = svd(q, left_legs; cutoff=0.0, get_lists=true)
    telum_vals = Float64[]
    for (singular_value, degeneracy, _, _) in result.kept_list
        for _ in 1:degeneracy
            push!(telum_vals, singular_value)
        end
    end
    filter!(>(1e-10), telum_vals)
    sort!(telum_vals; rev=true)

    dense_vals = filter(>(1e-10), _dense_svd_values(q, left_legs))

    @test length(telum_vals) == length(dense_vals)
    @test telum_vals ≈ dense_vals atol=1e-10 rtol=1e-10
end

function _test_identity_svd_values_are_one(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    result = svd(q.I, (1,); cutoff=0.0, get_lists=true)

    @test !isempty(result.kept_list)
    @test all(entry -> isapprox(first(entry), 1.0; atol=1e-10, rtol=1e-10), result.kept_list)
end

@testset "svd reconstruction" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(2, :U1, :SU2, :SU2),
        FermionSOptions(3, :U1, :SU2, :SU3),
        )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            @testset "$label" begin
                diff = test_svdQS(q, left_legs; verbose=false)
                @test diff < 1e-9
            end
        end
    end
end

@testset "identity svd singular values are one" begin
    _test_identity_svd_values_are_one(FermionSOptions(1, :U1, :SU2, nothing))
    _test_identity_svd_values_are_one(FermionSOptions(3, :U1, :SU2, :SU3))
end

test_svd_diag_rmt()

@testset "svd singular values match dense SVD" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(3, :U1, :SU2, :SU3),
    )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            @testset "$label" begin
                _test_svd_singular_values_match_dense(q, left_legs)
            end
        end
    end
end

@testset "svd truncation keeps leading singular values" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(3, :U1, :SU2, :SU3),
    )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            @testset "$label" begin
                _test_svd_truncation_matches_full(q, left_legs)
            end
        end
    end
end

@testset "factorizations accept lazy contraction input" begin
    q = getLocalSpace(SpinOptions(:SU2, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.S, ("bond", "bond", "right"))
    lazy = contract(left, (2,), right, (1,))

    @test lazy isa TLArrayContraction
    @test svd(lazy, (1,)).U isa TLArray
    @test qr(lazy, (1,)).Q isa TLArray
end

@testset "qr reconstruction" begin
    q0 = getLocalSpace(SpinOptions(nothing, 1))
    _test_qr_reconstructs(q0.Sz, (1,))
    _test_qr_reconstructs(q0.Sm, (1,))

    q_u1 = getLocalSpace(SpinOptions(:U1, 1))
    _test_qr_reconstructs(q_u1.Sz, (1,))
    _test_qr_reconstructs(q_u1.Sm, (1, 2))

    q_su2 = getLocalSpace(SpinOptions(:SU2, 1))
    _test_qr_reconstructs(q_su2.S, (1, 2))

    q_fermion = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    _test_qr_reconstructs(q_fermion.F, (1, 2))
end

@testset "qr reconstruction higher-rank tensors" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(2, :U1, :SU2, :SU2),
        FermionSOptions(3, :U1, :SU2, :SU3),
        )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            ndims(q) > 2 || continue
            @testset "$label" begin
                _test_qr_reconstructs(q, left_legs; tol=1e-8)
            end
        end
    end
end

@testset "qr accepts DiagRMT sector storage" begin
    symm = (U1,)
    qlabels = [(((0,),), ((0,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    spaces = ([(((0,),), 2)], [(((0,),), 2)])
    q = TLArray(symm, qlabels, wmatdata, wmatinfo,
                [DiagRMT([3.0, -1.5], Val(3), (1, 2))],
                (TLIndex("left", '+'), TLIndex("right", '-')), spaces)

    @test all(i -> q.iszero[i] || q.RMTs[i] isa DiagRMT, eachindex(q.RMTs))
    _test_qr_reconstructs(q, (1,); tol=1e-10)

    q_su2 = get1jtensor(getLocalSpace(SpinOptions(:SU2, 1)).I, 1)
    @test all(i -> q_su2.iszero[i] || q_su2.RMTs[i] isa DiagRMT, eachindex(q_su2.RMTs))
    _test_qr_reconstructs(q_su2, (1,); tol=1e-10)
end

@testset "qr accepts TLArrayView input" begin
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))

    scaled_view = 2.0 * q.F
    @test scaled_view isa TLArrayView
    _test_qr_reconstructs(scaled_view, (1, 2))

    permuted_view = permutedims(q.F, (3, 1, 2))
    @test permuted_view isa TLArrayView
    _test_qr_reconstructs(permuted_view, (2, 3))
end

@testset "qr keyword selection and errors" begin
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    q4 = addSingleton(q.F, 1; itag="left_aux", dir='+')
    explicit = qr(q4, (1, 2, 3), "kwqr")
    keyword = qr(q4, "kwqr"; dir='+')
    @test norm(explicit.Q * explicit.R - keyword.Q * keyword.R) < 1e-9
    @test_throws ArgumentError qr(q4)
    @test_throws ArgumentError qr(q4, Int[])
    @test_throws ArgumentError qr(q4, collect(1:ndims(q4)))
    @test_throws ArgumentError qr(q4, [1, 1])
    @test_throws ArgumentError qr(q4, [0])
end
