@testset "spaces of svd" begin
    test_spaces_svdQS(FermionSOptions(1, :U1, :SU2, nothing))
    test_spaces_svdQS(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD preprocessing for svd" begin
    test_svd_cgtsvd_preprocess(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_preprocess(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD intermediate qlabels for svd" begin
    test_svd_cgtsvd_intermediate_qrows(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_intermediate_qrows(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD intermediate qlabel row classes for svd" begin
    test_svd_cgtsvd_intermediate_qrow_equivclasses(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_intermediate_qrow_equivclasses(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD signature order for svd" begin
    test_svd_cgtsvd_signature_order(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_signature_order(FermionSOptions(3, :U1, :SU2, :SU3))
    test_svd_stored_leg_order_preserves_physical_leg_ties()
end

@testset "CGTSVD block reduction for svd" begin
    test_svd_cgtsvd_block_reduction(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_block_reduction(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD full svd factorization" begin
    test_svd_cgtsvd_factorization(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_factorization(FermionSOptions(3, :U1, :SU2, :SU3))
    test_svd_cgtsvd_heterogeneous_product_qlabels()
    test_svd_cgtsvd_zero_leading_sector()
end

@testset "CGTSVD truncation" begin
    test_truncate_svd_cgtsvd(FermionSOptions(1, :U1, :SU2, nothing))
    test_truncate_svd_cgtsvd(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "svd truncation of TLArray" begin
    test_truncate_svdQS(FermionSOptions(1, :U1, :SU2, nothing))
    test_truncate_svdQS(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "spaces of eigen" begin
    test_spaces_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_spaces_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "missing spaces of eigen" begin
    test_missing_spaces_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_missing_spaces_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
    test_missing_spaces_eigen_zero_diagonal()
end

@testset "truncate missing zero spaces of eigen" begin
    test_truncate_missing_zero_spaces_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_truncate_missing_zero_spaces_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "svd test" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q = getLocalSpace(option)
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a = getIdentity((qi1, 2), (qi2, 2))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    ct = qf * a
    test_svdQS(ct, [2, 4])
    test_svdQS(ct, [1, 4])
    test_svdQS(ct, [1, 2])
end

@testset "eig of TLArray" begin
    test_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig autodetect of TLArray" begin
    test_eigen_autodetect(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_autodetect(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig permuted input of TLArray" begin
    test_eigen_permuted_input(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_permuted_input(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig Hermitian leg guard of TLArray" begin
    test_eigen_hermitian_leg_guard(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_hermitian_leg_guard(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig truncation of TLArray" begin
    test_discard_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_discard_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig truncation tol of TLArray" begin
    test_discard_eigen_tol(FermionSOptions(1, :U1, :SU2, nothing))
    test_discard_eigen_tol(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig full discard of TLArray" begin
    test_eigen_general_discard(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_general_discard(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig discard itag of TLArray" begin
    test_discard_eigen_itag(FermionSOptions(1, :U1, :SU2, nothing))
    test_discard_eigen_itag(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "svd_leg" begin
    @testset "shape and reconstruction (random 3-D, each leg)" begin
        A = randn(3, 4, 5)

        for leg in 1:3
            U, SV, S = svd_leg(A, leg; cutoff=1e-12)
            chi = length(S)

            expected_size = ntuple(i -> i == leg ? chi : size(A, i), 3)
            @test size(SV) == expected_size
            @test size(U) == (size(A, leg), chi)

            rec = reconstruct(U, SV, leg)
            @test size(rec) == size(A)
            @test norm(A - rec) < 1e-9
        end
    end

    @testset "low-rank truncation" begin
        u1, u2 = randn(4), randn(4)
        v1, v2 = randn(3), randn(3)
        w1, w2 = randn(5), randn(5)
        A = reshape(u1, 4, 1, 1) .* reshape(v1, 1, 3, 1) .* reshape(w1, 1, 1, 5) .+
            reshape(u2, 4, 1, 1) .* reshape(v2, 1, 3, 1) .* reshape(w2, 1, 1, 5)

        U, SV, S = svd_leg(A, 1; cutoff=1e-10)
        @test length(S) == 2
        rec = reconstruct(U, SV, 1)
        @test norm(A - rec) < 1e-9
    end

    @testset "maxdim truncation" begin
        A = randn(6, 5, 4)

        _, _, S_full = svd_leg(A, 2; cutoff=1e-12)

        U3, SV3, S3 = svd_leg(A, 2; cutoff=1e-12, maxdim=3)
        @test length(S3) <= 3
        @test length(S3) <= length(S_full)
        @test S3 ≈ S_full[1:length(S3)]
    end

    @testset "2-D matrix (standard SVD)" begin
        M = randn(5, 4)

        for leg in 1:2
            U, SV, S = svd_leg(M, leg; cutoff=1e-12)
            rec = reconstruct(U, SV, leg)
            @test norm(M - rec) < 1e-9
        end
    end
end
