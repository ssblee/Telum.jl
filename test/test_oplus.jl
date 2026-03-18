function _make_test_qspace_rank4_oplus()
    option = FermionSOptions(U1, SU{2}, nothing, 1)
    q0  = getLocalSpace(option)
    qi1 = QSpace(q0.I, ("b1a", "b1b"))
    qi2 = QSpace(q0.I, ("b2a", "b2b"))
    qi3 = QSpace(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itag="fused")
    return QSpace(a4, ("l1", "l2", "l3", "fused"))
end

@testset "QSpace constructor wmat orientation" begin
    option = FermionSOptions(U1, SU{2}, SU{3}, 3)
    q0 = getLocalSpace(option)
    q = QSpace(q0.F, ("site1", "site2", "op"))

    rows_neg = deepcopy(q.rows)
    for r in rows_neg
        r.cgrs[1].wmat[:] .*= -1
    end

    q_oriented = QSpace(q.symm, rows_neg, q.inds, q.spaces)
    arr_ref = -Array(to_sparse_array(q))
    arr_oriented = Array(to_sparse_array(q_oriented))

    @test all(r.cgrs[1].wmat[1] >= 0 for r in q_oriented.rows)
    @test norm(arr_oriented - arr_ref) < 1e-10
end

@testset "QSpace direct sum" begin
    option = FermionSOptions(U1, SU{2}, SU{3}, 3)
    q0 = getLocalSpace(option)
    A = QSpace(q0.F, ("site1", "site2", "op"))
    B = 2.0 * A

    @testset "binary wrapper" begin
        qsum = oplus(A, B, 3)
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector selected legs" begin
        qsum = oplus([A, B, 3.0 * A], (2, 3))
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B, 3.0 * A], (2, 3))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], spaces_ref[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector allows different green fields" begin
        B_green = QSpace(B, (QSpaces.change_green(B.inds[1]), B.inds[2], QSpaces.change_green(B.inds[3])))
        qsum = oplus([A, B_green], 3)
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector validation" begin
        B_bad = QSpace(B, ("other1", "site2", "op"))
        @test_throws ArgumentError oplus([A, nothing], 3)
        @test_throws ArgumentError oplus([A, B_bad], 3)
    end

    @testset "matrix concatenation" begin
        q4 = _make_test_qspace_rank4_oplus()
        mat = Matrix{QSpace}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = 2.0 * q4
        mat[1, 2] = 3.0 * q4
        mat[2, 2] = 4.0 * q4

        qsum = oplus(mat, (3, 4))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref(mat, (3, 4))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == q4.inds
        @test qsum.spaces == (q4.spaces[1], q4.spaces[2], spaces_ref[3], spaces_ref[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix allows different green fields" begin
        q4 = _make_test_qspace_rank4_oplus()
        q4_green = QSpace(q4, (QSpaces.change_green(q4.inds[1]), q4.inds[2], q4.inds[3], QSpaces.change_green(q4.inds[4])))
        mat = Matrix{QSpace}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = 2.0 * q4_green
        mat[1, 2] = 3.0 * q4_green
        mat[2, 2] = 4.0 * q4

        qsum = oplus(mat, (3, 4))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref([q4 3.0 * q4; 2.0 * q4 4.0 * q4], (3, 4))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == q4.inds
        @test qsum.spaces == (q4.spaces[1], q4.spaces[2], spaces_ref[3], spaces_ref[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix missing entries and tuple-valued dims" begin
        q4 = _make_test_qspace_rank4_oplus()
        mat = Matrix{Any}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = nothing
        mat[1, 2] = 2.0 * q4
        mat[2, 2] = 3.0 * q4

        filled = complete_oplus_matrix(mat, ((1, 2), 3))
        qsum = oplus(mat, ((1, 2), 3))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref(mat, ((1, 2), 3))
        arr_qsum = Array(to_sparse_array(qsum))

        @test filled isa Matrix{QSpace}
        @test filled[2, 1].inds == q4.inds
        @test isempty(filled[2, 1].rows)
        @test filled[2, 1].spaces[1] == q4.spaces[1]
        @test filled[2, 1].spaces[2] == q4.spaces[2]
        @test filled[2, 1].spaces[3] == q4.spaces[3]
        @test filled[2, 1].spaces[4] == q4.spaces[4]
        @test qsum.inds == q4.inds
        @test qsum.spaces == (spaces_ref[1], spaces_ref[2], spaces_ref[3], q4.spaces[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix inference failure" begin
        q4 = _make_test_qspace_rank4_oplus()
        mat = Matrix{Any}(undef, 2, 2)
        mat[1, 1] = nothing
        mat[2, 1] = nothing
        mat[1, 2] = q4
        mat[2, 2] = 2.0 * q4

        @test_throws ArgumentError oplus(mat, (3, 4))
    end
end
