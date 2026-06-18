function _make_test_qspace_rank4_oplus()
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0  = getLocalSpace(option)
    qi1 = TLArray(q0.I, ("b1a", "b1b"))
    qi2 = TLArray(q0.I, ("b2a", "b2b"))
    qi3 = TLArray(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itag="fused")
    return TLArray(a4, ("l1", "l2", "l3", "fused"))
end

@testset "TLArray constructor wmat orientation" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    q = TLArray(q0.F, ("site1", "site2", "op"))

    wmatdata_neg = copy(q.wmatdata)
    wmatinfo_neg = copy(q.wmatinfo)
    for sector_index in Telum.sector_slots(q)
        offset, nrow, ncol = wmatinfo_neg[sector_index][1]
        offset == 0 && continue
        wmatdata_neg[offset:offset + nrow * ncol - 1] .*= -1
    end

    q_oriented = Telum.TLArray(symm(q), copy(q.qlabels), wmatdata_neg, wmatinfo_neg,
                                      deepcopy(q.RMTs), q.inds, q.spaces)
    arr_ref = -Array(to_sparse_array(q))
    arr_oriented = Array(to_sparse_array(q_oriented))

    @test all(Telum.sector_wmat_slot(q_oriented, sector_index, 1)[1] >= 0
              for sector_index in 1:Telum.nsectors(q_oriented))
    @test norm(arr_oriented - arr_ref) < 1e-10
end

@testset "TLArray direct sum" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    A = TLArray(q0.F, ("site1", "site2", "op"))
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

    @testset "vector permutes inputs to reference index order" begin
        B_perm = permutedims(B, (2, 1, 3))
        qsum = oplus([A, B_perm], 3)
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "keyword leg selection" begin
        qsum = oplus([A, B]; itag="op")
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10

        qsum_dir = oplus(A, B; dir='-')
        arr_ref_dir, spaces_ref_dir = _dense_vector_oplus_ref([A, B], (2, 3))
        arr_qsum_dir = Array(to_sparse_array(qsum_dir))

        @test qsum_dir.spaces == (A.spaces[1], spaces_ref_dir[2], spaces_ref_dir[3])
        @test size(arr_qsum_dir) == size(arr_ref_dir)
        @test norm(arr_qsum_dir - arr_ref_dir) < 1e-10
    end

    @testset "vector allows different green fields" begin
        B_green = TLArray(B, (Telum.change_green(B.inds[1]), B.inds[2], Telum.change_green(B.inds[3])))
        qsum = oplus([A, B_green], 3)
        arr_ref, spaces_ref = _dense_vector_oplus_ref([A, B], 3)
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == A.inds
        @test qsum.spaces == (A.spaces[1], A.spaces[2], spaces_ref[3])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "vector validation" begin
        B_bad = TLArray(B, ("other1", "site2", "op"))
        @test_throws ArgumentError oplus([A, nothing], 3)
        @test_throws ArgumentError oplus([A, B_bad], 3)
    end

    @testset "matrix concatenation" begin
        q4 = _make_test_qspace_rank4_oplus()
        mat = Matrix{TLArray}(undef, 2, 2)
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

    @testset "matrix permutes entries to reference index order" begin
        q4 = _make_test_qspace_rank4_oplus()
        mat = Matrix{TLArray}(undef, 2, 2)
        mat[1, 1] = q4
        mat[2, 1] = permutedims(2.0 * q4, (2, 1, 3, 4))
        mat[1, 2] = 3.0 * q4
        mat[2, 2] = 4.0 * q4

        qsum = oplus(mat, (3, 4))
        arr_ref, spaces_ref = _dense_matrix_oplus_ref([q4 3.0 * q4; 2.0 * q4 4.0 * q4], (3, 4))
        arr_qsum = Array(to_sparse_array(qsum))

        @test qsum.inds == q4.inds
        @test qsum.spaces == (q4.spaces[1], q4.spaces[2], spaces_ref[3], spaces_ref[4])
        @test size(arr_qsum) == size(arr_ref)
        @test norm(arr_qsum - arr_ref) < 1e-10
    end

    @testset "matrix allows different green fields" begin
        q4 = _make_test_qspace_rank4_oplus()
        q4_green = TLArray(q4, (Telum.change_green(q4.inds[1]), q4.inds[2], q4.inds[3], Telum.change_green(q4.inds[4])))
        mat = Matrix{TLArray}(undef, 2, 2)
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

        @test filled isa Matrix{TLArray}
        @test filled[2, 1].inds == q4.inds
        @test Telum.nsectors(filled[2, 1]) == 0
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
