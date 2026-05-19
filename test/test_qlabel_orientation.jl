function _assert_qlabel_storage(q::TLArray)
    @test size(q.qlabels, 1) == ndims(q)
    @test size(q.qlabels, 2) >= Telum.nsectors(q)
    for sector_index in 1:Telum.nsectors(q)
        @test Tuple(q.qlabels[:, sector_index]) ==
              ntuple(leg -> Telum.sector_qlabel(q, sector_index, leg), ndims(q))
    end
end

function _assert_wmat_storage(q::TLArray)
    nonabelian = Telum.nonabelian_symmetry_indices(Telum.productsymm(q))
    @test length(q.wmats) == Telum.nsectors(q)
    @test eltype(q.wmats) <: NTuple{length(nonabelian), Matrix{Float64}}
    for sector_index in 1:Telum.nsectors(q), n in 1:length(symm(q))
        wmat = Telum.sector_wmat(q, sector_index, n)
        if n in nonabelian
            slot = Telum.nonabelian_wmat_slot(Telum.productsymm(q), n)
            @test wmat === q.wmats[sector_index][slot]
        else
            @test size(wmat) == (1, 1)
            @test wmat[1] == 1.0
        end
    end
end

function _assert_metadata_inferred(q::TLArray)
    Telum.nsectors(q) == 0 && return
    @inferred Telum.sector_qlabel(q, 1, 1)
    @inferred Telum.sector_wmat(q, 1, 1)
    @inferred Telum._sector_cgt_metadata(q, 1, 1)
end

function _assert_zero_sector_constructor_filter(q::TLArray)
    Telum.nsectors(q) == 0 && return

    old_n = Telum.nsectors(q)
    qlabels = Matrix{eltype(q.qlabels)}(undef, ndims(q), old_n + 1)
    qlabels[:, 1:old_n] = q.qlabels[:, 1:old_n]
    qlabels[:, old_n + 1] = q.qlabels[:, 1]

    wmats = deepcopy(q.wmats)
    push!(wmats, deepcopy(q.wmats[1]))
    RMTs = deepcopy(q.RMTs)
    zero_rmt = deepcopy(Telum.sector_rmt(q, 1))
    fill!(zero_rmt.data, zero(eltype(zero_rmt.data)))
    push!(RMTs, zero_rmt)

    filtered = TLArray(symm(q), qlabels, wmats, RMTs, q.inds, q.spaces)
    @test Telum.nsectors(filtered) == old_n
    @test length(filtered.wmats) == old_n
    @test length(filtered.RMTs) == old_n
    @test size(filtered.qlabels, 2) == old_n + 1
end

@testset "wmat slot mappings infer" begin
    PS = ProductSymm(U1, SU{2}, U1, SU{2})
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(1))) === nothing
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(2))) == 1
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(3))) === nothing
    @test @inferred(Telum.wmat_tuple_slot(PS, Val(4))) == 2
    @test @inferred(Telum.product_symmetry_index_from_wmat_slot(PS, Val(1))) == 2
    @test @inferred(Telum.product_symmetry_index_from_wmat_slot(PS, Val(2))) == 4
end

@testset "TLArray qlabel storage orientation" begin
    fermion_u1 = getLocalSpace(FermionOptions(U1))
    fermion_su2 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("in", "out", "op"))

    samples = (
        fermion_u1.I,
        fermion_u1.F,
        fermion_su2.I,
        fermion_su2.F,
        empty_qspace((U1, SU{2}), (TLIndex("empty_in", '+'), TLIndex("empty_out", '-'))),
        getvac(fermion_su2.I),
    )

    @testset "storage columns are sectors" begin
        for q in samples
            _assert_qlabel_storage(q)
            _assert_wmat_storage(q)
        end
    end

    @testset "constructor drops zero sectors without compacting qlabels" begin
        for q in samples
            _assert_zero_sector_constructor_filter(q)
        end
    end

    @testset "metadata helpers infer and preserve ordering" begin
        for q in samples
            _assert_metadata_inferred(q)
            for sector_index in 1:Telum.nsectors(q), n in 1:length(symm(q))
                qlabels, cgp, legdir = Telum._sector_cgt_metadata(q, sector_index, n)
                stored_to_phys = Telum._stored_leg_order(q, sector_index, n)
                @test qlabels == ntuple(i -> Telum.sector_qlabel(q, sector_index, stored_to_phys[i])[n], ndims(q))
                @test cgp == ntuple(leg -> findfirst(==(leg), stored_to_phys), ndims(q))
                @test legdir == (count(leg -> q.inds[leg].dir == '+', 1:ndims(q)),
                                 count(leg -> q.inds[leg].dir == '-', 1:ndims(q)))
            end
        end
    end

    @testset "field operations keep orientation" begin
        q = fermion_su2.F
        _assert_qlabel_storage(copy(q))
        _assert_qlabel_storage(TLArray(q, :))
        _assert_qlabel_storage(permutedims(q, (2, 1, 3)))
        _assert_qlabel_storage(conj(q))
        _assert_qlabel_storage(q + q)
        @test norm(q + q) ≈ 2norm(q)
        _assert_qlabel_storage(sum((q, q, q)))
        @test norm(sum((q, q, q))) ≈ 3norm(q)

        q_perm = permutedims(q, (2, 1, 3))
        from_permuted_only = sum((zero(q), q_perm))
        _assert_qlabel_storage(from_permuted_only)
        @test norm(from_permuted_only - q) < 1e-10

        sorted = copy(q)
        Telum.sort_sectors!(sorted)
        _assert_qlabel_storage(sorted)
    end

    @testset "decomposition assembly keeps orientation" begin
        U, S, Vd = svd(fermion_su2.F, (1,))
        _assert_qlabel_storage(U)
        _assert_qlabel_storage(S)
        _assert_qlabel_storage(Vd)

        eres = eigen(TLArray(fermion_su2.I, ("eig", "eig")); hermitian=true)
        _assert_qlabel_storage(eres.V)
        _assert_qlabel_storage(eres.D)
    end
end
