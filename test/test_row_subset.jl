function _field_subset_equal(q1::TLArray, q2::TLArray)
    q1.qlabels == q2.qlabels || return false
    q1.wmats == q2.wmats || return false
    length(q1.RMTs) == length(q2.RMTs) || return false
    all(Telum.sector_rmt(q1, i) == Telum.sector_rmt(q2, i) for i in 1:Telum.nsectors(q1))
end

@testset "TLArray sector subset selection" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q = getLocalSpace(option, ("lur", "lur", "op")).I
    @test Telum.nsectors(q) >= 2

    subset = TLArray(q, 2:Telum.nsectors(q))
    subset_getsub = getsub(q, 2:Telum.nsectors(q))
    subset_mask = TLArray(q, vcat(false, trues(Telum.nsectors(q) - 1)))
    subset_index = q[2:end]

    @test symm(subset) == symm(q)
    @test subset.inds == q.inds
    @test subset.spaces == q.spaces
    @test all(subset.spaces[leg] !== q.spaces[leg] for leg in 1:length(q.spaces))
    @test _field_subset_equal(subset, TLArray(q, 2:Telum.nsectors(q)))
    @test _field_subset_equal(subset_getsub, subset)
    @test _field_subset_equal(subset_mask, subset)
    @test _field_subset_equal(subset_index, subset)

    removed_sector = Telum.sector_qlabel(q, 1, 1)
    @test any(qlabels == removed_sector for (qlabels, _) in subset.spaces[1])
    @test all(Telum.sector_qlabel(subset, i, 1) != removed_sector for i in 1:Telum.nsectors(subset))

    clone = TLArray(q, :)
    original_entry = Telum.sector_rmt(q, 1)[1]
    Telum.sector_rmt(clone, 1)[1] += one(eltype(Telum.sector_rmt(clone, 1)))
    @test Telum.sector_rmt(q, 1)[1] == original_entry

    original_space = q.spaces[1][1]
    clone.spaces[1][1] = (original_space[1], original_space[2] + 1)
    @test q.spaces[1][1] == original_space

    @test Telum.nsectors(TLArray(q, Int[])) == 0
    @test Telum.nsectors(q[BitVector(falses(Telum.nsectors(q)))]) == 0

    @test_throws BoundsError TLArray(q, 0)
    @test_throws BoundsError TLArray(q, [Telum.nsectors(q) + 1])
    @test_throws ArgumentError TLArray(q, [1, 1])
    @test_throws DimensionMismatch TLArray(q, trues(Telum.nsectors(q) + 1))
end
