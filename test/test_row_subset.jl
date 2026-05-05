function _rows_equal(rows1, rows2)
    length(rows1) == length(rows2) || return false
    return all(zip(rows1, rows2)) do (r1, r2)
        r1.RMT.data == r2.RMT.data || return false
        length(r1.cgrs) == length(r2.cgrs) || return false
        all(zip(r1.cgrs, r2.cgrs)) do (c1, c2)
            symm(c1) == symm(c2) &&
            c1.qlabels == c2.qlabels &&
            c1.wmat.data == c2.wmat.data &&
            c1.cgp == c2.cgp &&
            c1.legdir == c2.legdir
        end
    end
end

function _row_sector(qs::TLArray, r, leg::Int)
    return Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:length(symm(qs)))
end

@testset "TLArray row subset selection" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q = getLocalSpace(option, ("lur", "lur", "op")).I
    @test length(q.rows) >= 2

    subset = TLArray(q, 2:length(q.rows))
    subset_getsub = getsub(q, 2:length(q.rows))
    subset_mask = TLArray(q, vcat(false, trues(length(q.rows) - 1)))
    subset_index = q[2:end]

    @test symm(subset) == symm(q)
    @test subset.inds == q.inds
    @test subset.spaces == q.spaces
    @test all(subset.spaces[leg] !== q.spaces[leg] for leg in 1:length(q.spaces))
    @test _rows_equal(subset.rows, q.rows[2:end])
    @test _rows_equal(subset_getsub.rows, subset.rows)
    @test _rows_equal(subset_mask.rows, subset.rows)
    @test _rows_equal(subset_index.rows, subset.rows)

    removed_sector = _row_sector(q, q.rows[1], 1)
    @test any(qlabels == removed_sector for (qlabels, _) in subset.spaces[1])
    @test all(_row_sector(subset, r, 1) != removed_sector for r in subset.rows)

    clone = TLArray(q, :)
    original_entry = q.rows[1].RMT.data[1]
    clone.rows[1].RMT.data[1] += one(eltype(clone.rows[1].RMT.data))
    @test q.rows[1].RMT.data[1] == original_entry

    original_space = q.spaces[1][1]
    clone.spaces[1][1] = (original_space[1], original_space[2] + 1)
    @test q.spaces[1][1] == original_space

    @test isempty(TLArray(q, Int[]).rows)
    @test isempty(q[BitVector(falses(length(q.rows)))].rows)

    @test_throws BoundsError TLArray(q, 0)
    @test_throws BoundsError TLArray(q, [length(q.rows) + 1])
    @test_throws ArgumentError TLArray(q, [1, 1])
    @test_throws DimensionMismatch TLArray(q, trues(length(q.rows) + 1))
end
