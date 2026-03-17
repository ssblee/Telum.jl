get1jtensor(q::QSpace{T, QD, N, RD}, leg::Int) where {T, QD, N, RD} =
get1jtensor(leginfo(q, leg))

function get1jtensor(leginfo::leginfo{N}) where N
    rows1 = row{Float64, 2, N, 2+N}[]

    symm, ind = leginfo.symm, leginfo.ind
    inds1 = (change_dir(ind), change_dir(green(ind)))
    dir1 = inds1[1].dir
    for (RMTd, qlabels) in leginfo.splist
        RMT1 = QTensor(reshape(Matrix{Float64}(I, RMTd, RMTd), RMTd, RMTd, (1 for _=1:N)...))
        dual_qlabels = Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)

        cgrs1 = CGR{2}[]
        for n in 1:N
            spdim = dimension(symm[n], qlabels[n])
            cgr_qs1 = sort((qlabels[n], dual_qlabels[n]))
            if qlabels[n] == dual_qlabels[n] cgp1 = (1, 2)
            elseif qlabels[n] < dual_qlabels[n] cgp1 = (1, 2)
            else cgp1 = (2, 1) end
            wmat1 = QTensor([sqrt(spdim);;])
            legdir1 = dir1 == '+' ? (2, 0) : (0, 2)

            push!(cgrs1, CGR(symm[n], cgr_qs1, wmat1, cgp1, legdir1))
        end

        push!(rows1, row(Tuple(cgrs1), RMT1))
    end

    # leg 1 = original space, leg 2 = dual space
    ET = eltype(leginfo.splist)
    dual_splist = sort!(ET[(RMTd, Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)) 
                           for (RMTd, qlabels) in leginfo.splist], by=x->x[2])
    spaces1 = (leginfo.splist, dual_splist)

    q1 = QSpace(symm, rows1, inds1, spaces1)
    return q1
end
