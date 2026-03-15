get1jpair(q::QSpace{T, QD, N, RD}, leg::Int) where {T, QD, N, RD} = 
get1jpair(leginfo(q, leg))

function get1jpair(leginfo::leginfo{N}) where N
    rows1 = row{Float64, 2, N, 2+N}[]
    rows2 = row{Float64, 2, N, 2+N}[]

    symm, ind = leginfo.symm, leginfo.ind
    inds1 = (change_dir(ind), change_dir(green(ind)))
    inds2 = (green(ind), ind)
    dir1 = inds1[1].dir; dir2 = inds2[1].dir
    for (RMTd, qlabels) in leginfo.splist
        RMT1 = QTensor(reshape(Matrix{Float64}(I, RMTd, RMTd), RMTd, RMTd, (1 for _=1:N)...))
        RMT2 = QTensor(reshape(Matrix{Float64}(I, RMTd, RMTd), RMTd, RMTd, (1 for _=1:N)...))
        dual_qlabels = Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)

        cgrs1 = CGR{2}[]; cgrs2 = CGR{2}[]
        sign_fac = 1
        for n in 1:N
            spdim = dimension(symm[n], qlabels[n])
            cgr_qs1 = cgr_qs2 = sort((qlabels[n], dual_qlabels[n]))
            if qlabels[n] == dual_qlabels[n] cgp1 = cgp2 = (1, 2)
            elseif qlabels[n] < dual_qlabels[n] cgp1 = (1, 2); cgp2 = (2, 1) 
            else cgp1 = (2, 1); cgp2 = (1, 2) end
            if !isabelian(symm[n]) && qlabels[n] == dual_qlabels[n]
                S = symm[n]; NZ = nzops(S); zeroqlab = Tuple(0 for _=1:NZ)
                r = getNsave_Rsymbol(S, BigInt, qlabels[n], zeroqlab).rsym_mat
                @assert size(r) == (1, 1)
                sign_fac *= Int(sign(r[1]))
            end
            wmat1 = QTensor([sqrt(spdim);;]); wmat2 = QTensor([sqrt(spdim);;])
            legdir1 = dir1 == '+' ? (2, 0) : (0, 2)
            legdir2 = dir2 == '+' ? (2, 0) : (0, 2)

            push!(cgrs1, CGR(symm[n], cgr_qs1, wmat1, cgp1, legdir1))
            push!(cgrs2, CGR(symm[n], cgr_qs2, wmat2, cgp2, legdir2))
        end
        RMT2 *= sign_fac

        push!(rows1, row(Tuple(cgrs1), RMT1))
        push!(rows2, row(Tuple(cgrs2), RMT2))
    end

    # q1: leg 1 = original space, leg 2 = dual space
    # q2: leg 1 = dual space, leg 2 = original space
    ET = eltype(leginfo.splist)
    dual_splist = sort!(ET[(RMTd, Tuple(get_dualq(symm[n], qlabels[n]) for n in 1:N)) 
                           for (RMTd, qlabels) in leginfo.splist], by=x->x[2])
    spaces1 = (leginfo.splist, dual_splist)
    spaces2 = (dual_splist, leginfo.splist)

    q1 = QSpace(symm, rows1, inds1, spaces1)
    q2 = QSpace(symm, rows2, inds2, spaces2)
    return q1, q2
end