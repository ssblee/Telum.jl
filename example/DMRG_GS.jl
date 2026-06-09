macro time_block(enabled, msg, ex)
    quote
        if $(esc(enabled))
            println($(esc(msg)))
            @time $(esc(ex))
        else
            $(esc(ex))
        end
    end
end

function eigs_GS(Hl, Hs, Hr, M; tol, nKrylov, time_blocks=true)
    #println("================================================")
    # solve the effective Hamiltonian eigenvalue problem by Lanczos method
    # return the ground state and energy
    As = Vector{AbstractTLArray}(undef, nKrylov)
    As[1] = M / norm(M)
    αs = zeros(nKrylov); βs = zeros(nKrylov-1)
    cnt = 0

    @time_block time_blocks "Lanczos iteration:" begin
        for i in 1:nKrylov
            #println("Lanczos iteration $i")
            Amul = Hl * As[i]
            for H in Hs Amul = Amul * H end
            Amul = Amul * Hr
            αs[i] = (As[i]' * Amul)[]
            cnt += 1

            #println("Orthogonalizing the Lanczos vector...")
            if i < nKrylov
                for _=1:2 
                    coeffs = [(As[j]' * Amul)[] for j in 1:i]
                    Amul = sum((Amul, (-coeffs[j] * As[j] for j in 1:i)...))
                end
                Anorm = norm(Amul)
                if Anorm < tol break end
                As[i+1] = Amul / Anorm
                βs[i] = Anorm
            end
        end
    end

    @time_block time_blocks "Diagonalizing the tridiagonal matrix..." begin
        Hkrylov = SymTridiagonal(αs[1:cnt], βs[1:cnt-1])
        _, V = eigen(Hkrylov); vec = V[:, 1]
        Anew = sum((vec[i] * As[i] for i in 1:cnt))
        #println("Getting final state tensor")
        Enew = Hl * Anew
        for H in Hs Enew = Enew * H end
        Enew = ((Enew * Hr) * Anew')[]
        @assert imag(Enew) < tol
    end
    return Anew, real(Enew)
end

function getHrl(MPO, MPS)
    N = length(MPS)
    Hrl = Vector{AbstractTLArray}(undef, N + 2)

    li = findleg(MPS[1]; itag="SLeft")
    left_id = getIdentity((MPS[1]', li); itag="SLeft")
    Hrl[1] = addSingleton(left_id; itag="OLeft", dir='-')

    li = findleg(MPS[end]; itag="SRight")
    right_id = getIdentity((MPS[end], li); itag="SRight")

    Hrl[end] = addSingleton(right_id; itag="ORight", dir='+')
    for i in 1:N Hrl[i+1] = MPS[i]' * lock(Hrl[i] * MPO[i] * MPS[i]; itag="SB,$i") end
    return Hrl
end

function DMRG_GS_1site!(MPS::Vector{<:TLArray{T1, 3}},
    MPO::Vector{<:TLArray{T2, 4}}, 
    Nkeep::Int, 
    Nsweep::Int; time_blocks=true) where {T1, T2}

    println("One-site DMRG...")
    tol = 1e-8; nKrylov = 5
    N = length(MPO); @assert length(MPS) == N
    Hrl = getHrl(MPO, MPS)
    MPS[N] = legflip(MPS[N]; itag="SRight")
    Hrl[end] = legflip(Hrl[end]; itag="SRight")
    E = 0.0

    for si in 1:Nsweep
        # right to left sweep
        println("DMRG right->left sweep $si")
        for i in N:-1:1
            println("Site $i")
            M, E = eigs_GS(Hrl[i], [MPO[i]], Hrl[i+2], MPS[i]; tol, nKrylov, time_blocks)
            target_tag = i == 1 ? "SLeft" : "SB,$(i-1)"
            if i > 1
                @time_block time_blocks "Time for svd: " begin
                    result = svd(M, "temp", target_tag; itag=target_tag)
                    U, S, Vd = result.U, result.S, result.Vd
                    MPS[i] = Vd; MPS[i-1] = (MPS[i-1] * U) * S
                end
            else MPS[i] = M end
            Hrl[i+1] = (Hrl[i+2] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=target_tag)
        end
        println("Energy: $E")

        # left to right sweep
        println("DMRG left->right sweep $si")
        for i in 1:N
            println("Site $i")
            M, E = eigs_GS(Hrl[i], [MPO[i]], Hrl[i+2], MPS[i]; tol, nKrylov, time_blocks)
            target_tag = i == N ? "SRight" : "SB,$i"
            if i < N
                @time_block time_blocks "Time for svd: " begin
                    result = svd(M, target_tag, "temp"; itag=target_tag, rev=true)
                    U, S, Vd = result.U, result.S, result.Vd
                    MPS[i] = U; MPS[i+1] = (MPS[i+1] * Vd) * S
                end
            else MPS[i] = M end
            Hrl[i+1] = (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=target_tag)
        end
        println("Energy: $E")
    end
    return MPS, E
end

function DMRG_GS_2site!(MPS::Vector{<:TLArray{T1, 3}},
    MPO::Vector{<:TLArray{T2, 4}}, 
    Nkeep::Int, 
    Nsweep::Int; time_blocks=true) where {T1, T2}

    println("Two-site DMRG...")
    tol = 1e-8; nKrylov = 5
    N = length(MPO); @assert length(MPS) == N
    Hrl = getHrl(MPO, MPS)
    E = 0.0

    for si in 1:Nsweep
        # right to left sweep
        println("DMRG right->left sweep $si")
        for i in N-1:-1:1
            if time_blocks println("Sites $i and $(i+1)") end
            M, E = eigs_GS(Hrl[i], [MPO[i], MPO[i+1]], Hrl[i+3], MPS[i] * MPS[i+1]; tol, nKrylov, time_blocks)

            @time_block time_blocks "Time for svd: " begin
                tags = i == 1 ? ("SLeft", "S,$i") : ("SB,$(i-1)", "S,$i")
                lids = [findleg(M; itag=t) for t in tags]
                result = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=Nkeep)
                U, S, Vd = result.U, result.S, result.Vd

                MPS[i] = removeitag(U * S, "right")
                MPS[i+1] = removeitag(Vd, "right")
                Hrl[i+2] = (Hrl[i+3] * MPS[i+1]) * MPO[i+1] * lock(MPS[i+1]'; itag="SB,$i")
            end
        end
        println("Energy: $E")

        # left to right sweep
        println("DMRG left->right sweep $si")
        for i in 1:N-1
            if time_blocks println("Sites $i and $(i+1)") end
            M, E = eigs_GS(Hrl[i], [MPO[i], MPO[i+1]], Hrl[i+3], MPS[i] * MPS[i+1]; tol, nKrylov, time_blocks)

            @time_block time_blocks "Time for svd: " begin
                tags = i == 1 ? ("SLeft", "S,$i") : ("SB,$(i-1)", "S,$i")
                lids = [findleg(M; itag=t) for t in tags]
                result = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=Nkeep)
                U, S, Vd = result.U, result.S, result.Vd

                MPS[i] = removeitag(U, "left")
                MPS[i+1] = removeitag(S * Vd, "left")
                Hrl[i+1] = (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag="SB,$i")
            end
        end
        println("Energy: $E")
    end
    return MPS, E
end
