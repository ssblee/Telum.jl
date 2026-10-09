module BenchmarkDMRG

using LinearAlgebra
using Telum

macro time_block(enabled, msg, ex)
    @info "$msg started"
    quote
        if $(esc(enabled))
            println($(esc(msg)))
            @time $(esc(ex))
        else
            $(esc(ex))
        end
    end
end

# Lanczos algorithm for the local DMRG effective problem.
function eigs_GS(Hl, Hs, Hr, M; tol, nKrylov, time_blocks=true)
    As = Vector{TLArray}(undef, nKrylov)
    As[1] = M / norm(M)
    αs = zeros(nKrylov)
    βs = zeros(nKrylov - 1)
    cnt = 0

    @time_block time_blocks "Lanczos iteration:" begin
        for i in 1:nKrylov
            Amul = @lazy begin
                tmp = Hl * As[i]
                for H in Hs tmp = tmp * H end
                tmp * Hr
            end
            αs[i] = (As[i]' * Amul)[]
            cnt += 1

            if i < nKrylov
                for _ in 1:2
                    coeffs = [(As[j]' * Amul)[] for j in 1:i]
                    Amul = sum([Amul, (-coeffs[j] * As[j] for j in 1:i)...])
                end
                Anorm = norm(Amul)
                Anorm < tol && break
                As[i + 1] = Amul / Anorm
                βs[i] = Anorm
            end
        end
    end

    @time_block time_blocks "Diagonalizing the tridiagonal matrix..." begin
        Hkrylov = SymTridiagonal(αs[1:cnt], βs[1:cnt - 1])
        _, V = eigen(Hkrylov)
        vec = V[:, 1]
        Anew = sum([vec[i] * As[i] for i in 1:cnt])
        Enew = @lazy begin
            tmp = Hl * Anew
            for H in Hs
                tmp = tmp * H
            end
            ((tmp * Hr) * Anew')[]
        end
        @assert imag(Enew) < tol
    end
    return Anew, real(Enew)
end

function getHrl(MPO, MPS)
    N = length(MPS)
    Hrl = Vector{TLArray}(undef, N + 2)

    li = findleg(MPS[1]; itag="SLeft")
    left_id = getIdentity((MPS[1]', li); itag="SLeft")
    Hrl[1] = addSingleton(left_id; itag="OLeft", dir='-')

    li = findleg(MPS[end]; itag="SRight")
    right_id = getIdentity((MPS[end], li); itag="SRight")

    Hrl[end] = addSingleton(right_id; itag="ORight", dir='+')
    for i in 1:N
        Hrl[i + 1] = MPS[i]' * lock(Hrl[i] * MPO[i] * MPS[i]; itag="SB,$i")
    end
    return Hrl
end

# One-site sweep is kept for compatibility with older experiments; the required
# validation runner below uses the two-site sweep.
function DMRG_GS_1site!(MPS::Vector{<:TLArray{T1, 3}},
                        MPO::Vector{<:TLArray{T2, 4}},
                        Nkeep::Int,
                        Nsweep::Int; time_blocks=true) where {T1, T2}

    println("One-site DMRG...")
    tol = 1e-8
    nKrylov = 5
    N = length(MPO)
    @assert length(MPS) == N
    Hrl = getHrl(MPO, MPS)
    MPS[N] = legflip(MPS[N]; itag="SRight")
    Hrl[end] = legflip(Hrl[end]; itag="SRight")
    E = 0.0

    for si in 1:Nsweep
        println("DMRG right->left sweep $si")
        for i in N:-1:1
            println("Site $i")
            M, E = eigs_GS(Hrl[i], [MPO[i]], Hrl[i + 2], MPS[i]; tol, nKrylov, time_blocks)
            target_tag = i == 1 ? "SLeft" : "SB,$(i - 1)"
            if i > 1
                @time_block time_blocks "Time for svd: " begin
                    result = svd(M, "temp", target_tag; itag=target_tag)
                    U, S, Vd = result.U, result.S, result.Vd
                    MPS[i] = Vd
                    MPS[i - 1] = (MPS[i - 1] * U) * S
                end
            else
                MPS[i] = M
            end
            Hrl[i + 1] = @lazy (Hrl[i + 2] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=target_tag)
        end
        println("Energy: $E")

        println("DMRG left->right sweep $si")
        for i in 1:N
            println("Site $i")
            M, E = eigs_GS(Hrl[i], [MPO[i]], Hrl[i + 2], MPS[i]; tol, nKrylov, time_blocks)
            target_tag = i == N ? "SRight" : "SB,$i"
            if i < N
                @time_block time_blocks "Time for svd: " begin
                    result = svd(M, target_tag, "temp"; itag=target_tag, rev=true)
                    U, S, Vd = result.U, result.S, result.Vd
                    MPS[i] = U
                    MPS[i + 1] = (MPS[i + 1] * Vd) * S
                end
            else
                MPS[i] = M
            end
            Hrl[i + 1] = @lazy (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=target_tag)
        end
        println("Energy: $E")
    end
    return MPS, E
end

function DMRG_GS_2site!(MPS::Vector{<:TLArray{T1, 3}},
                        MPO::Vector{<:TLArray{T2, 4}},
                        Nkeep::Union{Int, AbstractVector{<:Integer}},
                        Nsweep::Int; time_blocks=true, tol=1e-8, nKrylov=5) where {T1, T2}

    println("Two-site DMRG...")
    N = length(MPO)
    @assert length(MPS) == N
    if Nkeep isa AbstractVector
        length(Nkeep) == Nsweep || throw(ArgumentError(
            "Nkeep schedule must have one entry per sweep ($Nsweep), got $(length(Nkeep))"))
        all(>(0), Nkeep) || throw(ArgumentError("all Nkeep schedule entries must be positive"))
    else
        Nkeep > 0 || throw(ArgumentError("Nkeep must be positive"))
    end
    Hrl = getHrl(MPO, MPS)
    E = 0.0

    for si in 1:Nsweep
        nkeep = Nkeep isa Integer ? Nkeep : Nkeep[si]
        println("DMRG right->left sweep $si")
        for i in N - 1:-1:1
            time_blocks && println("Sites $i and $(i + 1)")
            M, E = eigs_GS(Hrl[i], [MPO[i], MPO[i + 1]], Hrl[i + 3], MPS[i] * MPS[i + 1];
                           tol, nKrylov, time_blocks)

            @time_block time_blocks "Time for svd: " begin
                tags = i == 1 ? ("SLeft", "S,$i") : ("SB,$(i - 1)", "S,$i")
                lids = [findleg(M; itag=t) for t in tags]
                result = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=nkeep)
                U, S, Vd = result.U, result.S, result.Vd

                MPS[i] = removeitag(U * S, "right")
                MPS[i + 1] = removeitag(Vd, "right")
                Hrl[i + 2] = @lazy (Hrl[i + 3] * MPS[i + 1]) * MPO[i + 1] * lock(MPS[i + 1]'; itag="SB,$i")
            end
        end
        println("Energy: $E")

        println("DMRG left->right sweep $si")
        for i in 1:N - 1
            time_blocks && println("Sites $i and $(i + 1)")
            M, E = eigs_GS(Hrl[i], [MPO[i], MPO[i + 1]], Hrl[i + 3], MPS[i] * MPS[i + 1];
                           tol, nKrylov, time_blocks)

            @time_block time_blocks "Time for svd: " begin
                tags = i == 1 ? ("SLeft", "S,$i") : ("SB,$(i - 1)", "S,$i")
                lids = [findleg(M; itag=t) for t in tags]
                result = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=nkeep)
                U, S, Vd = result.U, result.S, result.Vd

                MPS[i] = removeitag(U, "left")
                MPS[i + 1] = removeitag(S * Vd, "left")
                Hrl[i + 1] = @lazy (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag="SB,$i")
            end
        end
        println("Energy: $E")
    end
    return MPS, E
end

# Initial MPS construction by iterative diagonalization of left blocks.
function init_MPS(MPO::Vector{<:TLArray}, Nkeep::Int, Nkeep_last::Int=1;
                  tol=0.0, target_sector=nothing)
    N = length(MPO)
    MPS = Vector{TLArray{Float64, 3}}(undef, N)
    zq = zero_qlabels(MPO[1])
    Aprev = getvac(MPO[1], ("SLeft", "SLeft"))
    Hprev = addSingleton(Aprev; itag="OLeft", dir='-')
    E, sp = nothing, nothing

    for i in 1:N
        Hmat = @lazy begin
            li = findleg(Aprev; itag=i == 1 ? "SLeft" : "SB", dir='-')
            Anow = getIdentity((Aprev, li), (MPO[i], 2); itag="SL,$i")
            Hnow = Anow' * lock(Anow * Hprev * MPO[i]; itag="SL,$i")

            bli = findleg(Hnow; itag=i == N ? "ORight" : "OB,$i")
            deleteSingleton(getsub(Hnow, bli, x -> x == zq ? 1 : nothing), bli)
        end
        if i == N && target_sector !== nothing
            Hmat = getsub(Hmat, (1,2), x -> x == target_sector ? Colon() : nothing)
            isempty(Telum.spaces(Hmat)[1]) && throw(ArgumentError(
                "requested MPS target sector $target_sector is not present"))
        end
        e = eigen((Hmat + Hmat') / 2; hermitian=true)

        ek, _ = discard_eigen(e, i == N ? Nkeep_last : Nkeep,
                              i == N ? "SRight" : "SB,$i", "SD,$i"; tol)
        if i == N && target_sector !== nothing
            alli = findleg(Anow; itag="SL,$i")
            Anow = getsub(Anow, alli, x -> x == target_sector ? Colon() : nothing)
        end
        MPS[i] = Aprev = Anow * ek.V
        if i < N
            Hprev = lock(ek.V * Hnow; itag="SB") * ek.V'
        else
            E, sp = ek.eig_list[1][1], ek.eig_list[1][3]
        end
    end
    return MPS, E, sp
end

function do_dmrg_result(MPO, Nkeep_init=50, Nkeep_DMRG=50, Nsweep=4;
                        time_blocks=true, lanczos_tol=1e-8, nKrylov=5)
    MPS, Einit, sp = init_MPS(MPO, Nkeep_init; tol=0.0)
    println("Initial energy: $Einit")
    MPS, E = DMRG_GS_2site!(MPS, MPO, Nkeep_DMRG, Nsweep;
                             time_blocks, tol=lanczos_tol, nKrylov)
    return MPS, E
end

do_dmrg(MPO, Nkeep_init=50, Nkeep_DMRG=50, Nsweep=4;
        time_blocks=true, lanczos_tol=1e-8, nKrylov=5) =
    first(do_dmrg_result(MPO, Nkeep_init, Nkeep_DMRG, Nsweep;
                         time_blocks, lanczos_tol, nKrylov))


export eigs_GS, getHrl, DMRG_GS_1site!, DMRG_GS_2site!
export init_MPS, do_dmrg_result, do_dmrg

end # module BenchmarkDMRG
