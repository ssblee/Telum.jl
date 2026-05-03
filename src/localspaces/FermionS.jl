# N: The number of channels
function FermionS_basicops(N::Int)
    @assert N > 0 "Number of channels must be positive"

    # Spin lowering operator for single channel
    ss = f4down' * f4up

    # Annihilation operations, dim: (channel, spin)
    # FF[1, n]: spin-up annihilation operator for channel n
    # FF[2, n]: spin-down annihilation operator for channel n
    FF = Matrix{SparseMatrixCSC{Int}}(undef, 2, N)
    NN = Matrix{SparseMatrixCSC{Int}}(undef, 2, N)
    # SS[n]: spin lowering operator for channel n
    SS = Vector{SparseMatrixCSC{Int}}(undef, N)
    # CC[n]: charge lowering operator for channel n
    CC = Vector{SparseMatrixCSC{Int}}(undef, N)
    
    # Total number of fermionic modes: 2 spins × N channels
    total_modes = 2 * N

    for i in 1:total_modes
        mats = vcat([z2 for _=1:i-1], [f2], [I2 for _=i+1:total_modes])
        FF[i] = reduce(⊗, mats)
        NN[i] = FF[i]' * FF[i]
    end
    
    for i in 1:N
        mats = vcat([I2 for _=1:2*(i-1)], [ss], [I2 for _=1:2*(N-i)])
        SS[i] = reduce(⊗, mats)
        @assert SS[i] == FF[2, i]' * FF[1, i]
    end

    for i in 1:N
        mats = vcat([I2 for _=1:2*(i-1)], [f4down * f4up], [I2 for _=1:2*(N-i)])
        CC[i] = reduce(⊗, mats)
        @assert CC[i] == FF[2, i] * FF[1, i]
    end

    SZ = Vector{SparseMatrixCSC{Int}}(undef, N)
    for i in 1:N
        mats = vcat([I2 for _=1:2*(i-1)], [sz4], [I2 for _=1:2*(N-i)])
        SZ[i] = reduce(⊗, mats)
    end 

    # Channel lowering operators for SU{N}: N-1 operators
    # chan_l[i] connects channel i to channel i+1
    chan_l = Vector{SparseMatrixCSC{Int}}(undef, N-1)
    for i in 1:(N-1)
        chan_l[i] = FF[1, i+1]'*FF[1, i] + FF[2, i+1]'*FF[2, i]
    end
    
    # Channel z-operators for SU{N}: N-1 operators following Cartan subalgebra
    # Using standard Dynkin basis: H_i = E_{i,i} - E_{i+1,i+1}
    chan_z = Vector{Vector{Int}}(undef, N-1)
    for i in 1:(N-1)
        # H_i: occupation of channel i minus occupation of channel i+1
        chan_z[i] = Int.(diag(sum([NN[1, j] + NN[2, j] for j=1:i]) 
        - i * NN[1, i+1] - i * NN[2, i+1]))
    end

    spin_z = Int.(diag(sum(SZ)))
    spin_lowering = sum(SS)

    # Verify commutation relations for SU{N} algebra
    @assert comm(spin_lowering, diagm(spin_z)) == 2 * spin_lowering
    for i in 1:(N-1)
        # [H_i, E_i] = 2*E_i (lowering operator i raised by H_i by 2)
        @assert comm(chan_l[i], diagm(chan_z[i])) == (i+1) * chan_l[i]
        
        # [H_i, E_{i-1}] = -E_{i-1} (lowering operator i-1 lowered by H_i by 1)
        if i > 1
            @assert comm(chan_l[i-1], diagm(chan_z[i])) == 0 * chan_l[i-1]
            @assert comm(chan_l[i], diagm(chan_z[i-1])) == -(i-1) * chan_l[i]
        end

        
        # [H_i, E_j] = 0 for |i-j| > 1
        for j in 1:(N-1)
            if abs(i - j) > 1
                @assert comm(chan_l[j], diagm(chan_z[i])) == 0 * chan_l[j]
            end
        end
    end

    return (; FF, NN, SS, CC, SZ, chan_l, chan_z)
end

# For now, we only implement U1 charge relative to half-filling
# TODO: Z_N charge symmetry, relative to other fillings, etc.
function charge_weights(opts::FermionSOptions, basic_ops)
    nchan = opts.nchannels
    charge = Int.(diag(sum(basic_ops.NN)) .- nchan)
    return [(Int(i),) for i in charge]
end

function charge_lowering(opts::FermionSOptions, basic_ops)
    if opts.charge_symm == SU{2} return [sum(basic_ops.CC)]
    elseif opts.charge_symm in [U1, nothing] return Matrix{Int}[]
    else error("Unsupported charge symmetry") end
end

# This covers U1 and SU(2) spin symmetries since the weights are 
# just the total spin-z component
function spin_weights(opts::FermionSOptions, basic_ops)
    spin_z = Int.(diag(sum(basic_ops.SZ)))
    return [(Int(i),) for i in spin_z]
end

# For U1 spin symmetry, there is no lowering operator.
# For SU(2) spin symmetry, the lowering operator is 
# the sum of all single-channel spin lowering operators.
function spin_lowering(opts::FermionSOptions, basic_ops)
    @assert opts.spin_symm in [SU{2}, U1, nothing]
    if opts.spin_symm == SU{2} return [-sum(basic_ops.SS)]
    elseif opts.spin_symm == U1 return Matrix{Int}[]
    else error("Unsupported spin symmetry") end
end

# For now, we only implement SU(N) channel symmetry
function chan_weights(opts::FermionSOptions, basic_ops)
    N = opts.nchannels
    # TODO: generalize to another channel symmetry
    @assert opts.channel_symm == SU{N}
    return collect(zip(basic_ops.chan_z...))
end

function chan_lowering(opts::FermionSOptions, basic_ops)
    N = opts.nchannels
    # TODO: generalize to another channel symmetry
    @assert opts.channel_symm == SU{N}
    return basic_ops.chan_l
end

function getSymmetryInfo(opts::FermionSOptions)
    # For N-channel SU(2)-spin case, local Hilbert space is:
    #   |0>, |↑>, |↓>, |↑↓>
    # This should be changed for more general options

    N = opts.nchannels

    basic_ops = FermionS_basicops(N)
    symms = Vector{Type{<:Symmetry}}()
    weights = Vector{<:Tuple{Vararg{Int}}}[]
    lowering_ops = Vector{<:AbstractMatrix{Int}}[]

    # If charge symmetry is present, add it 
    if opts.charge_symm !== nothing
        push!(symms, opts.charge_symm)
        push!(weights, charge_weights(opts, basic_ops))
        push!(lowering_ops, charge_lowering(opts, basic_ops))
    end

    # If spin symmetry is present, add it
    if opts.spin_symm !== nothing
        push!(symms, opts.spin_symm)
        push!(weights, spin_weights(opts, basic_ops))
        push!(lowering_ops, spin_lowering(opts, basic_ops))
    end

    # If channel symmetry is present, add it 
    if opts.channel_symm !== nothing
        push!(symms, opts.channel_symm)
        push!(weights, chan_weights(opts, basic_ops))
        push!(lowering_ops, chan_lowering(opts, basic_ops))
    end

    totalN = diag(sum(basic_ops.NN[i] for i in 1:2*N))
    mwirops = Dict{Symbol, Tuple{AbstractMatrix{Int}, Float64}}()
    if opts.spin_symm == SU{2}
        mwirops[:S] = (sum(basic_ops.SS[i]' for i in 1:N), -1/sqrt(2))
    elseif opts.spin_symm == U1 || opts.spin_symm === nothing
        mwirops[:Sp] = (sum(basic_ops.SS[i]' for i in 1:N), -1/sqrt(2))
        mwirops[:Sz] = (sum(basic_ops.SZ[i] for i in 1:N), 1/2)
        mwirops[:Sm] = (sum(basic_ops.SS[i] for i in 1:N), 1/sqrt(2))
    else error("Unsupported spin symmetry") end
    if opts.spin_symm == SU{2}
        if opts.charge_symm == SU{2}
            mwirops[:F] = (sum(basic_ops.FF[1, i]' for i in 1:N), 1.0)
        elseif opts.charge_symm == U1 || opts.charge_symm === nothing
            mwirops[:F] = (basic_ops.FF[2, N], 1.0)
        else error("Not implemented yet") end
    else
        if opts.charge_symm == SU{2}
            mwirops[:Fu] = (sum(basic_ops.FF[2, i]' for i in 1:N), 1.0)
            mwirops[:Fd] = (sum(basic_ops.FF[1, i]' for i in 1:N), 1.0)
        elseif opts.charge_symm == U1 || opts.charge_symm === nothing
            mwirops[:Fu] = (basic_ops.FF[1, N], 1.0)
            mwirops[:Fd] = (basic_ops.FF[2, N], 1.0)
        else error("Not implemented yet") end
    end
    mwirops[:Z] = (diagm([i%2==0 ? 1 : -1 for i in totalN]), 1.0)
    mwirops[:I] = (sparse(I, 4^N, 4^N), 1.0)

    return Tuple(symms), Tuple(weights), Tuple(lowering_ops), mwirops
end