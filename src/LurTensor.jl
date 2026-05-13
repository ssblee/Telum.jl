"""
    LurTensor{T, N, A}

An arbitrary dimensional array type that supports both CPU and GPU.
Wraps an underlying AbstractArray `A` of element type `T` and dimension `N`.
"""
struct LurTensor{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end

# Basic constructors
LurTensor{T}(dims::Int...) where {T} = LurTensor(zeros(T, dims...))
LurTensor{T}(dims::NTuple{N, Int}) where {T, N} = LurTensor(zeros(T, dims))

_lurtensor_parent(t::LurTensor) = t.data
_lurtensor_parent(t::AbstractArray) = t

function _lurtensor_wrap_like(::LurTensor, arr::AbstractArray)
    return arr isa LurTensor ? arr : LurTensor(arr)
end

_lurtensor_wrap_like(::AbstractArray, arr::AbstractArray) = arr

# Basic array interface
Base.size(t::LurTensor{T, N}) where {T, N} = size(t.data)::NTuple{N, Int}
Base.size(t::LurTensor, d::Int) = size(t.data, d)::Int
Base.ndims(::LurTensor{T, N, A}) where {T, N, A} = N
Base.eltype(::LurTensor{T, N, A}) where {T, N, A} = T

Base.getindex(t::LurTensor, inds...) = getindex(t.data, inds...)
Base.setindex!(t::LurTensor, v, inds...) = setindex!(t.data, v, inds...)
Base.copy(t::LurTensor) = LurTensor(copy(t.data))

Base.similar(t::LurTensor{T, N, A}) where {T, N, A} = LurTensor(similar(t.data))
Base.similar(t::LurTensor, ::Type{S}) where {S} = LurTensor(similar(t.data, S))
Base.similar(t::LurTensor, dims::Dims) = LurTensor(similar(t.data, dims))
Base.similar(t::LurTensor, ::Type{S}, dims::Dims) where {S} = LurTensor(similar(t.data, S, dims))

Base.:*(t::LurTensor, fac::Number) = LurTensor(t.data * fac)
Base.:*(fac::Number, t::LurTensor) = LurTensor(fac * t.data)
Base.:/(t::LurTensor, fac::Number) = LurTensor(t.data / fac)
Base.:+(t1::LurTensor, t2::LurTensor) = LurTensor(t1.data + t2.data)
Base.:-(t1::LurTensor, t2::LurTensor) = LurTensor(t1.data - t2.data)

# Support for Adapt.jl (for GPU compatibility)
# If you use CUDA.jl or AMDGPU.jl, this allows `cu(lurtensor)` or `roc(lurtensor)`
# to automatically move the underlying data to the GPU.
# using Adapt
# Adapt.adapt_structure(to, t::LurTensor) = LurTensor(Adapt.adapt(to, t.data))
