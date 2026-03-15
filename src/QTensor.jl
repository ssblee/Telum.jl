"""
    QTensor{T, N, A}

An arbitrary dimensional array type that supports both CPU and GPU.
Wraps an underlying AbstractArray `A` of element type `T` and dimension `N`.
"""
struct QTensor{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end

# Basic constructors
QTensor{T}(dims::Int...) where {T} = QTensor(zeros(T, dims...))
QTensor{T}(dims::NTuple{N, Int}) where {T, N} = QTensor(zeros(T, dims))

# Basic array interface
Base.size(t::QTensor) = size(t.data)
Base.size(t::QTensor, d::Int) = size(t.data, d)
Base.ndims(::QTensor{T, N, A}) where {T, N, A} = N
Base.eltype(::QTensor{T, N, A}) where {T, N, A} = T

Base.getindex(t::QTensor, inds...) = getindex(t.data, inds...)
Base.setindex!(t::QTensor, v, inds...) = setindex!(t.data, v, inds...)

Base.similar(t::QTensor{T, N, A}) where {T, N, A} = QTensor(similar(t.data))
Base.similar(t::QTensor, ::Type{S}) where {S} = QTensor(similar(t.data, S))
Base.similar(t::QTensor, dims::Dims) = QTensor(similar(t.data, dims))
Base.similar(t::QTensor, ::Type{S}, dims::Dims) where {S} = QTensor(similar(t.data, S, dims))

Base.:*(t::QTensor, fac::Number) = QTensor(t.data * fac)
Base.:*(fac::Number, t::QTensor) = QTensor(fac * t.data)
Base.:/(t::QTensor, fac::Number) = QTensor(t.data / fac)

# Support for Adapt.jl (for GPU compatibility)
# If you use CUDA.jl or AMDGPU.jl, this allows `cu(qtensor)` or `roc(qtensor)`
# to automatically move the underlying data to the GPU.
# using Adapt
# Adapt.adapt_structure(to, t::QTensor) = QTensor(Adapt.adapt(to, t.data))
