using SparseArrays, LinearAlgebra, KernelAbstractions
using CUDA

using Adapt

Anonsym = sprand(5,5,0.5)

# Symmetrize for simplicity
A = Anonsym'*Anonsym
x = ones(5)
b = zeros(5)
mul!(b, A, x)

struct KASparseMatrixCSR{Tv, Ti, IndexStorageType <: AbstractVector{Ti}, ValueStorageType <: AbstractVector{Tv}} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    rowptr::IndexStorageType
    colval::IndexStorageType
    nzval::ValueStorageType
end
Adapt.@adapt_structure KASparseMatrixCSR

Base.size(A::KASparseMatrixCSR)              = (A.n,A.m)
Base.size(A::KASparseMatrixCSR,i)            = Base.size(A)[i]
Base.IndexStyle(::Type{<:KASparseMatrixCSR}) = IndexCartesian()

SparseArrays.issparse(A::KASparseMatrixCSR) = true
SparseArrays.nnz(A::KASparseMatrixCSR)      = length(A.nzval)
SparseArrays.nonzeros(A::KASparseMatrixCSR) = A.nzval

SparseArrays.nzrange(S::KASparseMatrixCSR, row::Integer) = S.rowptr[row]:S.rowptr[row+1]-1

Base.@propagate_inbounds function SparseArrays.getindex(A::KASparseMatrixCSR{T}, i0::Integer, i1::Integer) where T
    0.0
end

function spmv!(y::AbstractVector, A::KASparseMatrixCSR, x::AbstractVector)
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())
    backend = KernelAbstractions.get_backend(x)
    @assert backend == KernelAbstractions.get_backend(y)
    kernel! = spmv_kernel!(backend)
    kernel!(y, A, x, ndrange = length(y))
    return y
end


@kernel function spmv_kernel!(y::AbstractVector, A::KASparseMatrixCSR, x::AbstractVector)
    row = @index(Global)
    @inbounds if row ≤ A.m
        v = zero(eltype(y))
        for nz in nzrange(A, row)
            col = A.colval[nz]
            v += A.nzval[nz]*x[col]
        end
        y[row] = v
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::KASparseMatrixCSR, x::AbstractVector)
    return spmv!(y,A,x)
end


# Matrix is symmetric
@info "CPU"
A2 = KASparseMatrixCSR(
    5,5,
    A.colptr,
    A.rowval,
    A.nzval
)
mul!(b, A2, x) # Smoke test

@info "CUDA"
Acu = KASparseMatrixCSR(
    5,5,
    cu(A.colptr),
    cu(A.rowval),
    cu(A.nzval),
)
bcu = cu(b)
xcu = cu(x)
mul!(bcu, Acu, xcu) # Fails

# Works
backend = KernelAbstractions.get_backend(xcu)
directkernel! = spmv_kernel!(backend)
directkernel!(bcu, Acu, xcu, ndrange = length(bcu))
