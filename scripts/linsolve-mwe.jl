using Reactant
using SparseArrays, LinearAlgebra, KernelAbstractions

# Reactant.set_default_backend("cpu") # This line gets ignored for some reason, so we fall back to CUDA for now
using CUDA, Adapt

Anonsym = sprand(5,5,0.5)

# Symmetrize for simplicity
A = Anonsym'*Anonsym
x = ones(5)
b = zeros(5)
mul!(b, A, x)

struct GenericSparseMatrixCSR{Tv, Ti, IndexStorageType1 <: Union{AbstractVector{Ti}, Reactant.RArray{Reactant.TracedRNumber{Ti}, 1}}, IndexStorageType2 <: Union{AbstractVector{Ti}, Reactant.RArray{Reactant.TracedRNumber{Ti}, 1}}, ValueStorageType <: Union{AbstractVector{Tv}, Reactant.RArray{Reactant.TracedRNumber{Tv}, 1}}} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    # Not sure if we can get around this hack here
    rowptr::IndexStorageType1
    colval::IndexStorageType2
    nzval::ValueStorageType
end
function GenericSparseMatrixCSR(m,n,rowptr,colval,nzval)
    @info typeof(rowptr), typeof(colval)
    error("bla")
end
Adapt.@adapt_structure GenericSparseMatrixCSR

Base.size(A::GenericSparseMatrixCSR)              = (A.n,A.m)
Base.size(A::GenericSparseMatrixCSR,i)            = Base.size(A)[i]
Base.IndexStyle(::Type{<:GenericSparseMatrixCSR}) = IndexCartesian()

# SparseMatricesCSR.getrowptr(A::GenericSparseMatrixCSR) = SparseMatricesCSR.getrowptr(A)
# SparseMatricesCSR.getnzval(A::GenericSparseMatrixCSR)  = SparseMatricesCSR.getnzval(A)
# SparseMatricesCSR.getcolval(A::GenericSparseMatrixCSR) = SparseMatricesCSR.getcolval(A)

SparseArrays.issparse(A::GenericSparseMatrixCSR) = true
SparseArrays.nnz(A::GenericSparseMatrixCSR)      = length(A.nzval)
SparseArrays.nonzeros(A::GenericSparseMatrixCSR) = A.nzval

SparseArrays.nzrange(S::GenericSparseMatrixCSR, row::Integer) = S.rowptr[row]:S.rowptr[row+1]-1

Base.@propagate_inbounds function SparseArrays.getindex(A::GenericSparseMatrixCSR{T}, i0::Integer, i1::Integer) where T
    0.0
end
# Base.@propagate_inbounds function SparseArrays.getindex(A::GenericSparseMatrixCSR{T}, i0::Integer, i1::Integer) where T
#     getindex(A,i0,i1)
# end
# SparseArrays.getindex(A::GenericSparseMatrixCSR, ::Colon, ::Colon) = copy(A)
# SparseArrays.getindex(A::GenericSparseMatrixCSR, i::Int, ::Colon)  = getindex(A, i, 1:size(A, 2))
# SparseArrays.getindex(A::GenericSparseMatrixCSR, ::Colon, i::Int)  = getindex(A, 1:size(A, 1), i)


function spmv!(y::AbstractVector, A::GenericSparseMatrixCSR, x::AbstractVector)
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())
    backend = KernelAbstractions.get_backend(x)
    @assert backend == KernelAbstractions.get_backend(y)
    kernel! = spmv_kernel!(backend)
    kernel!(y, A, x, ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    return y
end


@kernel function spmv_kernel!(y::AbstractVector, A::GenericSparseMatrixCSR, x::AbstractVector)
    row = @index(Global, Linear) # using this triggers a scalar indexing error?
    if row ≤ A.m
        v = zero(eltype(y))
        for nz in nzrange(A, row)
            col = A.colval[nz]
            v += A.nzval[nz]*x[col]
        end
        y[row] = v
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::GenericSparseMatrixCSR, x::AbstractVector)
    return spmv!(y,A,x)
end


# Matrix is symmetric
@info "CPU"
A2 = GenericSparseMatrixCSR(
    5,5,
    A.colptr,
    A.rowval,
    A.nzval
)
mul!(b, A2, x) # Smoke test

@info "CUDA"
Acu = GenericSparseMatrixCSR(
    5,5,
    cu(A.colptr),
    cu(A.rowval),
    cu(A.nzval),
)
bcu = cu(b)
xcu = cu(x)
mul!(bcu, Acu, xcu)

# Ar = Reactant.to_rarray(A2)
Ar = GenericSparseMatrixCSR(
    5,5,
    Reactant.to_rarray(A2.rowptr),
    Reactant.to_rarray(A2.colval),
    Reactant.to_rarray(A2.nzval),
)
br = Reactant.to_rarray(b)
br .= 0.0
xr = Reactant.to_rarray(x)

@info "f1"
f1! = @compile mul!(br, Ar, xr) # Calls into a generic dense matrix-vector kernel
f1!(br, Ar, xr)

@info "f2"
f2! = @compile spmv!(br, Ar, xr) # Adaption failure
f2!(br, Ar, xr)

# using LinearSolve
