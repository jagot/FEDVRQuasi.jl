# * Diagonal matrices
DiagonalBlockDiagonal(A::AbstractMatrix, (rows,cols)::Tuple) =
    BandedBlockBandedMatrix(A, (rows,cols), (0,0), (0,0))

DiagonalBlockDiagonal(A::AbstractMatrix, rows) =
    DiagonalBlockDiagonal(A, (rows,rows))

function (B::FEDVR)(D::Diagonal)
    n = size(B,2)
    @assert size(D) == (n,n)
    all(order(B) .== 2) ? D : DiagonalBlockDiagonal(D, block_structure(B))
end

function (B::RestrictedQuasiArray{<:Any,2,<:FEDVR})(D::Diagonal)
    n = size(B,2)
    @assert size(D) == (n,n)
    all(order(B) .== 2) ? D : DiagonalBlockDiagonal(D, block_structure(B))
end

# * Dense operators

function Matrix(::UndefInitializer, B::Union{FEDVR{T},RestrictedQuasiArray{T,2,FEDVR{T}}}) where T
    if all(order(B) .== 2)
        n = size(B,2)
        dl = Vector{T}(undef, n-1)
        d = Vector{T}(undef, n)
        du = Vector{T}(undef, n-1)
        Tridiagonal(dl, d, du)
    else
        rows = block_structure(B)
        l,u = block_bandwidths(B,rows)

        BlockSkylineMatrix{T}(undef, rows, rows, (l,u))
    end
end

function set_block!(fun::Function, A::BlockSkylineMatrix{T}, B::FEDVR{T}, i, j) where T
    b = fun(i)
    o = B.order[i]
    @assert size(b,1) == size(b,2) == o

    s = 1+(i>1)
    e = size(b,1)-(i<nel(B))

    if o > 2
        j += i > 1
        A[Block(j,j)] .= b[s:e,s:e]
        if i < nel(B)
            A[Block(j+1,j+1)] .+= b[end,end]
            A[Block(j,j+1)] = b[s:e,end]
            A[Block(j+1,j)] = reshape(b[end,s:e], 1, e-s+1)
        end
        if i > 1
            A[Block(j-1,j-1)] .+= b[1,1]
            A[Block(j,j-1)] = b[s:e,1]
            A[Block(j-1,j)] = reshape(b[1,s:e], 1, e-s+1)
        end
        if i > 1 && i < nel(B)
            A[Block(j-1,j+1)] = b[1,end]
            A[Block(j+1,j-1)] = b[end,1]
        end
    else
        A[Block(j,j)] .+= b[1,1]
        A[Block(j+1,j)] .= b[2,1]
        A[Block(j,j+1)] .= b[1,2]
        A[Block(j+1,j+1)] .+= b[2,2]
    end

    j += 1
end

function set_blocks!(fun::Function, A::BlockSkylineMatrix{T}, B::FEDVR{T}) where T
    nel = length(B.order)

    A.data .= zero(T)

    j = 1
    for i in eachindex(B.order)
        j = set_block!(fun, A, B, i, j)
    end
    A
end

function set_blocks!(fun::Function, A::BlockSkylineMatrix{T}, B::RestrictedQuasiArray{T,2,FEDVR{T}}) where T
    B′,restriction = B.args
    nel = length(B′.order)

    A.data .= zero(T)

    a,b = restriction_extents(restriction)

    if nel == 1
        b₁ = fun(1)
        A[Block(1,1)] = b₁[1+a:end-b,1+a:end-b]
        return A
    end

    b₁ = fun(1)
    A[Block(1,1)] = b₁[1+a:end-1,1+a:end-1]
    A[Block(2,2)] = b₁[end,end]
    A[Block(1,2)] = b₁[1+a:end-1,end]
    A[Block(2,1)] = reshape(b₁[end,1+a:end-1], 1, :)

    j = 2
    for i in 2:nel-1
        j = set_block!(fun, A, B′, i, j)
    end

    b∞ = fun(nel)
    A[Block(j,j)] .+= b∞[1,1]
    A[Block(j+1,j)] = b∞[2:end-b,1]
    A[Block(j,j+1)] = reshape(b∞[1,2:end-b], 1, :)
    A[Block(j+1,j+1)] = b∞[2:end-b,2:end-b]

    A
end

# * Scalar operators

@simplify function *(Ac::QuasiAdjoint{<:Any,<:FEDVR},
                     D::QuasiDiagonal,
                     B::FEDVR)
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply functions on different grids"))

    Diagonal(getindex.(Ref(D.diag), B.x))
end

# A & B restricted
function materialize(M::Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix},
                                          <:QuasiAdjoint{<:Any,<:FEDVR{T}},
                                          <:QuasiDiagonal,
                                          <:FEDVR{T},
                                          <:RestrictionMatrix}}) where T
    restAc,Ac,D,B,restB = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))

    # This is mainly for type-stability; it would be trivial to
    # generate the proper banded matrix with one off-diagonal, from
    # the combination of two differently restricted bases, but we
    # would like to have Diagonal as the result if they are equal, and
    # this has higher priority. On the other hand, you typically only
    # compute scalar operators in the beginning of the calculation,
    # and thus type-instability is not a big problem, so this
    # behaviour may change in the future.
    restAc' == restB ||
        throw(ArgumentError("Non-equal restriction matrices not supported"))

    a,b = restriction_extents(restB)
    Diagonal(getindex.(Ref(D.diag), B.x[1+a:end-b]))
end
