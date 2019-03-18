module FEDVRQuasi

import Base: axes, size, ==, getindex, checkbounds, copyto!, similar, diff, show
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: ℵ₁
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray, Inclusion, ApplyQuasiArray

using BandedMatrices

using IntervalSets

using LazyArrays
import LazyArrays: ⋆
using FillArrays

using LinearAlgebra
import LinearAlgebra: Matrix, dot

using FastGaussQuadrature, BlockBandedMatrices

using Printf

# * Auxilliary type definitions for restricted bases

const RestrictionMatrix = BandedMatrix{<:Int, <:FillArrays.Ones}

const RestrictedBasis{B<:AbstractQuasiMatrix} = Mul{<:Any,<:Tuple{B, <:RestrictionMatrix}}
const AdjointRestrictedBasis{B<:AbstractQuasiMatrix} = Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix}, <:QuasiAdjoint{<:Any,B}}}

const RestrictedQuasiArray{T,N,B<:AbstractQuasiMatrix} = MulQuasiArray{T,N,<:RestrictedBasis{B}}
const AdjointRestrictedQuasiArray{T,N,B<:AbstractQuasiMatrix} = MulQuasiArray{T,N,<:AdjointRestrictedBasis{B}}

const BasisOrRestricted{B<:AbstractQuasiMatrix} = Union{B,RestrictedBasis{<:B},<:RestrictedQuasiArray{<:Any,<:Any,<:B}}
const AdjointBasisOrRestricted{B<:AbstractQuasiMatrix} = Union{<:QuasiAdjoint{<:Any,B},AdjointRestrictedBasis{<:B},<:AdjointRestrictedQuasiArray{<:Any,<:Any,<:B}}

unrestricted_basis(R::AbstractQuasiMatrix) = R
unrestricted_basis(R::RestrictedBasis) = first(R.args)
unrestricted_basis(R::RestrictedQuasiArray) = unrestricted_basis(R.applied)

# * Gauß–Lobatto grid

# https://github.com/JuliaLang/julia/pull/18777
lerp(a::T,b::T,t) where T = T(fma(t, b, fma(-t, a, a)))
lerp(a::R,b::R,t::C) where {R<:Real,C<:Complex} = lerp(a,b,real(t)) + im*lerp(a,b,imag(t))
lerp(a::C,b::C,t::R) where {R<:Real,C<:Complex} = lerp(real(a),real(b),t) + im*lerp(imag(a),imag(b),(t))

function element_grid(order, a::T, b::T, c::T=zero(T), eiϕ=one(T)) where T
    x,w = gausslobatto(order)
    c .+ lerp.(Ref(eiϕ*(a-c)), Ref(eiϕ*(b-c)), (x .+ 1)/2),(b-a)*w/2
end

# * Basis construction

struct FEDVR{T,R<:Real,O<:AbstractVector} <: AbstractQuasiMatrix{T}
    t::AbstractVector{R}
    order::O
    i₀::Integer
    t₀::R
    eiϕ::T
    x::Vector{T}
    wⁱ::Vector{Vector{R}}
    n::Vector{T}
    elems::Vector{UnitRange}
    function FEDVR(t::AbstractVector{R}, order::O; t₀::R=zero(R), ϕ::R=zero(R)) where {R<:Real,O<:AbstractVector}
        @assert length(order) == length(t)-1
        @assert all(order .> 1)

        i₀,eiϕ,T = if ϕ ≠ zero(R)
            findfirst(tt -> tt ≥ t₀, t),exp(im*ϕ),complex(R)
        else
            1,one(R),R
        end
        i₀ === nothing && throw(ArgumentError("Complex scaling starting point outside grid $(t)"))
        t₀ = t[i₀]

        x = Vector{Vector{T}}()
        w = Vector{Vector{T}}()
        for i in eachindex(order)
            xw = element_grid(order[i], t[i], t[i+1], t₀, i ≥ i₀ ? eiϕ : one(T))
            push!(x, xw[1])
            push!(w, xw[2])
        end

        rot = (i,v) -> i ≥ i₀ ? eiϕ*v : v
        n = [one(T) ./ .√(rot(i,wⁱ)) for (i,wⁱ) in enumerate(w)]
        for i in 1:length(order)-1
            n[i][end] = n[i+1][1] = one(T) ./ √(rot(i,w[i][end]) + rot(i+1,w[i+1][1]))
        end

        X = vcat(x[1], [x[i][2:end] for i in 2:length(order)]...)
        N = vcat(n[1], [n[i][2:end] for i in 2:length(order)]...)

        elems = [1:order[1]]

        l = order[1]
        for i = 2:length(order)
            l′ = l+order[i]-1
            push!(elems, l:l′)
            l = l′
        end

        new{T,R,O}(t,order,i₀,t₀,eiϕ,X,[w...],N,elems)
    end
end

FEDVR(t::AbstractVector{T},order::Integer; kwargs...) where T =
    FEDVR(t, Fill(order,length(t)-1); kwargs...)

const FEDVROrRestricted{T} = BasisOrRestricted{<:FEDVR{T}}
const AdjointFEDVROrRestricted{T} = AdjointBasisOrRestricted{<:FEDVR{T}}

# * Properties

axes(B::FEDVR) = (Inclusion(first(B.t)..last(B.t)), Base.OneTo(length(B.x)))
size(B::FEDVR) = (ℵ₁, length(B.x))
size(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) = (ℵ₁, length(B.applied.args[2].data))
==(A::FEDVR,B::FEDVR) = A.t == B.t && A.order == B.order
==(A::FEDVROrRestricted,B::FEDVROrRestricted) = unrestricted_basis(A) == unrestricted_basis(B)

order(B::FEDVR) = B.order
# This assumes that the restriction matrices do not remove blocks
order(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) = order(B.applied.args[1])

nel(B::FEDVR) = length(order(B))
element_boundaries(B::FEDVR) = vcat(1,1 .+ cumsum(B.order .- 1))

complex_rotate(x,B::FEDVR{T}) where {T<:Real} = x
complex_rotate(x,B::FEDVR{T}) where {T<:Complex} = x < B.t₀ ? x : B.t₀ + (x-B.t₀)*B.eiϕ

macro elem(B,v,i)
    :(@view($(esc(B)).$v[$(esc(B)).elems[$(esc(i))]]))
end

function show(io::IO, B::FEDVR{T}) where T
    write(io, "FEDVR{$(T)} basis with $(nel(B)) elements on $(axes(B,1).domain)")
    if T <: Complex
        rot = @printf(io, " with %s @ %.2f°", B.t₀ <= first(B.t) ? "ICS" : "ECS", rad2deg(angle(B.eiϕ)))
        B.t₀ > first(B.t) && @printf(io, " starting at %.2g", B.t₀)
    end
end

function show(io::IO, B::RestrictedQuasiArray{T,2,FEDVR{T}}) where T
    B′,restriction = B.applied.args
    a,b = restriction_extents(restriction)
    N = length(B′.x)
    show(io, B′)
    write(io, ", restricted to basis functions $(1+a)..$(N-b) $(a>0 || b>0 ? "⊂" : "⊆") 1..$(N)")
end

function block_structure(B::FEDVR)
    if length(B.order) > 1
        bs = o -> o > 2 ? [o-2,1] : [1]
        Vector{Int}(vcat(B.order[1]-1,1,
                         vcat([bs(B.order[i]) for i = 2:length(B.order)-1]...),
                         B.order[end]-1))
    else
        [B.order[1]]
    end
end

function restriction_extents(restriction::RestrictionMatrix)
    a = restriction.l
    b = restriction.raxis.stop - restriction.data.axes[2].stop - a
    a,b
end

restriction_extents(B::FEDVR) = 0,0
restriction_extents(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    restriction_extents(B.applied.args[2])

function block_structure(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B,restriction = B.applied.args
    bs = block_structure(B)
    a,b = restriction_extents(restriction)

    a ≤ bs[1] && b ≤ bs[end] ||
        throw(ArgumentError("Cannot restrict basis beyond first/last block"))
    bs[1] -= a
    bs[end] -= b
    bs
end

function block_bandwidths(B::Union{FEDVR,RestrictedQuasiArray{<:Any,2,<:FEDVR}}, rows::Vector{<:Integer})
    nrows = length(rows)
    if nrows > 1
        bw = o -> o > 2 ? [2,1] : [1]
        bws = bw.(order(B))
        l = vcat(1,bws[2:end]...)
        u = vcat(reverse.(bws[1:end-1])...,1)
        length(l) < nrows && (l = vcat(l,0))
        length(u) < nrows && (u = vcat(0,u))
        l,u
    else
        [0],[0]
    end
end

locs(B::FEDVR) = B.x

function locs(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B′,restriction = B.applied.args
    a,b = FEDVRQuasi.restriction_extents(restriction)
    B′.x[1+a:end-b]
end

IntervalSets.leftendpoint(B::FEDVR) = B.x[1]
IntervalSets.rightendpoint(B::FEDVR) = B.x[end]

IntervalSets.leftendpoint(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    leftendpoint(B.applied.args[1])
IntervalSets.rightendpoint(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    rightendpoint(B.applied.args[1])

# * Basis functions

function getindex(B::FEDVR{T}, x::Real, i::Integer, m::Integer) where T
    (x < B.t[i] || x > B.t[i+1]) && return zero(T)
    xⁱ = @elem(B,x,i)
    χ = @elem(B,n,i)[m]
    x′ = complex_rotate(x, B)
    for j = 1:B.order[i]
        j == m && continue
        χ *= (x′ - xⁱ[j])/(xⁱ[m]-xⁱ[j])
    end
    χ
end

checkbounds(B::FEDVR{T}, x::Real, k::Integer) where T =
    x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2) || throw(BoundsError())

@inline function getindex(B::FEDVR{T}, x::Real, k::Integer) where T
    # @boundscheck checkbounds(B, x, k) # Slow
    i = 1
    m = k
    while i < length(B.order) && m > B.order[i]
        m -= B.order[i] - 1
        i += 1
    end
    x < B.t[i] && return zero(T)
    if x > B.t[i+1]
        if i < length(B.t)-1 && m == B.order[i]
            i += 1
            m = 1
        else
            return zero(T)
        end
    end
    B[x,i,m]
end

@inline function Base.getindex(B::RestrictedBasis{<:FEDVR{T}}, x::Real, k::Integer) where {T}
    B′,restriction = B.args
    B′[x,k+restriction.l]
end

# * Types

const FEDVRArray{T,N,B<:FEDVR} = MulQuasiArray{T,N,<:Mul{<:Any,<:Tuple{B,<:AbstractArray{T,N}}}}
const FEDVRVector{T,B<:FEDVR} = FEDVRArray{T,1,B}
const FEDVRMatrix{T,B<:FEDVR} = FEDVRArray{T,2,B}
const FEDVRVecOrMat{T,B<:FEDVR} = Union{FEDVRVector{T,B},FEDVRMatrix{T,B}}

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

# * Mass matrix
function materialize(M::Mul{<:Any,<:Tuple{<:QuasiAdjoint{<:Any,<:FEDVR{T}},
                                          <:FEDVR{T}}}) where T
    Ac, B = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    Diagonal(ones(T, size(A,2)))
end

function materialize(M::Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix},
                                          <:QuasiAdjoint{<:Any,<:FEDVR},
                                          <:FEDVR{T},
                                          <:RestrictionMatrix}}) where T
    restAc,Ac,B,restB = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    restAc' == restB ||
        throw(ArgumentError("Non-equal restriction matrices not supported"))

    n = size(M,1)
    Diagonal(ones(T, n))
end

# * Norms

_norm(R::FEDVROrRestricted, ϕ::AbstractArray, p::Real=2) = norm(ϕ, p)

LinearAlgebra.norm(v::FEDVRVecOrMat, p::Real=2) = _norm(v.applied.args..., p)
LinearAlgebra.norm(v::Mul{<:Any, <:Tuple{<:FEDVROrRestricted, <:AbstractArray}},
                   p::Real=2) = _norm(v.args..., p)

function LinearAlgebra.normalize!(v::FEDVRVecOrMat, p::Real=2)
    v.applied.args[2][:] /= norm(v, p)
    v
end

function LinearAlgebra.normalize!(v::Mul{<:Any, <:Tuple{<:FEDVROrRestricted, <:AbstractArray}},
                                  p::Real=2)
    v.args[2][:] /= norm(v, p)
    v
end

# * Inner products

const FEDVRInnerProduct{T,U,B₁<:AdjointFEDVROrRestricted{U},B₂<:FEDVROrRestricted{U},V<:AbstractVector{T}} =
    Mul{<:Any, <:Tuple{<:Adjoint{<:Any,<:V}, <:B₁, <:B₂, <:V}}

const LazyFEDVRInnerProduct{B₁<:AdjointFEDVROrRestricted,B₂<:FEDVROrRestricted} = Mul{<:Any,<:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:Adjoint{<:Any,<:AbstractVector},
        <:B₁}},
    <:Mul{<:Any, <:Tuple{
        <:B₂,
        <:AbstractVector}}}}

function _inner_product(u::Adjoint{<:Any,<:AbstractVector}, A::AdjointFEDVROrRestricted,
                        B::FEDVROrRestricted, v::AbstractVector)
    A′ = unrestricted_basis(A')
    B′ = unrestricted_basis(B)
    a₁,b₁ = restriction_extents(A')
    a₂,b₂ = restriction_extents(B)
    A′ == B′ || throw(DimensionMismatch("Incompatible bases"))
    n = A′.n
    N = length(n)
    sel = (1+max(a₁,a₂)):(N-max(b₁,b₂))

    dot(conj(@view(u[sel .- a₁])),@view(v[sel .- a₂]))
end

LazyArrays.materialize(inner_product::FEDVRInnerProduct{T,U,B₁,B₂,V}) where {T,U,B₁,B₂,V} =
    _inner_product(inner_product.args...)

function LazyArrays.materialize(inner_product::LazyFEDVRInnerProduct{<:AdjointFEDVROrRestricted,<:FEDVROrRestricted})
    aA,Bb = inner_product.args
    _inner_product(aA.args..., Bb.args...)
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

        BlockSkylineMatrix{T}(undef, (rows,rows), (l,u))
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
    B′,restriction = B.applied.args
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

Matrix(f::Function, B::FEDVR{T}) where T = B(Diagonal(f.(B.x)))
function Matrix(f::Function, B::RestrictedQuasiArray{T,2,FEDVR{T}}) where T
    B′,restriction = B.applied.args
    a,b = restriction_extents(restriction)
    B(Diagonal(f.(B′.x[1+a:end-b])))
end
Matrix(::UniformScaling, B::FEDVROrRestricted{T}) where T = B(Diagonal(ones(T, size(B,2))))


# * Derivatives

"""
    lagrangeder!(xⁱ, m, L′)

Calculate the derivative of the Lagrange interpolating polynomial
Lⁱₘ(x), given the roots `xⁱ`, *at* the roots, and storing the result
in `L′`.

∂ₓ Lⁱₘ(xⁱₘ,) = (xⁱₘ-xⁱₘ,)⁻¹ ∏(k≠m,m′) (xⁱₘ,-xⁱₖ)/(xⁱₘ-xⁱₖ), m≠m′,
                [δ(m,n) - δ(m,1)]/2wⁱₘ,                    m=m′

Eq. (20) Rescigno2000
"""
function lagrangeder!(xⁱ::AbstractVector, wⁱ::AbstractVector,
                      L′::AbstractMatrix)
    δ(a,b) = a == b ? 1 : 0
    n = length(xⁱ)
    for m in 1:n
        L′[m,m] = (δ(m,n)-δ(m,1))/2wⁱ[m]
        for m′ in 1:n
            m′ == m && continue
            f = 1
            for k = 1:n
                k in [m,m′] && continue
                f *= (xⁱ[m′]-xⁱ[k])/(xⁱ[m]-xⁱ[k])
            end
            L′[m,m′] = f/(xⁱ[m]-xⁱ[m′])
        end
    end
end

function diff!(D, L̃, L′, wⁱ, nⁱ)
    n = size(L′,1)
    for m = 1:n
        for m′ = 1:n
            D[m,m′] = nⁱ[m]*nⁱ[m′]*dot(L̃[m,:],wⁱ.*L′[m′,:])
        end
    end
    D
end

function diff(B::FEDVR{T}, n::Integer, i::Integer) where T
    o = B.order[i]

    # L′ contains derivatives of un-normalized basis functions at
    # the quadrature roots.
    L′ = Matrix{T}(undef, o, o)
    lagrangeder!(@elem(B,x,i), B.wⁱ[i],L′)

    # D contains ⟨ξᵢ|χⱼ′⟩ where ξᵢ = χᵢ⁽ⁿ⁻¹⁾
    D = similar(L′)
    L̃ = n == 1 ? Matrix{T}(I, size(L′)...) : -L′ # ∂ᴴ = -∂
    diff!(D, L̃, L′, B.wⁱ[i], @elem(B,n,i))
    i ≥ B.i₀ && (D ./= √(B.eiϕ)) # TODO Check if correct
    D
end

difffun(B::FEDVR, n::Integer) = i -> diff(B,n,i)
difffun(B::RestrictedQuasiArray{<:Any,2,FEDVR}, n::Integer) = i -> diff(B.applied.args[1],n,i)

derop!(A::BlockSkylineMatrix{T}, B::FF, n::Integer) where {T,FF<:FEDVROrRestricted} =
    set_blocks!(difffun(B,n), A, B)

function derop!(A::Tridiagonal{T}, B::FEDVR{T}, n::Integer) where T
    A.dl .= zero(T)
    A.d .= zero(T)
    A.du .= zero(T)

    for i in eachindex(B.order)
        b = diff(B,n,i)
        A.dl[i] = b[2,1]
        A.d[i:i+1] .+= diag(b)
        A.du[i] = b[1,2]
    end

    A
end

function derop!(A::Tridiagonal{T}, B::RestrictedQuasiArray{T,2,FEDVR{T}}, n::Integer) where T
    B′,restriction = B.applied.args
    s,e = restriction_extents(restriction)

    A.dl .= zero(T)
    A.d .= zero(T)
    A.du .= zero(T)

    nel = length(B′.order)
    if s > 0
        b = diff(B′,n,s)
        A.d[1] = b[2,2]
    end
    for i in (1+s):(nel-e)
        b = diff(B′,n,i)
        ii = i-s
        A.dl[ii] = b[2,1]
        A.d[ii:ii+1] .+= diag(b)
        A.du[ii] = b[1,2]
    end
    if e > 0
        b = diff(B′,n,nel)
        A.d[end] += b[1,1]
    end

    A
end

const FlatFirstDerivative = Mul{<:Any, <:Tuple{
    <:QuasiAdjoint{<:Any, <:FEDVR},
    <:Derivative,
    <:FEDVR}}
const FlatRestrictedFirstDerivative = Mul{<:Any, <:Tuple{
    <:Adjoint{<:Any,<:RestrictionMatrix},
    <:QuasiAdjoint{<:Any, <:FEDVR},
    <:Derivative,
    <:FEDVR,
    <:RestrictionMatrix}}

const LazyFirstDerivative = Mul{<:Any, <:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:QuasiAdjoint{<:Any, <:FEDVR},
        <:Derivative}},
    <:FEDVR}}

const LazyRestrictedFirstDerivative = Mul{<:Any, <:Tuple{
    <:Mul{<:Any,<:Tuple{
        <:MulQuasiArray{<:Any, 2, <:Mul{<:Any, <:Tuple{
            <:Adjoint{<:Any,<:RestrictionMatrix},
            <:QuasiAdjoint{<:Any,<:FEDVR}}}},
        <:Derivative}},
    <:RestrictedQuasiArray{<:Any,2,<:FEDVR}}}

const FirstDerivative = Union{FlatFirstDerivative, FlatRestrictedFirstDerivative,
                              LazyFirstDerivative, LazyRestrictedFirstDerivative}

const FlatSecondDerivative = Mul{<:Any, <:Tuple{
    <:QuasiAdjoint{<:Any, <:FEDVR},
    <:QuasiAdjoint{<:Any, <:Derivative},
    <:Derivative,
    <:FEDVR}}
const FlatRestrictedSecondDerivative = Mul{<:Any, <:Tuple{
    <:Adjoint{<:Any,<:RestrictionMatrix},
    <:QuasiAdjoint{<:Any, <:FEDVR},
    <:QuasiAdjoint{<:Any, <:Derivative},
    <:Derivative,
    <:FEDVR,
    <:RestrictionMatrix}}

const LazySecondDerivative = Mul{<:Any, <:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:Mul{<:Any, <:Tuple{
            <:QuasiAdjoint{<:Any, <:FEDVR}, <:QuasiAdjoint{<:Any, <:Derivative}}},
        <:Derivative}},
    <:FEDVR}}

const LazyRestrictedSecondDerivative = Mul{<:Any, <:Tuple{
    <:Mul{<:Any,<:Tuple{
        <:Mul{<:Any,<:Tuple{
            <:MulQuasiArray{<:Any, 2, <:Mul{<:Any, <:Tuple{
                <:Adjoint{<:Any,<:RestrictionMatrix},
                <:QuasiAdjoint{<:Any,<:FEDVR}}}},
            <:QuasiAdjoint{<:Any,<:Derivative}}},
        <:Derivative}},
    <:RestrictedQuasiArray{<:Any,2,<:FEDVR}}}

const SecondDerivative = Union{FlatSecondDerivative,FlatRestrictedSecondDerivative,
                               LazySecondDerivative,LazyRestrictedSecondDerivative}

const FirstOrSecondDerivative = Union{FirstDerivative,SecondDerivative}

difforder(::FirstDerivative) = 1
difforder(::SecondDerivative) = 2

function copyto!(dest::Union{Tridiagonal,BlockSkylineMatrix}, M::FirstOrSecondDerivative)
    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))
    derop!(dest, basis(M), difforder(M))
    dest
end

basis(M::Union{FlatFirstDerivative,LazyFirstDerivative,
               FlatSecondDerivative,LazySecondDerivative}) = last(M.args)

basis(M::Union{FlatRestrictedFirstDerivative, FlatRestrictedSecondDerivative}) =
    M.args[end-1]*M.args[end]

similar(M::FirstOrSecondDerivative, ::Type{T}) where T = Matrix(undef, basis(M))
materialize(M::FirstOrSecondDerivative) = copyto!(similar(M, eltype(M)), M)

# * Densities
function Base.Broadcast.broadcasted(::typeof(*), a::M, b::M) where {T,N,M<:FEDVRArray{T,N}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    A,ca = a.applied.args
    B,cb = b.applied.args
    A == B || throw(DimensionMismatch("Incompatible bases"))
    c = similar(ca)
    # We want the first MulQuasiArray to be conjugated, if complex
    @. c = conj(ca) * cb * A.n
    A*c
end

struct FEDVRDensity{T,B<:FEDVROrRestricted,V<:AbstractVecOrMat{T}}
    R::B
    u::V
    v::V
end

function _FEDVRDensity(Ra::FEDVROrRestricted, ca::AbstractVecOrMat,
                       Rb::FEDVROrRestricted, cb::AbstractVecOrMat)
    Ra == Rb || throw(DimensionMismatch("Incompatible bases"))
    FEDVRDensity(Ra, ca, cb)
end

function Base.copyto!(cρ::AbstractVecOrMat{T}, ld::FEDVRDensity{T,R}, Rρ::R) where {T,R}
    Rρ == ld.R || throw(DimensionMismatch("Incompatible bases"))
    size(cρ) == size(ld.u) || throw(DimensionMismatch("Incompatible sizes"))
    a,b = restriction_extents(Rρ)
    n = unrestricted_basis(Rρ).n
    # We want the first MulQuasiArray to be conjugated, if complex
    @. cρ = conj(ld.u) * ld.v * @view(n[1+a:end-b])
end

function Base.Broadcast.broadcasted(::typeof(⋆), a::V, b::V) where {T,B<:FEDVR,V<:FEDVRVecOrMat{T,B}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    _FEDVRDensity(a.applied.args..., b.applied.args...)
end

function Base.copyto!(ρ::FEDVRVecOrMat{T,R}, ld::FEDVRDensity{T,R}) where {T,R}
    copyto!(ρ.applied.args[2], ld, ρ.applied.args[1])
    ρ
end

function Base.Broadcast.broadcasted(::typeof(⋆), a::V₁, b::V₂) where {T,B<:FEDVROrRestricted,
                                                                      V₁<:Mul{<:Any, <:Tuple{B,<:AbstractVector{T}}},
                                                                      V₂<:Mul{<:Any, <:Tuple{B,<:AbstractVector{T}}}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    _FEDVRDensity(a.args..., b.args...)
end

function Base.copyto!(ρ::Mul{<:Any, <:Tuple{R,<:AbstractVector{T}}}, ld::FEDVRDensity{T,R}) where {T,R}
    copyto!(ρ.args[2], ld, ρ.args[1])
    ρ
end

# * Projections

function dot(B::FEDVR{T}, f::Function) where T
    v = zeros(T, size(B,2))
    for i ∈ 1:nel(B)
        @. v[B.elems[i]] += B.wⁱ[i]*f(@elem(B,x,i))
    end
    v .*= B.n
    v
end

function dot(B::RestrictedQuasiArray{T,2,FEDVR{T}}, f::Function) where T
    B′,restriction = B.applied.args
    a,b = restriction_extents(restriction)

    n = size(B,2)
    v = zeros(T, n)
    for i ∈ 1:nel(B′)
        sel = B′.elems[i]
        subsel = if 1+a<sel[1] && n-b > sel[end]
            Colon()
        else
            s = min(max(1+a,sel[1]),sel[end])
            e = max(min(n-b,sel[end]),sel[1])
            findfirst(isequal(s),sel):findfirst(isequal(e),sel)
        end

        @. v[sel[subsel] .- a] += @view((B′.wⁱ[i]*f(@elem(B′,x,i)))[subsel])
    end
    v .*= @view(B′.n[1+a:end-b])
    v
end

# * Exports

export FEDVR, Derivative, @elem, dot

end # module
