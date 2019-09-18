module FEDVRQuasi

import Base: axes, size, ==, getindex, checkbounds, copyto!, similar, diff, show
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: Basis, ℵ₁, @simplify

using QuasiArrays
import QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray,
    PInvQuasiMatrix, InvQuasiMatrix, BroadcastQuasiArray

using BandedMatrices

using IntervalSets

using LazyArrays
using FillArrays

using LinearAlgebra
import LinearAlgebra: Matrix, dot

using FastGaussQuadrature, BlockBandedMatrices

using Printf

# * Auxilliary type definitions for restricted bases

const RestrictionMatrix = BandedMatrix{<:Int, <:FillArrays.Ones}

const RestrictionTuple{B<:Basis} = Tuple{B, <:RestrictionMatrix}
const AdjointRestrictionTuple{B<:Basis} = Tuple{<:Adjoint{<:Any,<:RestrictionMatrix}, <:QuasiAdjoint{<:Any,B}}

const RestrictedBasis{B<:Basis} = Mul{<:Any,<:RestrictionTuple{B}}
const AdjointRestrictedBasis{B<:Basis} = Mul{<:Any,<:AdjointRestrictionTuple{B}}

const RestrictedQuasiArray{T,N,B<:Basis} = MulQuasiArray{T,N,<:RestrictionTuple{B}}
const AdjointRestrictedQuasiArray{T,N,B<:Basis} = MulQuasiArray{T,N,<:AdjointRestrictionTuple{B}}

const BasisOrRestricted{B<:Basis} = Union{B,RestrictedBasis{<:B},<:RestrictedQuasiArray{<:Any,<:Any,<:B}}
const AdjointBasisOrRestricted{B<:Basis} = Union{<:QuasiAdjoint{<:Any,B},AdjointRestrictedBasis{<:B},<:AdjointRestrictedQuasiArray{<:Any,<:Any,<:B}}

unrestricted_basis(R::AbstractQuasiMatrix) = R
unrestricted_basis(R::RestrictedBasis) = first(R.args)
unrestricted_basis(R::RestrictedQuasiArray) = first(R.args)

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

struct FEDVR{T,R<:Real,O<:AbstractVector} <: Basis{T}
    t::AbstractVector{R}
    order::O
    i₀::Int
    t₀::R
    eiϕ::T
    x::Vector{T}
    wⁱ::Vector{Vector{R}}
    n::Vector{T}
    elems::Vector{UnitRange{Int}}
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

const RestrictedFEDVR{T} = Union{RestrictedBasis{<:FEDVR{T}},<:RestrictedQuasiArray{<:Any,<:Any,<:FEDVR{T}}}
const AdjointRestrictedFEDVR{T} = Union{AdjointRestrictedBasis{<:FEDVR{T}},<:AdjointRestrictedQuasiArray{<:Any,<:Any,<:FEDVR{T}}}

const FEDVROrRestricted{T} = BasisOrRestricted{<:FEDVR{T}}
const AdjointFEDVROrRestricted{T} = AdjointBasisOrRestricted{<:FEDVR{T}}

# * Properties

axes(B::FEDVR) = (Inclusion(first(B.t)..last(B.t)), Base.OneTo(length(B.x)))
size(B::FEDVR) = (ℵ₁, length(B.x))
size(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) = (ℵ₁, length(B.args[2].data))
==(A::FEDVR,B::FEDVR) = A.t == B.t && A.order == B.order
==(A::FEDVROrRestricted,B::FEDVROrRestricted) = unrestricted_basis(A) == unrestricted_basis(B)

order(B::FEDVR) = B.order
# This assumes that the restriction matrices do not remove blocks
order(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) = order(B.args[1])

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
    B′,restriction = B.args
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
    restriction_extents(B.args[2])

function block_structure(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B,restriction = B.args
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

rlocs(B::FEDVR{<:Real}) = locs(B)

function rlocs(B::FEDVR{T}) where T
    R = real(T)
    nx = length(B.x)
    x = Vector{R}(undef, nx)
    ii = 1
    for i in eachindex(B.order)
        xw = element_grid(B.order[i], B.t[i], B.t[i+1], B.t[1], one(R))
        copyto!(view(x, ii:nx), xw[1])
        ii += length(xw[1])-1
    end
    x
end

function locs(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B′,restriction = B.args
    a,b = FEDVRQuasi.restriction_extents(restriction)
    B′.x[1+a:end-b]
end

function rlocs(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B′,restriction = B.args
    a,b = FEDVRQuasi.restriction_extents(restriction)
    rlocs(B′)[1+a:end-b]
end

IntervalSets.leftendpoint(B::FEDVR) = B.x[1]
IntervalSets.rightendpoint(B::FEDVR) = B.x[end]

IntervalSets.leftendpoint(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    leftendpoint(B.args[1])
IntervalSets.rightendpoint(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    rightendpoint(B.args[1])

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

# @inline function Base.getindex(B::RestrictedBasis{<:FEDVR{T}}, x::Real, k::Integer) where {T}
#     B′,restriction = B.args
#     B′[x,k+restriction.l]
# end

# * Types

const FEDVRArray{T,N,B<:FEDVROrRestricted} = MulQuasiArray{T,N,<:Tuple{B,<:AbstractArray{T,N}}}
const FEDVRVector{T,B<:FEDVROrRestricted} = FEDVRArray{T,1,B}
const FEDVRMatrix{T,B<:FEDVROrRestricted} = FEDVRArray{T,2,B}
const FEDVRVecOrMat{T,B<:FEDVROrRestricted} = Union{FEDVRVector{T,B},FEDVRMatrix{T,B}}

const AdjointFEDVRArray{T,N,B<:FEDVROrRestricted} = MulQuasiArray{T,<:Any,<:Tuple{<:Adjoint{T,<:AbstractArray{T,N}},
                                                                                  <:QuasiAdjoint{T,<:B}}}
const AdjointFEDVRVector{T,B<:FEDVROrRestricted} = AdjointFEDVRArray{T,1,B}
const AdjointFEDVRMatrix{T,B<:FEDVROrRestricted} = AdjointFEDVRArray{T,2,B}
const AdjointFEDVRVecOrMat{T,B<:FEDVROrRestricted} = Union{AdjointFEDVRVector{T,B},AdjointFEDVRMatrix{T,B}}

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

@simplify function *(Ac::QuasiAdjoint{<:Any,<:FEDVR}, B::FEDVR)
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    I
end

# A & B restricted
function materialize(M::Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix},
                                          <:QuasiAdjoint{<:Any,<:FEDVR{T}},
                                          <:FEDVR{T},
                                          <:RestrictionMatrix}}) where T
    restAc,Ac,B,restB = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))

    # This is mainly for type-stability; it would be trivial to
    # generate the proper restriction matrix from the combination of
    # two differently restricted bases, but we would like to have
    # UniformScaling as the result if they are equal, and this has
    # higher priority. On the other hand, you typically only compute
    # the mass matrix in the beginning of the calculation, and thus
    # type-instability is not a big problem, so this behaviour may
    # change in the future.
    restAc' == restB ||
        throw(ArgumentError("Non-equal restriction matrices not supported"))

    I
end

# A unrestricted, B restricted
materialize(M::Mul{<:Any,<:Tuple{UniformScaling{Bool},<:RestrictionMatrix}}) = M.args[2]
# A restricted, B unrestricted
materialize(M::Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix},UniformScaling{Bool}}}) = M.args[1]

# * Basis inverses

@simplify function *(A⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}},
                     B::FEDVR)
    A = parent(A⁻¹)
    A == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    I
end

@simplify function *(A::FEDVR,
                     B⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}})
    B = parent(B⁻¹)
    A == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    I
end

@simplify function *(A⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}},
                     v::FEDVRArray)
    A = parent(A⁻¹)
    B,c = v.args
    A == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    c
end

@simplify function *(v::AdjointFEDVRArray,
                     B⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}})
    c,Ac = v.args
    B = parent(B⁻¹)
    parent(Ac) == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    c
end

# * Norms

_norm(R::FEDVROrRestricted, ϕ::AbstractArray, p::Real=2) = norm(ϕ, p)

LinearAlgebra.norm(v::FEDVRVecOrMat, p::Real=2) = _norm(v.args..., p)
LinearAlgebra.norm(v::Mul{<:Any, <:Tuple{<:FEDVROrRestricted, <:AbstractArray}},
                   p::Real=2) = _norm(v.args..., p)

function LinearAlgebra.normalize!(v::FEDVRVecOrMat, p::Real=2)
    v.args[2][:] /= norm(v, p)
    v
end

function LinearAlgebra.normalize!(v::Mul{<:Any, <:Tuple{<:FEDVROrRestricted, <:AbstractArray}},
                                  p::Real=2)
    v.args[2][:] /= norm(v, p)
    v
end

# * Inner products

const FEDVRInnerProduct =
    Mul{<:Any, <:Tuple{<:Adjoint{<:Any,<:AbstractVecOrMat},
                       <:AdjointFEDVROrRestricted,
                       <:FEDVROrRestricted,
                       <:AbstractVecOrMat}}

const LazyFEDVRInnerProduct{B₁<:AdjointFEDVROrRestricted,B₂<:FEDVROrRestricted} = Mul{<:Any,<:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:Adjoint{<:Any,<:AbstractVecOrMat},
        <:B₁}},
    <:Mul{<:Any, <:Tuple{
        <:B₂,
        <:AbstractVecOrMat}}}}

# No restrictions
function _inner_product(u::Adjoint{<:Any,<:AbstractArray}, A::QuasiAdjoint{<:Any,<:FEDVR},
                        B::FEDVR, v::AbstractArray)
    A' == B || throw(DimensionMismatch("Incompatible bases"))
    u*v
end

selview(v::AbstractVector, sel) = view(v, sel)
# Using ' would conjugate the elements, which we do not want since
# they are already conjugated.
selview(v::Adjoint{<:Any,<:AbstractVector}, sel) = transpose(view(v, sel))

selview(A::AbstractMatrix, sel) = view(A, sel, :)
selview(A::Adjoint{<:Any,<:AbstractMatrix}, sel) = view(A, :, sel)

# Implementation for various restrictions
function _inner_product(u::Adjoint{<:Any,<:AbstractVecOrMat}, a₁, b₁,
                        A::FEDVR, B::FEDVR,
                        v::AbstractVecOrMat, a₂, b₂)
    A == B || throw(DimensionMismatch("Incompatible bases"))
    n = A.n
    N = length(n)
    sel = (1+max(a₁,a₂)):(N-max(b₁,b₂))

    selview(u, sel .- a₁) * selview(v, sel .- a₂)
end

# A & B restricted
function _inner_product(u::Adjoint{<:Any,<:AbstractVecOrMat}, A::AdjointRestrictedFEDVR,
                        B::RestrictedFEDVR, v::AbstractVecOrMat)
    A′ = unrestricted_basis(A')
    B′ = unrestricted_basis(B)
    a₁,b₁ = restriction_extents(A')
    a₂,b₂ = restriction_extents(B)
    _inner_product(u, a₁, b₁, A′, B′, v, a₂, b₂)
end

# A unrestricted, B restricted
function _inner_product(u::Adjoint{<:Any,<:AbstractVecOrMat}, A::QuasiAdjoint{<:Any,<:FEDVR},
                        B::RestrictedFEDVR, v::AbstractVecOrMat)
    B′ = unrestricted_basis(B)
    a,b = restriction_extents(B)
    _inner_product(u, 0, 0, A', B′, v, a, b)
end

# A restricted, B unrestricted
function _inner_product(u::Adjoint{<:Any,<:AbstractVecOrMat}, A::AdjointRestrictedFEDVR,
                        B::FEDVR, v::AbstractVecOrMat)
    A′ = unrestricted_basis(A')
    a,b = restriction_extents(A')
    _inner_product(u, a, b, A′, B, v, 0, 0)
end

LazyArrays.materialize(inner_product::FEDVRInnerProduct) =
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
difffun(B::RestrictedQuasiArray{<:Any,2,FEDVR}, n::Integer) = i -> diff(B.args[1],n,i)

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
    B′,restriction = B.args
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
        <:MulQuasiArray{<:Any, 2, <:Tuple{
            <:Adjoint{<:Any,<:RestrictionMatrix},
            <:QuasiAdjoint{<:Any,<:FEDVR}}},
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
            <:MulQuasiArray{<:Any, 2, <:Tuple{
                <:Adjoint{<:Any,<:RestrictionMatrix},
                <:QuasiAdjoint{<:Any,<:FEDVR}}},
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
    A,ca = a.args
    B,cb = b.args
    A == B || throw(DimensionMismatch("Incompatible bases"))
    c = similar(ca)
    a,b = restriction_extents(A)
    n = unrestricted_basis(A).n
    # We want the first MulQuasiArray to be conjugated, if complex
    @. c = conj(ca) * cb * @view(n[1+a:end-b])
    A*c
end

struct FEDVRDensity{T,B<:FEDVROrRestricted,
                    U<:AbstractVecOrMat{T},V<:AbstractVecOrMat{T}}
    R::B
    u::U
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
    _FEDVRDensity(a.args..., b.args...)
end

function Base.copyto!(ρ::FEDVRVecOrMat{T,R}, ld::FEDVRDensity{T,R}) where {T,R}
    copyto!(ρ.args[2], ld, ρ.args[1])
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

# * Function interpolation

function Base.:(\ )(B::FEDVR{T}, f::BroadcastQuasiArray) where T
    axes(f,1) == axes(B,1) ||
        throw(DimensionMismatch("Function on $(axes(f,1).domain) cannot be interpolated over basis on $(axes(B,1).domain)"))

    v = zeros(T, size(B,2))
    for i ∈ 1:nel(B)
        @. v[B.elems[i]] += B.wⁱ[i]*f[@elem(B,x,i)]
    end
    v .*= B.n
    v
end

function Base.:(\ )(B::RestrictedQuasiArray{T,2,FEDVR{T}}, f::BroadcastQuasiArray) where T
    axes(f,1) == axes(B,1) ||
        throw(DimensionMismatch("Function on $(axes(f,1).domain) cannot be interpolated over basis on $(axes(B,1).domain)"))

    B′,restriction = B.args
    a,b = restriction_extents(restriction)

    n = size(B′,2)
    v = zeros(T, size(B,2))
    for i ∈ 1:nel(B′)
        sel = B′.elems[i]
        # Find which basis functions of finite-element `i` should be
        # evaluated.
        subsel = if 1+a<sel[1] && n-b > sel[end]
            # Element is completely within the restricted basis.
            Colon()
        else
            # Element straddles restriction (we don't allow
            # restrictions that would throw away an entire
            # finite-element); find subset of functions that are
            # within the restriction.
            s = min(max(1+a,sel[1]),sel[end])
            e = max(min(n-b,sel[end]),sel[1])
            findfirst(isequal(s),sel):findfirst(isequal(e),sel)
        end

        @. v[sel[subsel] .- a] += @view((B′.wⁱ[i]*f[@elem(B′,x,i)])[subsel])
    end
    v .*= @view(B′.n[1+a:end-b])
    v
end

# * Exports

export FEDVR, Derivative, @elem, dot

end # module
