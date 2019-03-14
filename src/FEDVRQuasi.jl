module FEDVRQuasi

import Base: axes, size, ==, getindex, checkbounds, copyto!, similar, diff, show
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: ℵ₁
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray

using IntervalSets

using LazyArrays
import LazyArrays: ⋆
using FillArrays

using LinearAlgebra
import LinearAlgebra: Matrix, dot

using FastGaussQuadrature, BlockBandedMatrices

using Printf

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

# * Properties

axes(B::FEDVR) = (first(B.t)..last(B.t), Base.OneTo(length(B.x)))
size(B::FEDVR) = (ℵ₁, length(B.x))
==(A::FEDVR,B::FEDVR) = A.t == B.t && A.order == B.order

nel(B::FEDVR) = length(B.order)
element_boundaries(B::FEDVR) = vcat(1,1 .+ cumsum(B.order .- 1))

complex_rotate(x,B::FEDVR{T}) where {T<:Real} = x
complex_rotate(x,B::FEDVR{T}) where {T<:Complex} = x < B.t₀ ? x : B.t₀ + (x-B.t₀)*B.eiϕ

macro elem(B,v,i)
    :(@view($(esc(B)).$v[$(esc(B)).elems[$(esc(i))]]))
end

function show(io::IO, B::FEDVR{T}) where T
    write(io, "FEDVR{$(T)} basis with $(nel(B)) elements on $(axes(B,1))")
    if T <: Complex
        rot = @printf(io, " with %s @ %.2f°", B.t₀ <= first(B.t) ? "ICS" : "ECS", rad2deg(angle(B.eiϕ)))
        B.t₀ > first(B.t) && @printf(io, " starting at %.2g", B.t₀)
    end
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

function block_bandwidths(B::FEDVR, rows::Vector{<:Integer})
    nrows = length(rows)
    if nrows > 1
        bw = o -> o > 2 ? [2,1] : [1]
        bws = bw.(B.order)
        l = vcat(1,bws[2:end]...)
        u = vcat(reverse.(bws[1:end-1])...,1)
        length(l) < nrows && (l = vcat(l,0))
        length(u) < nrows && (u = vcat(0,u))
        l,u
    else
        [0],[0]
    end
end


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

# * Types

const FEDVRArray{T,N,B<:FEDVR} = MulQuasiArray{T,N,<:Mul{<:Any,<:Tuple{B,<:AbstractArray{T,N}}}}
const FEDVRVector{T,B<:FEDVR} = FEDVRArray{T,1,B}
const FEDVRMatrix{T,B<:FEDVR} = FEDVRArray{T,2,B}
const FEDVRVecOrMat{T,B} = Union{FEDVRVector{T,B},FEDVRMatrix{T,B}}

# * Diagonal matrices
DiagonalBlockDiagonal(A::AbstractMatrix, (rows,cols)::Tuple) =
    BandedBlockBandedMatrix(A, (rows,cols), (0,0), (0,0))

DiagonalBlockDiagonal(A::AbstractMatrix, rows) =
    DiagonalBlockDiagonal(A, (rows,rows))

function (B::FEDVR)(D::Diagonal)
    n = size(B,2)
    @assert size(D) == (n,n)
    DiagonalBlockDiagonal(D, block_structure(B))
end

# * Mass matrix
function materialize(M::Mul{<:Any,<:Tuple{<:QuasiAdjoint{<:Any,<:FEDVR{T}},<:FEDVR{T}}}) where T
    Ac, B = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    B(Diagonal(ones(T, size(A,2))))
end

# * Dense operators

function Matrix(::UndefInitializer, B::FEDVR{T}) where T
    if all(B.order .== 2)
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

function set_blocks!(fun::Function, A::BlockSkylineMatrix{T}, B::FEDVR{T}) where T
    nel = length(B.order)

    A.data .= zero(T)

    j = 1
    for i in eachindex(B.order)
        b = fun(i)
        o = B.order[i]
        @assert size(b,1) == size(b,2) == o

        s = 1+(i>1)
        e = size(b,1)-(i<nel)

        if o > 2
            j += i > 1
            A[Block(j,j)] .= b[s:e,s:e]
            if i < nel
                A[Block(j+1,j+1)] .+= b[end,end]
                A[Block(j,j+1)] = b[s:e,end]
                A[Block(j+1,j)] = reshape(b[end,s:e], 1, e-s+1)
            end
            if i > 1
                A[Block(j-1,j-1)] .+= b[1,1]
                A[Block(j,j-1)] = b[s:e,1]
                A[Block(j-1,j)] = reshape(b[1,s:e], 1, e-s+1)
            end
            if i > 1 && i < nel
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
    A
end

# * Scalar operators

Matrix(f::Function, B::FEDVR{T}) where T = B(Diagonal(f.(B.x)))
Matrix(::UniformScaling, B::FEDVR{T}) where T = B(Diagonal(ones(T, size(B,2))))


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

derop!(A::BlockSkylineMatrix{T}, B::FEDVR{T}, n::Integer) where T =
    set_blocks!(i -> diff(B,n,i), A, B)

function derop!(A::Tridiagonal{T}, B::FEDVR{T}, n::Integer) where T
    nel = length(B.order)

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

const FirstDerivative = Union{
    Mul{<:Any, <:Tuple{<:QuasiAdjoint{<:Any, <:FEDVR}, <:Derivative, <:FEDVR}},
    Mul{<:Any, <:Tuple{<:Mul{<:Any, <:Tuple{<:QuasiAdjoint{<:Any, <:FEDVR}, <:Derivative}}, <:FEDVR}}
}
const SecondDerivative = Union{
    Mul{<:Any, <:Tuple{<:QuasiAdjoint{<:Any, <:FEDVR}, <:QuasiAdjoint{<:Any, <:Derivative}, <:Derivative, <:FEDVR}},
    Mul{<:Any, <:Tuple{
        <:Mul{<:Any, <:Tuple{
            <:Mul{<:Any, <:Tuple{
                <:QuasiAdjoint{<:Any, <:FEDVR}, <:QuasiAdjoint{<:Any, <:Derivative}}},
            <:Derivative}}, <:FEDVR}}
}
# const FirstDerivative = Mul{<:Any, <:Tuple{<:QuasiAdjoint{<:Any, <:FEDVR}, <:Derivative, <:FEDVR}}
# const SecondDerivative = Mul{<:Any, <:Tuple{<:QuasiAdjoint{<:Any, <:FEDVR}, <:QuasiAdjoint{<:Any, <:Derivative}, <:Derivative, <:FEDVR}}
const FirstOrSecondDerivative = Union{FirstDerivative,SecondDerivative}

difforder(::FirstDerivative) = 1
difforder(::SecondDerivative) = 2

function copyto!(dest::AbstractMatrix, M::FirstOrSecondDerivative)
    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))
    derop!(dest, last(M.args), difforder(M))
    dest
end

similar(M::FirstOrSecondDerivative, ::Type{T}) where T = Matrix(undef, last(M.args))
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

struct FEDVRDensity{T,B<:FEDVR,V<:AbstractVecOrMat{T}}
    R::B
    u::V
    v::V
end

function Base.Broadcast.broadcasted(::typeof(⋆), a::V, b::V) where {T,B<:FEDVR,V<:FEDVRVecOrMat{T,B}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    Ra,ca = a.applied.args
    Rb,cb = b.applied.args
    Ra == Rb || throw(DimensionMismatch("Incompatible bases"))
    FEDVRDensity(Ra, ca, cb)
end

function Base.copyto!(ρ::FEDVRVecOrMat{T,R}, ld::FEDVRDensity{T,R}) where {T,R}
    Rρ,cρ = ρ.applied.args
    Rρ == ld.R || throw(DimensionMismatch("Incompatible bases"))
    size(cρ) == size(ld.u) || throw(DimensionMismatch("Incompatible sizes"))
    # We want the first MulQuasiArray to be conjugated, if complex
    @. cρ = conj(ld.u) * ld.v * Rρ.n
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

# * Exports

export FEDVR, Derivative, @elem, dot

end # module
