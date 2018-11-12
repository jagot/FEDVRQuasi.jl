module FEDVRQuasi

import Base: axes, size, ==, getindex, checkbounds, copyto!, similar, diff, show
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: ℵ₁
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint

using IntervalSets

using LazyArrays
import LazyArrays: Mul2
using FillArrays

using LinearAlgebra
import LinearAlgebra: Matrix

using FastGaussQuadrature, BlockBandedMatrices

using Printf

# https://github.com/JuliaLang/julia/pull/18777
lerp(a::T,b::T,t) where T = T(fma(t, b, fma(-t, a, a)))
lerp(a::R,b::R,t::C) where {R<:Real,C<:Complex} = lerp(a,b,real(t)) + im*lerp(a,b,imag(t))
lerp(a::C,b::C,t::R) where {R<:Real,C<:Complex} = lerp(real(a),real(b),t) + im*lerp(imag(a),imag(b),(t))

function element_grid(order, a::T, b::T, c::T=zero(T), eiϕ=one(T)) where T
    x,w = gausslobatto(order)
    c .+ lerp.(Ref(eiϕ*(a-c)), Ref(eiϕ*(b-c)), (x .+ 1)/2),(b-a)*w/2
end


struct FEDVR{T,R<:Real,O<:AbstractVector} <: AbstractQuasiMatrix{T}
    t::AbstractVector{R}
    order::O
    i₀::Integer
    t₀::R
    eiϕ::T
    x::Vector{T}
    xⁱ::Vector{<:SubArray{T}}
    wⁱ::Vector{Vector{R}}
    n::Vector{T}
    nⁱ::Vector{<:SubArray{T}}
    function FEDVR(t::AbstractVector{R}, order::O; t₀::R=zero(R), ϕ::R=zero(R)) where {R<:Real,O<:AbstractVector}
        @assert length(order) == length(t)-1
        @assert all(order .> 1)

        i₀,eiϕ,T = if ϕ ≠ zero(R)
            findfirst(tt -> tt ≥ t₀, t),exp(im*ϕ),complex(R)
        else
            1,one(R),R
        end
        i₀ === nothing && error("Complex scaling starting point outside grid $(t)")
        t₀ = t[i₀]

        x,w = zip([element_grid(order[i], t[i], t[i+1], t₀, i ≥ i₀ ? eiϕ : one(T))
                   for i in eachindex(order)]...)

        rot = (i,v) -> i ≥ i₀ ? eiϕ*v : v
        n = [one(T) ./ .√(rot(i,wⁱ)) for (i,wⁱ) in enumerate(w)]
        for i in 1:length(order)-1
            n[i][end] = n[i+1][1] = one(T) ./ √(rot(i,w[i][end]) + rot(i+1,w[i+1][1]))
        end

        X = vcat(x[1], [x[i][2:end] for i in 2:length(order)]...)
        N = vcat(n[1], [n[i][2:end] for i in 2:length(order)]...)

        xⁱ = [@view(X[1:order[1]])]
        nⁱ = [@view(N[1:order[1]])]

        l = order[1]
        for i = 2:length(order)
            l′ = l+order[i]-1
            push!(xⁱ, @view(X[l:l′]))
            push!(nⁱ, @view(N[l:l′]))
            l = l′
        end
        new{T,R,O}(t,order,i₀,t₀,eiϕ,X,xⁱ,[w...],N,nⁱ)
    end
end

FEDVR(t::AbstractVector{T},order::Integer; kwargs...) where T =
    FEDVR(t, Fill(order,length(t)-1); kwargs...)

axes(B::FEDVR) = (first(B.t)..last(B.t), Base.OneTo(length(B.x)))
size(B::FEDVR) = (ℵ₁, length(B.x))
==(A::FEDVR,B::FEDVR) = A.t == B.t && A.order == B.order

nel(B::FEDVR) = length(B.order)
element_boundaries(B::FEDVR) = vcat(1,1 .+ cumsum(B.order .- 1))

complex_rotate(x,B::FEDVR{T}) where {T<:Real} = x
complex_rotate(x,B::FEDVR{T}) where {T<:Complex} = x < B.t₀ ? x : B.t₀ + (x-B.t₀)*B.eiϕ

function show(io::IO, B::FEDVR{T}) where T
    write(io, "FEDVR{$(T)} basis with $(nel(B)) elements on $(axes(B,1))")
    if T <: Complex
        rot = @printf(" with %s @ %.2f°", B.t₀ <= first(B.t) ? "ICS" : "ECS", rad2deg(angle(B.eiϕ)))
        B.t₀ > first(B.t) && @printf(io, " starting at %.2g", B.t₀)
    end
end


# * Basis functions

function getindex(B::FEDVR{T}, x::Real, i::Integer, m::Integer) where T
    (x < B.t[i] || x > B.t[i+1]) && return zero(T)
    xⁱ = B.xⁱ[i]
    χ = B.nⁱ[i][m]
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
    @boundscheck checkbounds(B, x, k) # Slow
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

# * Mass matrix
function materialize(M::Mul2{<:Any,<:Any,<:QuasiAdjoint{<:Any,<:FEDVR{T}},<:FEDVR{T}}) where T
    Ac, B = M.factors
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    Diagonal(ones(T, size(A,2)))
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
        bs = o -> o > 2 ? [o-2,1] : [1]
        bw = o -> o > 2 ? [2,1] : [1]
        rows,l,u = if length(B.order) > 1
            rows = Vector{Int}(vcat(B.order[1]-1,1,
                                    vcat([bs(B.order[i]) for i = 2:length(B.order)-1]...),
                                    B.order[end]-1))
            bws = bw.(B.order)
            l = vcat(1,bws[2:end]...)
            u = vcat(reverse.(bws[1:end-1])...,1)
            length(l) < length(rows) && (l = vcat(l,0))
            length(u) < length(rows) && (u = vcat(0,u))
            rows,l,u
        else
            [B.order[1]],[0],[0]
        end

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

Matrix(f::Function, B::FEDVR{T}) where T = Diagonal(f.(B.x))
Matrix(::UniformScaling, B::FEDVR{T}) where T = Diagonal(ones(T, size(B,2)))


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
    lagrangeder!(B.xⁱ[i], B.wⁱ[i],L′)

    # D contains ⟨ξᵢ|χⱼ′⟩ where ξᵢ = χᵢ⁽ⁿ⁻¹⁾
    D = similar(L′)
    L̃ = n == 1 ? Matrix{T}(I, size(L′)...) : -L′ # ∂ᴴ = -∂
    diff!(D, L̃, L′, B.wⁱ[i], B.nⁱ[i])
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

const FirstDerivative{T} = Mul2{<:Any,<:Any,<:Derivative,<:FEDVR{T}}
const SecondDerivative{T} = Mul{<:Tuple{<:Any,<:Any,<:Any},<:Tuple{<:QuasiAdjoint{T,<:Derivative{T}},<:Derivative{T},<:FEDVR{T}}}

const FirstOrSecondDerivative{T} = Union{FirstDerivative{T},SecondDerivative{T}}

order(::FirstDerivative) = 1
order(::SecondDerivative) = 2

function copyto!(dest::Mul2{<:Any,<:Any,<:FEDVR{T}},
                 M::FirstOrSecondDerivative{T}) where T
    S = last(M.factors)
    S′, A = dest.factors
    x = S′.t

    derop!(A, S, order(M))
    display(A)

    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))
    S == S′ || throw(ArgumentError("Cannot multiply incompatible FEDVRs"))

    dest
end

function similar(M::FirstOrSecondDerivative{T}, ::Type{T}) where T
    B = last(M.factors)
    Mul(B, Matrix(undef, B))
end

materialize(M::FirstOrSecondDerivative{T}) where T =
    copyto!(similar(M, eltype(M)), M)

export FEDVR

end # module
