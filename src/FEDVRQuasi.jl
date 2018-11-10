module FEDVRQuasi

import Base: axes, size, ==, getindex, checkbounds, copyto!, similar
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: ℵ₁
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint

using IntervalSets

using LazyArrays
import LazyArrays: Mul2
using FillArrays


using FastGaussQuadrature, BlockBandedMatrices

# https://github.com/JuliaLang/julia/pull/18777
lerp(a::T,b::T,t) where T = T(fma(t, b, fma(-t, a, a)))

function element_grid(order, a::T, b::T) where T
    x,w = gausslobatto(order)
    lerp.(Ref(a),Ref(b), (x .+ 1)/2),(b-a)*w/2
end


struct FEDVR{T,O<:AbstractVector} <: AbstractQuasiMatrix{T}
    t::AbstractVector{T}
    order::O
    x::Vector{T}
    xⁱ::Vector{<:SubArray{T}}
    n::Vector{T}
    nⁱ::Vector{<:SubArray{T}}
    function FEDVR(t::AbstractVector{T}, order::O) where {T,O<:AbstractVector}
        @assert length(order) == length(t)-1
        @assert all(order .> 1)
        
        x,w = zip([element_grid(order[i], t[i], t[i+1])
                   for i in eachindex(order)]...)
        
        n = [one(T) ./ .√(wⁱ) for wⁱ in w]
        for i in 1:length(order)-1
            n[i][end] = n[i+1][1] = one(T) ./ √(w[i][end]+w[i+1][1])
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
        new{T,O}(t,order,X,xⁱ,N,nⁱ)
    end
end

FEDVR(t::Vector{T},order::Integer) where T = FEDVR(t, Fill(order,length(t)-1))

axes(B::FEDVR) = (first(B.t)..last(B.t), Base.OneTo(length(B.x)))
size(B::FEDVR) = (ℵ₁, length(B.x))
==(A::FEDVR,B::FEDVR) = A.t == B.t && A.order == B.order

function getindex(B::FEDVR{T}, x::Real, i::Integer, m::Integer) where T
    (x < B.t[i] || x > B.t[i+1]) && return zero(T)
    xⁱ = B.xⁱ[i]
    χ = B.nⁱ[i][m]
    for j = 1:B.order[i]
        j == m && continue
        χ *= (x - xⁱ[j])/(xⁱ[m]-xⁱ[j])
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

## Mass matrix
function materialize(M::Mul2{<:Any,<:Any,<:QuasiAdjoint{<:Any,<:FEDVR{T}},<:FEDVR{T}}) where T
    Ac, B = M.factors
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    Diagonal(ones(T, size(A,2)))
end

export FEDVR

end # module
