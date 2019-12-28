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
