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
