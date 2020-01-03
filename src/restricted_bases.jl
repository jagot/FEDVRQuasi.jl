# * Auxilliary type definitions for restricted bases

const RestrictionMatrix = BandedMatrix{<:Int, <:FillArrays.Ones}
const RestrictedQuasiArray{T,N,B<:Basis} = SubQuasiArray{T,N,B}
const AdjointRestrictedQuasiArray{T,N,B<:Basis} = QuasiAdjoint{T,<:RestrictedQuasiArray{T,N,B}}

const BasisOrRestricted{B<:Basis} = Union{B,<:RestrictedQuasiArray{<:Any,<:Any,<:B}}
const AdjointBasisOrRestricted{B<:Basis} = Union{<:QuasiAdjoint{<:Any,B},<:AdjointRestrictedQuasiArray{<:Any,<:Any,<:B}}

unrestricted_basis(R::AbstractQuasiMatrix) = R
unrestricted_basis(R::RestrictedQuasiArray) = parent(R)

restriction_extents(::Basis) = 0,0
function restriction_extents(B̃::RestrictedQuasiArray)
    B = parent(B̃)
    a,b = B̃.indices[2][[1,end]]
    a-1,size(B,2)-b
end

restriction(B̃::RestrictedQuasiArray) = last(LazyArrays.arguments(B̃))
