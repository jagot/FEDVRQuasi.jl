module FEDVRQuasi

function __init__()
    @warn "The FEDVRQuasi.jl package has been deprecated in favour of JuliaApproximation/CompactBases.jl"
    nothing
end

import Base: axes, size, ==, getindex, checkbounds, copyto!, similar, diff, show
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: Basis, ℵ₁, @simplify

using QuasiArrays
import QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray,
    PInvQuasiMatrix, InvQuasiMatrix, BroadcastQuasiArray, SubQuasiArray

using BandedMatrices

using IntervalSets

using LazyArrays
using FillArrays

using LinearAlgebra
import LinearAlgebra: Matrix, dot

using FastGaussQuadrature, BlockBandedMatrices

using Printf

include("restricted_bases.jl")
include("gauss_lobatto_grid.jl")
include("fedvr.jl")
include("inner_products.jl")
include("operators.jl")
include("derivatives.jl")
include("densities.jl")

export FEDVR, Derivative, @elem, dot

end # module
