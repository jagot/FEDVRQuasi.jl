using FEDVRQuasi
import FEDVRQuasi: nel, complex_rotate, rlocs, locs,
    unrestricted_basis, restriction_extents
using IntervalSets
using QuasiArrays
using ContinuumArrays
import ContinuumArrays: ℵ₁, Inclusion
using LinearAlgebra
using BandedMatrices
using BlockBandedMatrices
using LazyArrays
import LazyArrays: materialize
using Test

include("basics.jl")
include("complex_scaling.jl")
include("block_structure.jl")
include("scalar_operators.jl")
include("inner_products.jl")
include("function_interpolation.jl")
include("derivatives.jl")
include("densities.jl")
