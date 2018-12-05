using FEDVRQuasi
import FEDVRQuasi: nel, complex_rotate, FirstDerivative, SecondDerivative
using IntervalSets
using ContinuumArrays
import ContinuumArrays: ℵ₁
using LinearAlgebra
using BlockBandedMatrices
using LazyArrays
import LazyArrays: ⋆
using Test

@testset "Simple tests" begin
    B = FEDVR(range(0,stop=20,length=71), 10)
    C = FEDVR(range(0,stop=20,length=71), 10)

    @test B == C
    @test axes(B) == (0..20, 1:631)
    @test size(B) == (ℵ₁, 631)

    @test nel(B) == 70
end

@testset "Complex scaling" begin
    B = FEDVR(range(0,stop=20,length=71), 10)
    C = FEDVR(range(0,stop=20,length=71), 10, t₀=10.0, ϕ=π/3)
    @test C.t₀ ≥ 10.0
    @test_throws ArgumentError FEDVR(range(0,stop=20,length=71), 10, t₀=30.0, ϕ=π/3)
    @test complex_rotate(5, B) == 5
    @test complex_rotate(15, B) == 15
    @test complex_rotate(5, C) == 5
    @test complex_rotate(15, C) ≈ 10 + 5*exp(im*π/3)
end

@testset "Pretty printing" begin
    B = FEDVR(range(0,stop=20,length=71), 10)
    C = FEDVR(range(0,stop=20,length=71), 10, t₀=10.0, ϕ=π/3)
    io = IOBuffer()
    show(io, B)
    @test occursin("FEDVR{Float64} basis with 70 elements on 0.0..20.0", String(take!(io)))
    show(io, C)
    @test occursin("FEDVR{Complex{Float64}} basis with 70 elements on 0.0..20.0 with ECS @ 60.00° starting at 10", String(take!(io)))
end

@testset "Element access" begin
    B = FEDVR(range(0,stop=20,length=71), 10)
    @test first(@elem(B, x, 1)) == 0
    @test last(@elem(B, x, 1)) == first(@elem(B, x, 2))
    @test last(@elem(B, x, 69)) == first(@elem(B, x, 70))
    @test_throws BoundsError @elem(B, x, 71)
end

function test_block_structure(t, o, l, u)
    B = FEDVR(t, o)
    M = Matrix(undef, B)

    @test M isa BlockSkylineMatrix
    @test length(M.l) == length(l)+1
    @test length(M.u) == length(u)+1

    # TODO Test block sizes

    @test all(M.l[1:end-2] .== l[1:end-1])
    length(l) > 0 && @test M.l[end-1] ≥ l[end]
    @test all(M.u[3:end] .== u[2:end])
    length(u) > 0 && @test M.u[2] ≥ u[1]
end

@testset "Block structure" begin
    test_block_structure(1.0:2, [4], [], [])
    test_block_structure(1.0:7, [2,2,3,4,2,4], [1,1,2,1,2,1,1,1], [1,1,1,2,1,2,1,1])
    test_block_structure(1.0:8, [2,2,3,4,2,4,2], [1,1,2,1,2,1,1,2,1,1], [1,1,1,2,1,2,1,1,2,1])
    test_block_structure(range(0,1,length=7), 1 .+ (1:6), [1,2,1,2,1,2,1,2,1,1], [1,1,2,1,2,1,2,1,2,1])
    test_block_structure(range(0,1,length=7), reverse(1 .+ (1:6)), [1,2,1,2,1,2,1,2,1,1], [1,1,2,1,2,1,2,1,2,1])
    test_block_structure(range(0,1,length=7), 2 .+ (1:6), [1,2,1,2,1,2,1,2,1,1], [1,1,2,1,2,1,2,1,2,1])
    @test Matrix(undef, FEDVR(1.0:7, 2)) isa Tridiagonal
end

function test_blocks(f::Function, t, o)
    B = FEDVR(t, o)
    A = Matrix(undef, B)

    blocks = map(f, 1:FEDVRQuasi.nel(B))
    coords = vcat(1,1 .+ cumsum(B.order[1:end-1] .- 1))
    FEDVRQuasi.set_blocks!(f, A, B)

    for (i,(b,c)) in enumerate(zip(blocks,coords))
        sel = c .+ (0:size(b,1)-1)
        b′ = copy(A[sel,sel])
        i > 1 && (b′[1,1] -= blocks[i-1][end,end])
        i < length(blocks) && (b′[end,end] -= blocks[i+1][1,1])
        @test all(b′ .== b)
    end
end

@testset "Set blocks" begin
    test_blocks(1.0:3,[2,3]) do i
        i*ones(i+1,i+1)
    end
    begin
        o = [2,2,3,4,2,4]
        test_blocks(1.0:7,o) do i
            i*ones(o[i],o[i])
        end
    end
    begin
        o = 4 .+ (1:6)
        test_blocks(range(0,1,length=7), o) do i
            i*ones(o[i],o[i])
        end
    end
end

@testset "Lazy derivatives" begin
    for (t₀,ϕ) in [(1.0,0.0), (4.0,π/3)]
        B = FEDVR(1.0:7, 4, t₀=t₀, ϕ=ϕ)
        D = Derivative(axes(B,1))

        DB = D⋆B
        DDB = D'⋆D⋆B

        # This should hold, regardless of whether complex scaling is
        # employed or not.
        @test B'⋆DB isa FirstDerivative
        @test B'⋆DDB isa SecondDerivative

        @test B'DB == B'*D*B
        @test B'DDB == B'*D'*D*B
    end
end

@testset "Materialize derivatives" begin
    B = FEDVR(1.0:7, 4)
    D = Derivative(axes(B,1))

    ∇ = B'*D*B
    ∇² = B'D'D*B

    A′ = Matrix(undef, B)
    A′′ = Matrix(undef, B)

    FEDVRQuasi.derop!(A′, B, 1)
    FEDVRQuasi.derop!(A′′, B, 2)

    @test ∇ isa BlockSkylineMatrix
    @test ∇² isa BlockSkylineMatrix

    @test ∇ == A′
    @test ∇² == A′′
end

@testset "Projections" begin
    rₘₐₓ = 20
    R = FEDVR(range(0,stop=rₘₐₓ,length=11), 10)
    r = range(0,stop=rₘₐₓ,length=1001)
    χ = R[r,:]

    fu = r -> r^2*exp(-r)
    u = dot(R, fu)
    @test norm(χ*u - fu.(r)) < 1e-6
    fv = r -> r^6*exp(-r)
    v = dot(R, fv)
    @test norm(χ*v - fv.(r)) < 1e-4
end
