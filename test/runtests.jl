using FEDVRQuasi
import FEDVRQuasi: nel, complex_rotate
using IntervalSets
using ContinuumArrays
import ContinuumArrays: ℵ₁, Inclusion
using LinearAlgebra
using BlockBandedMatrices
using LazyArrays
import LazyArrays: ⋆
using Test

@testset "Simple tests" begin
    B = FEDVR(range(0,stop=20,length=71), 10)
    C = FEDVR(range(0,stop=20,length=71), 10)

    @test B == C
    @test axes(B) == (Inclusion(0..20), 1:631)
    @test size(B) == (ℵ₁, 631)

    @test nel(B) == 70

    @test B[0.0,1] ≠ 0.0
    @test B[20.0,end] ≠ 0.0

    @testset "Restricted bases" begin
        B̃ = B[:, 2:end-1]
        @test axes(B̃) == (Inclusion(0..20), 1:629)
        @test size(B̃) == (ℵ₁, 629)

        @test B̃[0.0,1] == 0.0
        @test B̃[20.0,end] == 0.0
    end
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
    @test occursin("FEDVR{Float64} basis with 70 elements on 0.0..20.0", string(B))
    @test occursin("FEDVR{Complex{Float64}} basis with 70 elements on 0.0..20.0 with ECS @ 60.00° starting at 10", string(C))
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

@testset "Mass matries" begin
    t = range(0,stop=20,length=5)
    @testset "Order 2" begin
        R = FEDVR(t, 2)
        R̃ = R[:,2:end-1]

        n = size(R,2)

        d = R'R
        @test d isa Diagonal
        @test size(d) == (n,n)

        d̃ = R̃'R̃
        @test d isa Diagonal
        @test size(d̃) == (n-2,n-2)
    end
    @testset "Higher order" begin
        R = FEDVR(t, 5)
        R̃ = R[:,2:end-1]

        n = size(R,2)

        d = R'R
        @test d isa BandedBlockBandedMatrix
        @test size(d) == (n,n)

        d̃ = R̃'R̃
        @test d isa BandedBlockBandedMatrix
        @test size(d̃) == (n-2,n-2)
    end
end

@testset "Scalar operators" begin
    B = FEDVR(1.0:7, 4)
    B̃ = B[:,2:end-1]

    V = Matrix(r -> r, B)
    @test diag(V) == B.x

    Ṽ = Matrix(r -> r, B̃)
    @test diag(Ṽ) == B.x[2:end-1]
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
        B̃ = B[:,2:end-1]
        D = Derivative(axes(B,1))

        BD = B'⋆D
        BDD = B'⋆D'⋆D

        # This should hold, regardless of whether complex scaling is
        # employed or not.
        @test BD⋆B isa FEDVRQuasi.FirstDerivative
        @test BD⋆B isa FEDVRQuasi.LazyFirstDerivative
        @test BDD⋆B isa FEDVRQuasi.SecondDerivative
        @test BDD⋆B isa FEDVRQuasi.LazySecondDerivative

        @test B̃' ⋆ D ⋆ B̃ isa FEDVRQuasi.FirstDerivative
        @test B̃' ⋆ D ⋆ B̃ isa FEDVRQuasi.LazyRestrictedFirstDerivative
        @test B̃' ⋆ D' ⋆ D ⋆ B̃ isa FEDVRQuasi.SecondDerivative
        @test B̃' ⋆ D' ⋆ D ⋆ B̃ isa FEDVRQuasi.LazyRestrictedSecondDerivative

        @test BD*B == B'*D*B
        @test BDD*B == B'*D'*D*B
    end
end

@testset "Materialize derivatives" begin
    B = FEDVR(1.0:7, 4)
    D = Derivative(axes(B,1))

    ∇ = B'*D*B
    ∇² = B'*D'*D*B

    A′ = Matrix(undef, B)
    A′′ = Matrix(undef, B)

    FEDVRQuasi.derop!(A′, B, 1)
    FEDVRQuasi.derop!(A′′, B, 2)

    @test ∇ isa BlockSkylineMatrix
    @test ∇² isa BlockSkylineMatrix

    @test ∇ == A′
    @test ∇² == A′′
end

@testset "Derivatives in restricted bases" begin
    t = range(0,stop=20,length=5)
    for (order,sel) in [(4,2:12),(4,1:13),([5,4,2,2],5:9),([5,4,3,2],5:10),
                        (2,2:3),(2,2:4),(2,2:5),(2,1:5)]
        R = FEDVR(t, order)
        R̃ = R[:,sel]

        D = Derivative(axes(R̃,1))

        expT = (any(order .> 2) ? BlockSkylineMatrix : Tridiagonal)

        ∇ = R' * D * R
        ∇̃ = R̃' * D * R̃

        @test ∇ isa expT
        @test ∇̃ isa expT

        @test Matrix(∇)[sel,sel] == Matrix(∇̃)

        ∇² = R' * D' * D * R
        ∇̃² = R̃' * D' * D * R̃

        @test ∇² isa expT
        @test ∇̃² isa expT

        @test Matrix(∇²)[sel,sel] == Matrix(∇̃²)
    end
end

@testset "Projections" begin
    rₘₐₓ = 20
    R = FEDVR(range(0,stop=rₘₐₓ,length=11), 10)
    r = range(0,stop=rₘₐₓ,length=1001)
    χ = R*R[r,:]'

    fu = r -> r^2*exp(-r)
    u = R*dot(R, fu)
    @test norm(χ'u - fu.(r)) < 1e-6
    fv = r -> r^6*exp(-r)
    v = R*dot(R, fv)
    @test norm(χ'v - fv.(r)) < 1e-4
end

@testset "Densities" begin
    rₘₐₓ = 20
    R = FEDVR(range(0,stop=rₘₐₓ,length=11), 10)
    r = range(0,stop=rₘₐₓ,length=1001)
    χ = R*R[r,:]'

    fu = r -> r^2*exp(-r)
    u = R*dot(R, fu)

    fv = r -> r^6*exp(-r)
    v = R*dot(R, fv)

    w = u .* v
    fw = r -> fu(r)*fv(r)

    @test norm(χ'w - fw.(r)) < 2e-4

    y = R*rand(ComplexF64, size(R,2))
    y² = y .* y
    @test all(isreal.(y².applied.args[2]))
    @test all(y².applied.args[2] .== abs2.(y.applied.args[2]) .* R.n)

    @testset "Lazy densities" begin
        uv = u .⋆ v
        @test uv isa FEDVRQuasi.FEDVRDensity

        w′ = similar(u)
        copyto!(w′, uv)
        @test norm(χ'w′ - fw.(r)) < 2e-4

        uu = R*repeat(u.applied.args[2],1,2)
        vv = R*repeat(v.applied.args[2],1,2)
        uuvv = uu .⋆ vv
        ww′ = similar(uu)
        copyto!(ww′, uuvv)

        @test norm(χ'ww′ .- fw.(r)) < 2e-4

        yy = y .⋆ y
        @test yy isa FEDVRQuasi.FEDVRDensity
        wy = similar(y)
        copyto!(wy, yy)
        @test all(isreal.(wy.applied.args[2]))
        @test all(wy.applied.args[2] .== abs2.(y.applied.args[2]) .* R.n)
    end
end
