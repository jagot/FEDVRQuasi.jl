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

        @test leftendpoint(B̃) == 0
        @test rightendpoint(B̃) == 20

        @test B̃[0.0,1] == 0.0
        @test B̃[20.0,end] == 0.0

        @test B isa FEDVR
        @test B isa FEDVRQuasi.FEDVROrRestricted
        @test B' isa QuasiAdjoint{<:Any,<:FEDVR}
        @test B' isa FEDVRQuasi.AdjointFEDVROrRestricted
        @test !(B̃ isa FEDVR)
        @test B̃ isa FEDVRQuasi.RestrictedFEDVR
        @test B̃ isa FEDVRQuasi.FEDVROrRestricted
        @test B̃' isa FEDVRQuasi.AdjointRestrictedFEDVR
        @test B̃' isa FEDVRQuasi.AdjointFEDVROrRestricted

        @test unrestricted_basis(B) == B
        @test unrestricted_basis(B̃) == B

        @test restriction_extents(B) == (0,0)
        @test restriction_extents(B̃) == (1,1)

        @test locs(B̃) == locs(B)[2:end-1]
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
    bs = M.block_sizes
    @test length(bs.l) == length(l)+1
    @test length(bs.u) == length(u)+1

    # TODO Test block sizes

    @test all(bs.l[1:end-2] .== l[1:end-1])
    length(l) > 0 && @test bs.l[end-1] ≥ l[end]
    @test all(bs.u[3:end] .== u[2:end])
    length(u) > 0 && @test bs.u[2] ≥ u[1]
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

@testset "Mass matrices and inverses" begin
    t = range(0,stop=20,length=5)
    @testset "Order $k" for k ∈ [2,5]
        R = FEDVR(t, k)
        R̃ = R[:,2:end-1]

        m = size(R,2)
        n = size(R̃,2)

        @testset "Mass matrices" begin
            d = R'R
            @test d === I

            d̃ = R̃'R̃
            @test d̃ === I

            @test R'R̃ == (R̃'R)' == BandedMatrix((-1 => ones(Int,n),), (m,n), (1,-1))

            # This behaviour is subject to change.
            @test_throws DimensionMismatch R̃'R[:,1:end-1]
        end

        @testset "Inverses" begin
            R⁻¹ = pinv(R)
            # R̃⁻¹ = pinv(R̃)

            @test R⁻¹*R === I
            @test R*R⁻¹ === I

            cu = rand(size(R,2))
            cv = rand(size(R,2))
            cuv = [cu cv]

            u = R*cu
            v = R*cv
            uv = R*cuv

            @test R⁻¹*u === cu
            @test R⁻¹*v === cv
            @test R⁻¹*uv === cuv

            ut = u'
            # Fails with: ERROR: MethodError: no method matching axes(::UniformScaling{Float64}, ::Int64)
            # @test ut*R⁻¹' === ut.args[1]

            @warn "Need to implement/test basis inverses for restricted bases"
        end
    end
end

@testset "Inner products" begin
    @testset "Vectors" begin
        t = range(0,stop=1,length=11)
        B = FEDVR(t, 4)
        B̃ = B[:,1:5]

        uv = ones(size(B,2))
        vv = ones(size(B̃,2))

        @testset "Direct inner products" begin
            u = B*uv
            v = B̃*vv

            @test u'v == 5
            @test (uv'*(B'B̃)*vv)[1] == 5
            @test norm(v) ≈ √5
        end

        @testset "Lazy inner products" begin
            lu = B ⋆ uv
            lv = B̃ ⋆ vv

            lazyip = lu' ⋆ lv

            @test lazyip isa FEDVRQuasi.LazyFEDVRInnerProduct
            @test materialize(lazyip) == 5

            ϕ = B ⋆ rand(ComplexF64, size(B,2))
            normalize!(ϕ)
            @test materialize(ϕ'⋆ϕ) ≈ 1.0
        end
    end

    @testset "Matrices" begin
        t = range(-1.0,stop=1.0,length=5)
        R = FEDVR(t, 4)[:,2:end-1]

        n = 3
        Φ = rand(ComplexF64, size(R,2), n)
        for i = 1:n
            # Orthogonalize against all previous vectors
            for j = 1:i-1
                c = Φ[:,j]'Φ[:,i]/(Φ[:,j]'Φ[:,j])
                Φ[:,i] -= c*Φ[:,j]
            end
            Φ[:,i] /= norm(R*Φ[:,i])
        end

        @testset "Direct inner products" begin
            Φ̂ = R*Φ
            Φ̂'Φ̂
            @test Φ̂'Φ̂ ≈ I
        end

        @testset "Lazy inner products" begin
            Φ̃ = R ⋆ Φ
            lzip = Φ̃' ⋆ Φ̃
            @test lzip isa FEDVRQuasi.LazyFEDVRInnerProduct
            @test size(lzip) == (n,n)
            @test materialize(lzip) ≈ I
        end
    end
end

@testset "Scalar operators" begin
    B = FEDVR(1.0:7, 4)
    B̃ = B[:,2:end-1]
    r = axes(B,1)

    V = B'QuasiDiagonal(identity.(r))*B
    @test V == Diagonal(B.x)

    Ṽ = apply(*, B̃', QuasiDiagonal(identity.(r)), B̃)
    @test Ṽ == Diagonal(B.x[2:end-1])
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

        @test applied(*, B', D, B) isa FEDVRQuasi.FirstDerivative
        @test applied(*, B', D, B) isa FEDVRQuasi.FlatFirstDerivative
        @test applied(*, B', D', D, B) isa FEDVRQuasi.SecondDerivative
        @test applied(*, B', D', D, B) isa FEDVRQuasi.FlatSecondDerivative

        @test applied(*, B̃', D, B̃) isa FEDVRQuasi.FirstDerivative
        @test applied(*, B̃', D, B̃) isa FEDVRQuasi.FlatRestrictedFirstDerivative
        @test applied(*, B̃', D', D, B̃) isa FEDVRQuasi.SecondDerivative
        @test applied(*, B̃', D', D, B̃) isa FEDVRQuasi.FlatRestrictedSecondDerivative

        @test BD*B == B'*D*B
        @test_broken BDD*B == B'*D'*D*B

        @test B'D*B == B'*D*B
        @test B'D'D*B == B'*D'*D*B
    end
end

@testset "Materialize derivatives" begin
    @testset "$style materialization" for (style,first_derivative,second_derivative) in [
        ("Infix", (B,D) -> B'*D*B, (B,D) -> B'*D'*D*B),
        ("applied", (B,D) -> materialize(applied(*, B', D, B)),
         (B,D) -> materialize(applied(*, B', D', D, B))),
        ("apply", (B,D) -> apply(*, B', D, B),
         (B,D) -> apply(*, B', D', D, B))
    ]
        B = FEDVR(1.0:7, 4)
        D = Derivative(axes(B,1))

        ∇ = first_derivative(B,D)
        ∇² = second_derivative(B,D)

        A′ = Matrix(undef, B)
        A′′ = Matrix(undef, B)

        FEDVRQuasi.derop!(A′, B, 1)
        FEDVRQuasi.derop!(A′′, B, 2)

        @test ∇ isa BlockSkylineMatrix
        @test ∇² isa BlockSkylineMatrix

        @test ∇ == A′
        @test ∇² == A′′
    end
end

@testset "Derivatives in restricted bases" begin
    @testset "$style materialization" for (style,first_derivative,second_derivative) in [
        ("Infix", (B,D) -> B'*D*B, (B,D) -> B'*D'*D*B),
        ("applied", (B,D) -> materialize(applied(*, B', D, B)),
         (B,D) -> materialize(applied(*, B', D', D, B))),
        ("apply", (B,D) -> apply(*, B', D, B),
         (B,D) -> apply(*, B', D', D, B))
    ]
        for (order,sel,N) in [(4,2:12,5),(4,1:13,5),([5,4,2,2],5:9,5),([5,4,3,2],5:10,5),
                              (2,2:3,5),(2,2:4,5),(2,2:5,5),(2,1:5,5),
                              (5,2:4,2)]
            t = range(0,stop=20,length=N)

            R = FEDVR(t, order)
            R̃ = R[:,sel]

            D = Derivative(axes(R̃,1))

            expT = (any(order .> 2) ? BlockSkylineMatrix : Tridiagonal)

            ∇ = first_derivative(R, D)
            ∇̃ = first_derivative(R̃, D)

            @test ∇ isa expT
            @test ∇̃ isa expT

            @test Matrix(∇)[sel,sel] == Matrix(∇̃)

            ∇² = second_derivative(R, D)
            ∇̃² = second_derivative(R̃, D)

            @test ∇² isa expT
            @test ∇̃² isa expT

            @test Matrix(∇²)[sel,sel] == Matrix(∇̃²)
        end
    end
end

@testset "Function interpolation" begin
    rₘₐₓ = 20
    @testset "t₀ = $(t₀), ϕ = $(ϕ)" for (t₀,ϕ) in [(0.0,0.0)# ,
                                                   # (rₘₐₓ/2, π/3)
                                                   ]
        R = FEDVR(range(0,stop=rₘₐₓ,length=11), 10, t₀=t₀, ϕ=ϕ)
        R⁻¹ = pinv(R)
        r = axes(R,1)
        r̃ = range(0,stop=rₘₐₓ,length=1001)
        χ = R[r̃,:]

        fu = r -> r^2*exp(-r)
        fv = r -> r^6*exp(-8r)

        u = R*(R \ fu.(r))
        @test norm(χ * (R⁻¹*u) - fu.(r̃)) < 1e-6
        v = R*(R \ fv.(r))
        @test norm(χ * (R⁻¹*v) - fv.(r̃)) < 5e-5

        @testset "Restricted basis" begin
            R̃ = R[:,2:end-1]
            χ̃ = R̃[r̃,:]

            # # Actually, we want
            # ũ = R̃*(R̃ \ fu.(r))
            # # but R̃*... expands the coefficient vector by two elements
            # # (which are "hidden" by the restriction matrix), which
            # # causes a dimension mismatch below.
            ũ = R̃ \ fu.(r)
            # @test norm(χ̃ * ũ.args[2] - fu.(r̃)) < 5e-6
            @test norm(χ̃ * ũ - fu.(r̃)) < 2e-6
            # ṽ = R̃*(R̃ \ fv.(r))
            ṽ = R̃ \ fv.(r)
            # @test norm(χ̃ * ṽ.args[2] - fv.(r̃)) < 1e-4
            @test norm(χ̃ * ṽ - fv.(r̃)) < 5e-5

            h = r -> (r-rₘₐₓ/2)^2
            c = R̃ \ h.(r)
            # Vandermonde matrix to compare with; its generation is
            # costly, which is why we prefer to compute overlaps with
            # basis functions instead.
            rl = locs(R̃)
            V = R̃[rl,:]
            c̃ = V \ h.(rl)

            @test c ≈ c̃ atol=1e-13
        end
    end
end

@testset "Real locations" begin
    rₘₐₓ = 20
    R = FEDVR(range(0,stop=rₘₐₓ,length=11), 10, t₀=rₘₐₓ/2, ϕ=π/3)
    R′ = FEDVR(range(0,stop=rₘₐₓ,length=11), 10)
    @test norm(rlocs(R)-rlocs(R′)) == 0
end

include("derivative_accuracy_utils.jl")

@testset "Derivative accuracy" begin
    (f,g,h,a,b),s,e = derivative_test_functions(1.0), 1, 1

    Ns = ceil.(Int, 2 .^ range(5,stop=9,length=30))

    orders = 2:10
    slopes = zeros(length(orders),3)

    for (i,order) in enumerate(orders)
        hs,ϵg,ϵh,ϵh′,pg,ph,ph′ = compute_derivative_errors(a, b, Ns, order, s, e, f, g, h)
        slopes[i,:] = [pg ph ph′]
    end

    println("Derivative convergence rates:")
    pretty_table([orders slopes], ["Order", "pg", "ph", "ph′"])
    println()
end

@testset "Densities" begin
    @testset "Dirichlet1" begin
        rₘₐₓ = 20
        R = FEDVR(range(0,stop=rₘₐₓ,length=11), 10)
        r = axes(R,1)
        R⁻¹ = pinv(R)
        r̃ = range(0,stop=rₘₐₓ,length=1001)
        χ = R[r̃,:]

        fu = r -> r^2*exp(-r)
        u = R*(R\fu.(r))

        fv = r -> r^6*exp(-r)
        v = R*(R\fv.(r))

        w = u .* v
        fw = r -> fu(r)*fv(r)

        @test norm(χ * (R⁻¹*w) - fw.(r̃)) < 2e-4

        y = R*rand(ComplexF64, size(R,2))
        y² = y .* y
        @test all(isreal.(R⁻¹*y²))
        @test all(R⁻¹*y² .== abs2.(R⁻¹*y) .* R.n)

        @testset "Lazy densities" begin
            uv = u .⋆ v
            @test uv isa FEDVRQuasi.FEDVRDensity

            w′ = similar(u)
            copyto!(w′, uv)
            @test norm(χ * (R⁻¹*w′) - fw.(r̃)) < 2e-4

            uu = R*repeat(R⁻¹*u,1,2)
            vv = R*repeat(R⁻¹*v,1,2)
            uuvv = uu .⋆ vv
            ww′ = similar(uu)
            copyto!(ww′, uuvv)

            @test norm(χ * (R⁻¹*ww′) .- fw.(r̃)) < 2e-4

            yy = y .⋆ y
            @test yy isa FEDVRQuasi.FEDVRDensity
            wy = similar(y)
            copyto!(wy, yy)
            @test all(isreal.(R⁻¹*wy))
            @test all(R⁻¹*wy .== abs2.(R⁻¹*y) .* R.n)
        end
    end

    @testset "Dirichlet0" begin
        f,g,h,a,b = derivative_test_functions(1.0)

        t = range(a,stop=b,length=20)
        R = FEDVR(t, 10)[:,2:end-1]
        r = axes(R,1)
        # R⁻¹ = pinv(R)

        r̃ = range(t[1],stop=t[end],length=1001)
        χ = R[r̃,:]

        u = R*(R\f.(r))
        v = R*(R\g.(r))

        w = u .* v
        fw = r -> f(r)*g(r)

        w′ = R*(R\fw.(r))

        # @test norm(R⁻¹*w - R⁻¹*w′) < 1e-15
        @test norm(w.args[2] - w′.args[2]) < 1e-15
    end
end
