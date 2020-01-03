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
