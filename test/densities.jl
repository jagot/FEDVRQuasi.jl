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
