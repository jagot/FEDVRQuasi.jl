using LinearAlgebra

using Test
using PrettyTables

function test_fedvr_derivatives(t::AbstractVector, order::Integer, s, e,
                                f::Function, g::Function, h::Function)
    R = FEDVR(t, order)
    if (s,e) != (0,0)
        n = size(R,2)
        R = R[:,(1+s):(n-e)]
    end

    D = Derivative(axes(R,1))

    ∇ = R' * D * R
    ∇² = R' * D' * D * R

    fv = R ⋅ f

    gv = ∇ * fv
    δgv = R ⋅ g - gv

    hv = ∇ * gv
    δhv = R ⋅ h - hv

    hv′ = ∇² * fv
    δhv′ = R ⋅ h - hv′

    R,fv,gv,hv,hv′,δgv,δhv,δhv′
end

function error_slope(loghs,ϵ)
    # To avoid the effect of round-off errors on the order
    # estimation.
    i = argmin(abs.(ϵ))

    ([loghs[1:i] ones(i)] \ log10.(abs.(ϵ[1:i])))[1]
end

function compute_derivative_errors(a, b, Ns, order::Integer, s::Integer, e::Integer,
                                   f::Function, g::Function, h::Function,
                                   verbosity=0)
    errors = map(Ns) do N
        t = range(a, stop=b, length=N)
        R,fv,gv,hv,hv′,δgv,δhv,δhv′ = test_fedvr_derivatives(t, order, s, e, f, g, h)
        [maximum(abs, δgv) maximum(abs, δhv) maximum(abs, δhv′)]
    end |> e -> vcat(e...)

    ϵg = errors[:,1]
    ϵh = errors[:,2]
    ϵh′ = errors[:,3]

    hs = 1.0 ./ Ns

    loghs = log10.(hs)
    pg = error_slope(loghs, ϵg)
    ph = error_slope(loghs, ϵh)
    ph′ = error_slope(loghs, ϵh′)

    verbosity > 0 &&
        pretty_table([hs errors], ["h", "δg [$(pg)]", "δh [$(ph)]", "δh′ [$(ph′)]"])

    # The convergence rate should be order - 1 (polynomial order =
    # polynomial degree + 1), but since the error slope fitting is a
    # bit error prone, we require that it is greater than order - 2.
    @test pg > order - 2
    # Since the approximation to h is calculated by computing ∇∇f, we
    # lose one order extra, compared to ∇²f.
    @test ph > order - 3
    @test ph′ > order - 2

    hs,ϵg,ϵh,ϵh′,pg,ph,ph′
end

function derivative_test_functions(d)
    a,b = √d*[-1,1]

    # The functions vanish at the boundaries; Dirichlet0 boundary
    # conditions. They /cannot/ be evaluated at the boundaries, since
    # that results in NaNs.
    f = x -> exp(-1/(d-x^2))
    g = x -> -2*exp(-1/(d-x^2))*x/((d-x^2)^2)
    h = x -> -2*exp(-1/(d-x^2))*(d^2 + 2*(d-1)*x^2-3x^4)/((d-x^2)^4)

    f,g,h,a,b
end
