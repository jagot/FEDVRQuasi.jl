# * Gauß–Lobatto grid

# https://github.com/JuliaLang/julia/pull/18777
lerp(a::T,b::T,t) where T = T(fma(t, b, fma(-t, a, a)))
lerp(a::R,b::R,t::C) where {R<:Real,C<:Complex} = lerp(a,b,real(t)) + im*lerp(a,b,imag(t))
lerp(a::C,b::C,t::R) where {R<:Real,C<:Complex} = lerp(real(a),real(b),t) + im*lerp(imag(a),imag(b),(t))

function element_grid(order, a::T, b::T, c::T=zero(T), eiϕ=one(T)) where T
    x,w = gausslobatto(order)
    c .+ lerp.(Ref(eiϕ*(a-c)), Ref(eiϕ*(b-c)), (x .+ 1)/2),(b-a)*w/2
end
