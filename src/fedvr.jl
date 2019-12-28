# * Basis construction

struct FEDVR{T,R<:Real,O<:AbstractVector} <: Basis{T}
    t::AbstractVector{R}
    order::O
    i₀::Int
    t₀::R
    eiϕ::T
    x::Vector{T}
    wⁱ::Vector{Vector{R}}
    n::Vector{T}
    elems::Vector{UnitRange{Int}}
    function FEDVR(t::AbstractVector{R}, order::O; t₀::R=zero(R), ϕ::R=zero(R)) where {R<:Real,O<:AbstractVector}
        @assert length(order) == length(t)-1
        @assert all(order .> 1)

        i₀,eiϕ,T = if ϕ ≠ zero(R)
            findfirst(tt -> tt ≥ t₀, t),exp(im*ϕ),complex(R)
        else
            1,one(R),R
        end
        i₀ === nothing && throw(ArgumentError("Complex scaling starting point outside grid $(t)"))
        t₀ = t[i₀]

        x = Vector{Vector{T}}()
        w = Vector{Vector{T}}()
        for i in eachindex(order)
            xw = element_grid(order[i], t[i], t[i+1], t₀, i ≥ i₀ ? eiϕ : one(T))
            push!(x, xw[1])
            push!(w, xw[2])
        end

        rot = (i,v) -> i ≥ i₀ ? eiϕ*v : v
        n = [one(T) ./ .√(rot(i,wⁱ)) for (i,wⁱ) in enumerate(w)]
        for i in 1:length(order)-1
            n[i][end] = n[i+1][1] = one(T) ./ √(rot(i,w[i][end]) + rot(i+1,w[i+1][1]))
        end

        X = vcat(x[1], [x[i][2:end] for i in 2:length(order)]...)
        N = vcat(n[1], [n[i][2:end] for i in 2:length(order)]...)

        elems = [1:order[1]]

        l = order[1]
        for i = 2:length(order)
            l′ = l+order[i]-1
            push!(elems, l:l′)
            l = l′
        end

        new{T,R,O}(t,order,i₀,t₀,eiϕ,X,[w...],N,elems)
    end
end

FEDVR(t::AbstractVector{T},order::Integer; kwargs...) where T =
    FEDVR(t, Fill(order,length(t)-1); kwargs...)

const RestrictedFEDVR{T} = Union{RestrictedBasis{<:FEDVR{T}},<:RestrictedQuasiArray{<:Any,<:Any,<:FEDVR{T}}}
const AdjointRestrictedFEDVR{T} = Union{AdjointRestrictedBasis{<:FEDVR{T}},<:AdjointRestrictedQuasiArray{<:Any,<:Any,<:FEDVR{T}}}

const FEDVROrRestricted{T} = BasisOrRestricted{<:FEDVR{T}}
const AdjointFEDVROrRestricted{T} = AdjointBasisOrRestricted{<:FEDVR{T}}

# * Properties

axes(B::FEDVR) = (Inclusion(first(B.t)..last(B.t)), Base.OneTo(length(B.x)))
size(B::FEDVR) = (ℵ₁, length(B.x))
size(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) = (ℵ₁, length(B.args[2].data))
==(A::FEDVR,B::FEDVR) = A.t == B.t && A.order == B.order
==(A::FEDVROrRestricted,B::FEDVROrRestricted) = unrestricted_basis(A) == unrestricted_basis(B)

order(B::FEDVR) = B.order
# This assumes that the restriction matrices do not remove blocks
order(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) = order(B.args[1])

nel(B::FEDVR) = length(order(B))
element_boundaries(B::FEDVR) = vcat(1,1 .+ cumsum(B.order .- 1))

complex_rotate(x,B::FEDVR{T}) where {T<:Real} = x
complex_rotate(x,B::FEDVR{T}) where {T<:Complex} = x < B.t₀ ? x : B.t₀ + (x-B.t₀)*B.eiϕ

macro elem(B,v,i)
    :(@view($(esc(B)).$v[$(esc(B)).elems[$(esc(i))]]))
end

function show(io::IO, B::FEDVR{T}) where T
    write(io, "FEDVR{$(T)} basis with $(nel(B)) elements on $(axes(B,1).domain)")
    if T <: Complex
        rot = @printf(io, " with %s @ %.2f°", B.t₀ <= first(B.t) ? "ICS" : "ECS", rad2deg(angle(B.eiϕ)))
        B.t₀ > first(B.t) && @printf(io, " starting at %.2g", B.t₀)
    end
end

function show(io::IO, B::RestrictedQuasiArray{T,2,FEDVR{T}}) where T
    B′,restriction = B.args
    a,b = restriction_extents(restriction)
    N = length(B′.x)
    show(io, B′)
    write(io, ", restricted to basis functions $(1+a)..$(N-b) $(a>0 || b>0 ? "⊂" : "⊆") 1..$(N)")
end

function block_structure(B::FEDVR)
    if length(B.order) > 1
        bs = o -> o > 2 ? [o-2,1] : [1]
        Vector{Int}(vcat(B.order[1]-1,1,
                         vcat([bs(B.order[i]) for i = 2:length(B.order)-1]...),
                         B.order[end]-1))
    else
        [B.order[1]]
    end
end

function restriction_extents(restriction::RestrictionMatrix)
    a = restriction.l
    b = restriction.raxis.stop - restriction.data.axes[2].stop - a
    a,b
end

restriction_extents(B::FEDVR) = 0,0
restriction_extents(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    restriction_extents(B.args[2])

function block_structure(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B,restriction = B.args
    bs = block_structure(B)
    a,b = restriction_extents(restriction)

    a ≤ bs[1] && b ≤ bs[end] ||
        throw(ArgumentError("Cannot restrict basis beyond first/last block"))
    bs[1] -= a
    bs[end] -= b
    bs
end

function block_bandwidths(B::Union{FEDVR,RestrictedQuasiArray{<:Any,2,<:FEDVR}}, rows::Vector{<:Integer})
    nrows = length(rows)
    if nrows > 1
        bw = o -> o > 2 ? [2,1] : [1]
        bws = bw.(order(B))
        l = vcat(1,bws[2:end]...)
        u = vcat(reverse.(bws[1:end-1])...,1)
        length(l) < nrows && (l = vcat(l,0))
        length(u) < nrows && (u = vcat(0,u))
        l,u
    else
        [0],[0]
    end
end

locs(B::FEDVR) = B.x

rlocs(B::FEDVR{<:Real}) = locs(B)

function rlocs(B::FEDVR{T}) where T
    R = real(T)
    nx = length(B.x)
    x = Vector{R}(undef, nx)
    ii = 1
    for i in eachindex(B.order)
        xw = element_grid(B.order[i], B.t[i], B.t[i+1], B.t[1], one(R))
        copyto!(view(x, ii:nx), xw[1])
        ii += length(xw[1])-1
    end
    x
end

function locs(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B′,restriction = B.args
    a,b = FEDVRQuasi.restriction_extents(restriction)
    B′.x[1+a:end-b]
end

function rlocs(B::RestrictedQuasiArray{<:Any,2,<:FEDVR})
    B′,restriction = B.args
    a,b = FEDVRQuasi.restriction_extents(restriction)
    rlocs(B′)[1+a:end-b]
end

IntervalSets.leftendpoint(B::FEDVR) = B.x[1]
IntervalSets.rightendpoint(B::FEDVR) = B.x[end]

IntervalSets.leftendpoint(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    leftendpoint(B.args[1])
IntervalSets.rightendpoint(B::RestrictedQuasiArray{<:Any,2,<:FEDVR}) =
    rightendpoint(B.args[1])

# * Basis functions

function getindex(B::FEDVR{T}, x::Real, i::Integer, m::Integer) where T
    (x < B.t[i] || x > B.t[i+1]) && return zero(T)
    xⁱ = @elem(B,x,i)
    χ = @elem(B,n,i)[m]
    x′ = complex_rotate(x, B)
    for j = 1:B.order[i]
        j == m && continue
        χ *= (x′ - xⁱ[j])/(xⁱ[m]-xⁱ[j])
    end
    χ
end

checkbounds(B::FEDVR{T}, x::Real, k::Integer) where T =
    x ∈ axes(B,1) && 1 ≤ k ≤ size(B,2) || throw(BoundsError())

@inline function getindex(B::FEDVR{T}, x::Real, k::Integer) where T
    # @boundscheck checkbounds(B, x, k) # Slow
    i = 1
    m = k
    while i < length(B.order) && m > B.order[i]
        m -= B.order[i] - 1
        i += 1
    end
    x < B.t[i] && return zero(T)
    if x > B.t[i+1]
        if i < length(B.t)-1 && m == B.order[i]
            i += 1
            m = 1
        else
            return zero(T)
        end
    end
    B[x,i,m]
end

# @inline function Base.getindex(B::RestrictedBasis{<:FEDVR{T}}, x::Real, k::Integer) where {T}
#     B′,restriction = B.args
#     B′[x,k+restriction.l]
# end

# * Types

const FEDVRArray{T,N,B<:FEDVROrRestricted} = MulQuasiArray{T,N,<:Tuple{B,<:AbstractArray{T,N}}}
const FEDVRVector{T,B<:FEDVROrRestricted} = FEDVRArray{T,1,B}
const FEDVRMatrix{T,B<:FEDVROrRestricted} = FEDVRArray{T,2,B}
const FEDVRVecOrMat{T,B<:FEDVROrRestricted} = Union{FEDVRVector{T,B},FEDVRMatrix{T,B}}

const AdjointFEDVRArray{T,N,B<:FEDVROrRestricted} = MulQuasiArray{T,<:Any,<:Tuple{<:Adjoint{T,<:AbstractArray{T,N}},
                                                                                  <:QuasiAdjoint{T,<:B}}}
const AdjointFEDVRVector{T,B<:FEDVROrRestricted} = AdjointFEDVRArray{T,1,B}
const AdjointFEDVRMatrix{T,B<:FEDVROrRestricted} = AdjointFEDVRArray{T,2,B}
const AdjointFEDVRVecOrMat{T,B<:FEDVROrRestricted} = Union{AdjointFEDVRVector{T,B},AdjointFEDVRMatrix{T,B}}

# * Mass matrix

@simplify function *(Ac::QuasiAdjoint{<:Any,<:FEDVR}, B::FEDVR)
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))
    I
end

# A & B restricted
function materialize(M::Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix},
                                          <:QuasiAdjoint{<:Any,<:FEDVR{T}},
                                          <:FEDVR{T},
                                          <:RestrictionMatrix}}) where T
    restAc,Ac,B,restB = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply incompatible FEDVR expansions"))

    # This is mainly for type-stability; it would be trivial to
    # generate the proper restriction matrix from the combination of
    # two differently restricted bases, but we would like to have
    # UniformScaling as the result if they are equal, and this has
    # higher priority. On the other hand, you typically only compute
    # the mass matrix in the beginning of the calculation, and thus
    # type-instability is not a big problem, so this behaviour may
    # change in the future.
    restAc' == restB ||
        throw(ArgumentError("Non-equal restriction matrices not supported"))

    I
end

# A unrestricted, B restricted
materialize(M::Mul{<:Any,<:Tuple{UniformScaling{Bool},<:RestrictionMatrix}}) = M.args[2]
# A restricted, B unrestricted
materialize(M::Mul{<:Any,<:Tuple{<:Adjoint{<:Any,<:RestrictionMatrix},UniformScaling{Bool}}}) = M.args[1]

# * Basis inverses

@simplify function *(A⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}},
                     B::FEDVR)
    A = parent(A⁻¹)
    A == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    I
end

@simplify function *(A::FEDVR,
                     B⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}})
    B = parent(B⁻¹)
    A == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    I
end

@simplify function *(A⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}},
                     v::FEDVRArray)
    A = parent(A⁻¹)
    B,c = v.args
    A == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    c
end

@simplify function *(v::AdjointFEDVRArray,
                     B⁻¹::PInvQuasiMatrix{<:Any,<:Tuple{<:FEDVR}})
    c,Ac = v.args
    B = parent(B⁻¹)
    parent(Ac) == B || throw(ArgumentError("Cannot multiply basis with inverse of other basis"))
    c
end

# * Function interpolation

function Base.:(\ )(B::FEDVR{T}, f::BroadcastQuasiArray) where T
    axes(f,1) == axes(B,1) ||
        throw(DimensionMismatch("Function on $(axes(f,1).domain) cannot be interpolated over basis on $(axes(B,1).domain)"))

    v = zeros(T, size(B,2))
    for i ∈ 1:nel(B)
        @. v[B.elems[i]] += B.wⁱ[i]*f[@elem(B,x,i)]
    end
    v .*= B.n
    v
end

function Base.:(\ )(B::RestrictedQuasiArray{T,2,FEDVR{T}}, f::BroadcastQuasiArray) where T
    axes(f,1) == axes(B,1) ||
        throw(DimensionMismatch("Function on $(axes(f,1).domain) cannot be interpolated over basis on $(axes(B,1).domain)"))

    B′,restriction = B.args
    a,b = restriction_extents(restriction)

    n = size(B′,2)
    v = zeros(T, size(B,2))
    for i ∈ 1:nel(B′)
        sel = B′.elems[i]
        # Find which basis functions of finite-element `i` should be
        # evaluated.
        subsel = if 1+a<sel[1] && n-b > sel[end]
            # Element is completely within the restricted basis.
            Colon()
        else
            # Element straddles restriction (we don't allow
            # restrictions that would throw away an entire
            # finite-element); find subset of functions that are
            # within the restriction.
            s = min(max(1+a,sel[1]),sel[end])
            e = max(min(n-b,sel[end]),sel[1])
            findfirst(isequal(s),sel):findfirst(isequal(e),sel)
        end

        @. v[sel[subsel] .- a] += @view((B′.wⁱ[i]*f[@elem(B′,x,i)])[subsel])
    end
    v .*= @view(B′.n[1+a:end-b])
    v
end

