import ..AbstractCostFunction, ..AbstractGradientFunction

"""
    functions(λ̄, x̄L, x̄R, σ̄)

A steep exponential penalty for each parameter exceeding its bounds.

# Arguments
- `λ̄`: vector of weights for each penalty
        Set `λ̄[i]=0` to skip penalties for the ith parameter.

- `x̄L`: vector of lower bounds for each parameter
- `x̄R`: vector of upper bounds for each parameter
- `σ̄`: vector of scalings (smaller=steeper) for each penalty

# Returns
- `f`: the cost function
- `g`: the gradient function

"""
function functions(λ̄, x̄L, x̄R, σ̄)
    f = CostFunction(λ̄, x̄L, x̄R, σ̄)
    g = GradientFunction(f)
    return f, g
end

struct CostFunction{F<:AbstractFloat} <: AbstractCostFunction
    λ̄::Vector{F}
    x̄L::Vector{F}
    x̄R::Vector{F}
    σ̄::Vector{F}

    function CostFunction(
        λ̄::AbstractVector,
        x̄L::AbstractVector,
        x̄R::AbstractVector,
        σ̄::AbstractVector,
    )
        F = promote_type(Float16, eltype(λ̄), eltype(x̄L), eltype(x̄R), eltype(σ̄))

        return new{F}(
            convert(Array{F}, λ̄),
            convert(Array{F}, x̄L),
            convert(Array{F}, x̄R),
            convert(Array{F}, σ̄),
        )
    end
end

function (f::CostFunction)(x̄::AbstractVector)
    χ̄L = (f.x̄L .- x̄) ./ f.σ̄
    χ̄R = (x̄ .- f.x̄R) ./ f.σ̄
    λ̄ = f.λ̄

    value = 0
    for i in eachindex(x̄)
        λ, χL, χR = λ̄[i], χ̄L[i], χ̄R[i]
        λ > 0 && χL > 0 && (value += λ * (exp( log(2) * χL^2 ) - 1))
        λ > 0 && χR > 0 && (value += λ * (exp( log(2) * χR^2 ) - 1))
    end
    return value
end

struct GradientFunction{F<:AbstractFloat} <: AbstractGradientFunction
    f::CostFunction{F}
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    # TODO (mid): THESE ARE ALLOCATIONS!!!!
    χ̄L = (g.f.x̄L .- x̄) ./ g.f.σ̄
    χ̄R = (x̄ .- g.f.x̄R) ./ g.f.σ̄
    λ̄ = g.f.λ̄

    ∇f̄ .= 0
    for i in eachindex(x̄)
        λ, σ, χL, χR = λ̄[i], g.f.σ̄[i], χ̄L[i], χ̄R[i]
        λ > 0 && χL > 0 && (∇f̄[i] -= (2log(2)) * λ * χL / σ * exp( log(2) * χL^2 ))
        λ > 0 && χR > 0 && (∇f̄[i] += (2log(2)) * λ * χR / σ * exp( log(2) * χR^2 ))
    end
    return ∇f̄
end
