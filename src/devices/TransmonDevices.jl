using Memoization: @memoize
import LinearAlgebra: I, mul!
using ...LinearAlgebraTools: List
import ...Parameters, ...LinearAlgebraTools, ...Signals, ...Devices

import ...TempArrays: array
const LABEL = Symbol(@__MODULE__)

@memoize Dict function bosonic_annihilator(::Type{F}, m::Int) where {F<:AbstractFloat}
    a = zeros(F, m, m)
    for i ∈ 1:m-1
        a[i,i+1] = √i
    end
    return a
end



abstract type AbstractTransmonDevice{F} <: Devices.LocallyDrivenDevice end

# THE INTERFACE TO IMPLEMENT

# Devices.nstates
# Devices.nqubits
resonancefrequency(::AbstractTransmonDevice, q::Int)::Real = error("Not Implemented")
anharmonicity(::AbstractTransmonDevice, q::Int)::Real = error("Not Implemented")

ncouplings(::AbstractTransmonDevice)::Int = error("Not Implemented")
couplingpair(::AbstractTransmonDevice, k::Int)::Devices.Quple = error("Not Implemented")
couplingstrength(::AbstractTransmonDevice, k::Int)::Real = error("Not Implemented")

# Devices.ndrives
# Devices.drivequbit
drivefrequency(::AbstractTransmonDevice, i::Int)::Real = error("Not Implemented")
drivesignal(::AbstractTransmonDevice, i::Int)::AbstractSignal = error("Not Implemented")

bindfrequencies(::AbstractTransmonDevice, ν̄::AbstractVector) = error("Not Implemented")


# THE INTERFACE ALREADY IMPLEMENTED

function Devices.ngrades(device::AbstractTransmonDevice)
    return 2 * Devices.ndrives(device)
end

function Devices.gradequbit(device::AbstractTransmonDevice, j::Int)
    return Devices.drivequbit(device, ((j-1) >> 1) + 1)
end

Devices.eltype_localloweringoperator(::AbstractTransmonDevice{F}) where {F} = F
function Devices.localloweringoperator(
    device::AbstractTransmonDevice{F},
    q::Int,
) where {F}
    return bosonic_annihilator(F, Devices.nstates(device, q))
end

Devices.eltype_qubithamiltonian(::AbstractTransmonDevice{F}) where {F} = F
function Devices.qubithamiltonian(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
    q::Int;
    result=nothing,
)
    a = ā[q]
    Im = Matrix(I, size(a))     # UNAVOIDABLE ALLOCATION?

    result === nothing && (result = Matrix{eltype(a)}(undef, size(a)))
    result .= 0
    result .-= (anharmonicity(device,q)/2)  .* Im       #       - δ/2    I
    result = LinearAlgebraTools.rotate!(a', result)     #       - δ/2   a'a
    result .+= resonancefrequency(device,q) .* Im       # ω     - δ/2   a'a
    result = LinearAlgebraTools.rotate!(a', result)     # ω a'a - δ/2 a'a'aa
    return result
end

Devices.eltype_staticcoupling(::AbstractTransmonDevice{F}) where {F} = F
function Devices.staticcoupling(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix{F}};
    result=nothing,
) where {F}
    shape = size(ā[1])  # NOTE: Not robust for empty devices.
    result === nothing && (result = Matrix{F}(undef, shape))

    result .= 0
    for pq in 1:ncouplings(device)
        g = couplingstrength(device, pq)
        p, q = couplingpair(device, pq)

        aTa = array(F, shape, LABEL)
        mul!(aTa, ā[p]', ā[q])

        result .+= g .* aTa
        result .+= g .* aTa'
    end
    return result
end

Devices.eltype_driveoperator(::AbstractTransmonDevice{F}) where {F} = Complex{F}
function Devices.driveoperator(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
    i::Int,
    t::Real;
    result=nothing,
)
    a = ā[Devices.drivequbit(device, i)]
    e = exp(im * drivefrequency(device, i) * t)
    Ω = drivesignal(device, i)(t)

    if result === nothing
        F = promote_type(eltype(a), eltype(e))  # Ω is no more complex than e.
        result = Matrix{F}(undef, size(a))
    end
    result .= 0

    result .+= (real(Ω) * e ) .* a
    result .+= (real(Ω) * e') .* a'

    if Ω isa Complex
        result .+= (imag(Ω) * im *e ) .* a
        result .+= (imag(Ω) * im'*e') .* a'
    end

    return result
end

Devices.eltype_gradeoperator(::AbstractTransmonDevice{F}) where {F} = Complex{F}
function Devices.gradeoperator(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
    j::Int,
    t::Real;
    result=nothing,
)
    i = ((j-1) >> 1) + 1
    a = ā[Devices.drivequbit(device, i)]
    e = exp(im * drivefrequency(device, i) * t)

    if result === nothing
        F = promote_type(eltype(a), eltype(e))
        result = Matrix{F}(undef, size(a))
    end
    result .= 0

    phase = Bool(j & 1) ? 1 : im    # Odd j -> "real" gradient operator; even j  -> "imag"
    result .+= (phase * e ) .* a
    result .+= (phase'* e') .* a'
    return result
end

function Devices.gradient(
    device::AbstractTransmonDevice,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix,
)::AbstractVector
    grad = zero.(Parameters.values(device))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    offset = 0
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        j = 2*(i-1) + 1             # If Julia indexed from 0, this could just be 2i...
        L = Parameters.count(Ω)
        for k in 1:L
            ∂̄ = Signals.partial(k, Ω, t̄)
            grad[offset + k] += sum(τ̄ .* real.(∂̄) .* ϕ̄[:,j])
            grad[offset + k] += sum(τ̄ .* imag.(∂̄) .* ϕ̄[:,j+1])
        end
        offset += L
    end

    # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        j = 2*(i-1) + 1             # If Julia indexed from 0, this could just be 2i...
        Ω̄ = Ω(t̄)
        grad[offset + i] += sum(τ̄ .* t̄ .* real.(Ω̄) .* ϕ̄[:,j+1])
        grad[offset + i] -= sum(τ̄ .* t̄ .* imag.(Ω̄) .* ϕ̄[:,j])

    end

    return grad
end

function Parameters.count(device::AbstractTransmonDevice)
    cnt = Devices.ndrives(device)           # NOTE: There are `ndrives` frequencies.
    for i in 1:Devices.ndrives(device)
        cnt += Parameters.count(drivesignal(device, i))
    end
    return cnt
end

function Parameters.names(device::AbstractTransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(names, (annotate(name,i) for name in Parameters.names(Ω)))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν̄[i]
    append!(names, ("ν$i" for i in 1:Devices.ndrives(device)))
    return names
end

function Parameters.values(device::AbstractTransmonDevice)
    values = []

    # STRING TOGETHER PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(values, Parameters.values(Ω))
    end

    # TACK ON PARAMETERS FOR EACH ν̄[i]
    append!(values, (drivefrequency(device, i) for i in 1:Devices.ndrives(device)))
    return identity.(values)
end

function Parameters.bind(device::AbstractTransmonDevice, x̄::AbstractVector)
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        L = Parameters.count(Ω)
        Parameters.bind(Ω, x̄[offset+1:offset+L])
        offset += L
    end

    # BIND PARAMETERS FOR EACH ν̄[i]
    bindfrequencies(device, x̄[offset+1:end])
end







struct TransmonDevice{
    F<:AbstractFloat,
} <: AbstractTransmonDevice{F}
    # QUBIT LISTS
    ω̄::Vector{F}
    δ̄::Vector{F}
    # COUPLING LISTS
    ḡ::Vector{F}
    quples::Vector{Devices.Quple}
    # DRIVE LISTS
    q̄::Vector{Int}
    ν̄::Vector{F}
    Ω̄::Vector{Signals.ArbitrarySignal}
    # OTHER PARAMETERS
    m::Int

    function TransmonDevice(
        ω̄::AbstractVector{<:Real},
        δ̄::AbstractVector{<:Real},
        ḡ::AbstractVector{<:Real},
        quples::AbstractVector{Devices.Quple},
        q̄::AbstractVector{Int},
        ν̄::AbstractVector{<:AbstractFloat},
        Ω̄::AbstractVector{<:Signals.AbstractSignal},
        m::Int,
    )
        # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
        @assert length(ω̄) == length(δ̄) ≥ 1              # NUMBER OF QUBITS
        @assert length(ḡ) == length(quples)             # NUMBER OF COUPLINGS
        @assert length(q̄) == length(ν̄) == length(Ω̄)     # NUMBER OF DRIVES

        # VALIDATE QUBIT INDICES
        for (p,q) in quples
            @assert 1 <= p <= length(ω̄)
            @assert 1 <= q <= length(ω̄)
        end
        for q in q̄
            @assert 1 <= q <= length(ω̄)
        end

        # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
        @assert m ≥ 2

        # STANDARDIZE TYPING AND CONVERT ALL LISTS TO IMMUTABLE TUPLE (except ν)
        F = promote_type(eltype(ω̄), eltype(δ̄), eltype(ḡ), eltype(ν̄))
        return new{F}(
            convert(Vector{F}, ω̄),
            convert(Vector{F}, δ̄),
            convert(Vector{F}, ḡ),
            quples,
            q̄,
            convert(Vector{F}, ν̄),
            [Signals.ArbitrarySignal(Ω) for Ω in Ω̄],
            m,
        )
    end
end

Devices.nstates(device::TransmonDevice, q::Int) = device.m

Devices.nqubits(device::TransmonDevice) = length(device.ω̄)
resonancefrequency(device::TransmonDevice, q::Int) = device.ω̄[q]
anharmonicity(device::TransmonDevice, q::Int) = device.δ̄[q]

ncouplings(device::TransmonDevice) = length(device.quples)
couplingpair(device::TransmonDevice, k::Int) = device.quples[k]
couplingstrength(device::TransmonDevice, k::Int) = device.ḡ[k]

Devices.ndrives(device::TransmonDevice) = length(device.q̄)
Devices.drivequbit(device::TransmonDevice, i::Int) = device.q̄[i]
drivefrequency(device::TransmonDevice, i::Int) = device.ν̄[i]
drivesignal(device::TransmonDevice, i::Int) = device.Ω̄[i]

bindfrequencies(device::TransmonDevice, ν̄::AbstractVector) = (device.ν̄ .= ν̄)


#= TODO (low): Other types

LegacyTransmonDevice: assume Ω(t) is real, no analytical gradient for ν, half as many grade operators.
FixedFrequencyTransmonDevice: ν is tuple and not included in parameters.
LinearTransmonDevice: quples and ḡ replaced by n-length tuple ḡ, efficient static propagate.
TransmonDeviceSansRWA: implicitly one channel per qubit, different drive
a mix of the three I guess...

=#