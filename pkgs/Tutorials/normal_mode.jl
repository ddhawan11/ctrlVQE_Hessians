#= optimize a pulse to find the ground-state energy of a molecular hamiltonian. =#

##########################################################################################
#= PREAMBLE =#
push!(LOAD_PATH,"/Users/dikshadhawan/Documents/ctrlVQE_Hessians/src")
push!(LOAD_PATH,"/Users/dikshadhawan/Documents/ctrlVQE_Hessians/src/costfns")
import CtrlVQE
import Random 
import NPZ, Optim, LineSearches
using LinearAlgebra
using Printf
using Plots
#import Evolutions, Parameters, LinearAlgebraTools, Devices
#import QubitOperators
#import Operators: StaticOperator, IDENTITY
#import ...EnergyFunctions, ...AbstractGradientFunction

matrix = "H2_sto-3g_singlet_1.5_P-m"    # MATRIX FILE
T = 5.0 # ns                # TOTAL DURATION OF PULSE
W = 10                      # NUMBER OF WINDOWS IN EACH PULSE

r = round(Int,20T)          # NUMBER OF STEPS IN TIME EVOLUTION
m = 2                       # NUMBER OF LEVELS PER TRANSMON

seed = 9999                 # RANDOM SEED FOR PULSE INTIALIZATION
init_Ω = 0.0 # 2π GHz       # AMPLITUDE RANGE FOR PULSE INITIALIZATION
init_φ = 0.0                # PHASE RANGE FOR PULSE INITIALIZATION
init_Δ = 0.0 # 2π GHz       # FREQUENCY RANGE FOR PULSE INITIALIZATION

ΩMAX = 2π * 0.02 # 2π GHz   # AMPLITUDE BOUNDS
λΩ = 1.0 # Ha               # PENALTY WEIGHT FOR EXCEEDING AMPLITUDE BOUNDS
σΩ = ΩMAX                   # PENALTY STEEPNESS FOR EXCEEDING AMPLITUDE BOUNDS

ΔMAX = 2π * 1.00 # 2π GHz   # FREQUENCY BOUNDS
λΔ = 1.0 # Ha               # PENALTY WEIGHT FOR EXCEEDING FREQUENCY BOUNDS
σΔ = ΔMAX                   # PENALTY STEEPNESS FOR EXCEEDING FREQUENCY BOUNDS

f_tol = 0.0                 # TOLERANCE IN FUNCTION EVALUATION
g_tol = 1e-6                # TOLERANCE IN GRADIENT NORM
maxiter = 10000             # MAXIMUM NUMBER OF ITERATIONS

##########################################################################################
#= SETUP =#

# LOAD MATRIX AND EXTRACT REFERENCE STATES
H = NPZ.npzread("$(@__DIR__)/matrix/$matrix.npy")
n = CtrlVQE.QubitOperators.nqubits(H)
ψ_REF = CtrlVQE.QubitOperators.reference(H) # REFERENCE STATE
REF = real(ψ_REF' * H * ψ_REF)              # REFERENCE STATE ENERGY

# IDENTIFY EXACT RESULTS
Λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H))
ψ_FCI = U[:,1]                              # GROUND STATE
FCI = Λ[1]                                  # GROUND STATE ENERGY
FES = Λ[2]                                  # FIRST EXCITED STATE

# CONSTRUCT THE MAJOR PACKAGE OBJECTS

pulse = CtrlVQE.UniformWindowed(CtrlVQE.Signals.ComplexConstant(0.0, 0.0), T, W)
ΩMAX /= √2  # Re-scale max amplitude so that bounds inscribe the complex circle.
            # Not needed for real or polar-parameterized amplitudes.
# pulse = CtrlVQE.UniformWindowed(CtrlVQE.Signals.Constant(0.0), T, W)
# pulse = CtrlVQE.UniformWindowed(CtrlVQE.Signals.PolarComplexConstant(0.0, 0.0), T, W)


device = CtrlVQE.Systematic(CtrlVQE.Devices.FixedFrequencyTransmonDevice, n, pulse)
# device = CtrlVQE.Systematic(CtrlVQE.Devices.TransmonDevice, n, pulse)

algorithm = CtrlVQE.Rotate(r)

# INITIALIZE PARAMETERS
Random.seed!(seed)
xi = CtrlVQE.Parameters.values(device)

L = length(xi)                      # NUMBER OF PARAMETERS
Ω = 1:L; φ = []; ν = []                 # INDEXING VECTORS (Cartesian sans Frequencies)
# Ω = 1:L-n; φ = []; ν = 1+L-n:L          # INDEXING VECTORS (Cartesian with Frequencies)
# Ω = 1:2:L-n; φ = 2:2:L-n; ν = 1+L-n:L   # INDEXING VECTORS (Polar with Frequenices)
# Ω = 1:2:L; φ = 2:2:L; ν = []            # INDEXING VECTORS (Polar sans Frequencies)

xi[Ω] .+= init_Ω .* (2 .* rand(length(Ω)) .- 1)
xi[φ] .+= init_φ .* (2 .* rand(length(φ)) .- 1)
xi[ν] .+= init_Δ .* (2 .* rand(length(ν)) .- 1)

##########################################################################################
#= PREPARE OPTIMIZATION OBJECTS =#

# ENERGY FUNCTIONS
O0 = CtrlVQE.QubitOperators.project(H, device)              # MOLECULAR HAMILTONIAN
ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, device)          # REFERENCE STATE

fn_energy, gd_energy = CtrlVQE.ProjectedEnergy.functions(
    O0, ψ0, T, device, r;
    frame=CtrlVQE.STATIC,
)
println("O0  ", O0)

# PENALTY FUNCTIONS
λ  = zeros(L);  λ[Ω] .=    λΩ                               # PENALTY WEIGHTS
μR = zeros(L); μR[Ω] .= +ΩMAX                               # PENALTY LOWER BOUNDS
μL = zeros(L); μL[Ω] .= -ΩMAX                               # PENALTY UPPER BOUNDS
σ  = zeros(L);  σ[Ω] .=    σΩ                               # PENALTY SCALINGS
fn_penalty, gd_penalty = CtrlVQE.SmoothBounds.functions(λ, μR, μL, σ)

# OPTIMIZATION FUNCTIONS
fn = CtrlVQE.CompositeCostFunction(fn_energy, fn_penalty)
gd = CtrlVQE.CompositeGradientFunction(gd_energy, gd_penalty)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
optimizer = Optim.LBFGS(linesearch=linesearch)

# OPTIMIZATION OPTIONS
options = Optim.Options(
    show_trace = true,
    show_every = 1,
    f_tol = f_tol,
    g_tol = g_tol,
    iterations = maxiter,
)


function hessian(f, g, x; stepsize=1e-4)

    nparams = length(x)
    H = zeros(nparams, nparams) # Hessian

    for i in 1:nparams
        x_add = deepcopy(x)
        x_sub = deepcopy(x)

        x_add[i] += stepsize
        x_sub[i] -= stepsize

        g_add = g(x_add)
        g_sub = g(x_sub)
        H[:,i] .= (g_add - g_sub) ./ (2 * stepsize)
    end

    return .5*(H+H')
end

H = hessian(fn, gd, xi)

function plot_pulse_normalmodes(H, nqubits; nvecs=4)
    np = size(H,2)
    size(H,2) % nqubits == 0 || throw(DimensionMismatch)
    U, σ, V = svd(H)
    σ = σ[1:nvecs]
    U = U[:, 1:nvecs]
    for σi in σ 
        @printf(" %12.8f\n", σi)
    end

    Ur = U[[i%2==1 for i in 1:np], :] 
    Ui = U[[i%2==0 for i in 1:np], :] 

    np_per_qubit = np ÷ nqubits

    Urs = Vector{Matrix}([])
    Uis = Vector{Matrix}([])

    shift = 1
    for i in 1:nqubits
        Uqr =  Ur[shift:shift+np_per_qubit ÷ 2 - 1, :]
        Uqi =  Ui[shift:shift+np_per_qubit ÷ 2 - 1, :]
        shift += np_per_qubit ÷ 2
        
        push!(Urs, Uqr)
        push!(Uis, Uqi)
    end

    println(" Real")
    for i in Urs
        display(i)
    end
    println(" Imag")
    for i in Uis
        display(i)
    end

    colors = [:red, :blue, :green]
    for i in 1:nvecs
        plot()
        for q in 1:nqubits
            plot!(Urs[q][:,i], linestyle=:solid, linecolor=colors[q], label=@sprintf("ℜ Qubit: %i",q))
            plot!(Uis[q][:,i], linestyle=:dash, linecolor=colors[q], label=@sprintf("ℑ Qubit: %i",q))
        end
        ylims!(-.5,.5)
        savefig(@sprintf("norm_mode_%i.pdf", i))
    end
end

plot_pulse_normalmodes(H, 2, nvecs=4)

function my_opt(f, g, xi; nvecs=2, trust=1, thresh=1e-6, maxiter=30)

    function costfn(xi)
        F = real(promote_type(Float16, eltype(O0), eltype(ψ0), eltype(T)))
    
        ψ = Array{CtrlVQE.LinearAlgebraTools.cis_type(F)}(undef, size(ψ0))
        CtrlVQE.Parameters.bind(f.device, xi)
        CtrlVQE.Evolutions.evolve(
            f.algorithm,
            f.device,
            f.basis,
            f.T,
            f.ψ0;
            result=ψ,)
        return(real(CtrlVQE.LinearAlgebraTools.expectation(f.OT, ψ)))
    end
    
    function gradfn(∇f̄::AbstractVector, xi::AbstractVector)
        CtrlVQE.Parameters.bind(g.f.device, xi)
        CtrlVQE.Evolutions.gradientsignals(
        g.f.device,
        g.f.basis,
        g.f.T,
        g.f.ψ0,
        g.r,
        g.f.OT;
        result=g.ϕ̄,
        evolution=g.f.algorithm,
        )
        τ, τ̄, t̄ = CtrlVQE.Evolutions.trapezoidaltimegrid(g.f.T, g.r)
        ∇f̄ .= CtrlVQE.Devices.gradient(g.f.device, τ̄, t̄, g.ϕ̄)
        ∇f̄  = - U * pinv(U' * H * U) * (U' * gd(xi)) # Projected pulse with initial inverse hessian
        ∇f̄  = -U * U' * ∇f̄  #optimized pulse
#        println(∇f̄)
        return ∇f̄
    end
    
        
    function newton_step(U, xi)

        H = hessian(fn, gd, xi)                           # Recompute hessian

        # return -gd(xi)                                # This will try to optimize the full pulse
        # return -U * U' * gd(xi)                       # This will try to optimize the projected pulse
        return - U * pinv(U' * H * U) * (U' * gd(xi))   # This will do projected pulse, and with the initial inverse hessian
    end

    function gradient_step(U, xi)
        return -U * U' * gd(xi)                       # This will try to optimize the projected pulse
    end
    H = hessian(f, g, xi)
    Ui, si, _ = svd(H)
    display(si[1:nvecs])
    U = Ui[:, 1:nvecs]
    U = Ui[:, 1:4]


    Hss = U' * H * U            # Hessian in the subspace
    display(Hss)
    println()
    xcurr = deepcopy(xi)
    # # OPTIMIZATION ALGORITHM
    linesearch = LineSearches.MoreThuente()
    optimizer = Optim.LBFGS(linesearch=linesearch)

    # # OPTIMIZATION OPTIONS
    options = Optim.Options(
         show_trace = true,
         show_every = 1,
         f_tol = f_tol,
         g_tol = g_tol,
         iterations = maxiter,
     )
    optimization = Optim.optimize(costfn, gradfn, xi, optimizer, options)
#    optimization = Optim.optimize(f,g, xi, optimizer, options)

    xf = Optim.minimizer(optimization)      # FINAL PARAMETERS
    # for i in 1:maxiter
    #      ecurr = f(xcurr)
    #      δx = newton_step(U, xcurr)
    #     # δx = gradient_step(U, xcurr)

    #     function stepsize_fun(α)
    #         return f(xcurr + δx*α[1])
    #     end

    #     α = 0.0
    #     res = Optim.optimize(stepsize_fun, [α])
    #     α = Optim.minimizer(res)[1]

    #     δx = δx * α
    #     nδx = norm(δx)

    #     δx .= δx .* min(trust, nδx) ./ nδx

    #     @printf(" Step: %4i %12.8f |g| = %12.8f |x| = %12.8f α = %12.8f\n", i, f(xcurr), norm(g(xcurr)), norm(δx), α)
    #     xcurr += δx
    #     if norm(δx) < thresh
    #         break
    #     end
    # end
end



my_opt(fn_energy, gd_energy, xi, nvecs=2, trust=.1, maxiter=20)



