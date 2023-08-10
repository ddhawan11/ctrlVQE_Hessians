import CtrlVQE
import Random, LinearAlgebra
import NPZ, Optim, LineSearches, Plots

using FiniteDiff, StaticArrays

matrix = "H4chain_sto3g_1.5"    # MATRIX FILE
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

# PENALTY STEEPNESS FOR EXCEEDING AMPLITUDE BOUNDS

ΔMAX = 2π * 1.00 # 2π GHz   # FREQUENCY BOUNDS
λΔ = 1.0 # Ha               # PENALTY WEIGHT FOR EXCEEDING FREQUENCY BOUNDS
σΔ = ΔMAX                   # PENALTY STEEPNESS FOR EXCEEDING FREQUENCY BOUNDS

f_tol = 0.0                 # TOLERANCE IN FUNCTION EVALUATION
g_tol = 1e-6                # TOLERANCE IN GRADIENT NORM
maxiter = 1#0000             # MAXIMUM NUMBER OF ITERATIONS

##########################################################################################
#= SETUP =#

# LOAD MATRIX AND EXTRACT REFERENCE STATES
H = NPZ.npzread("$(@__DIR__)/matrix/$matrix.npy")
#print(H)

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
#print("pulse  ", pulse)

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

# PENALTY FUNCTIONS
λ  = zeros(L);  λ[Ω] .=    λΩ                               # PENALTY WEIGHTS
μR = zeros(L); μR[Ω] .= +ΩMAX                               # PENALTY LOWER BOUNDS
μL = zeros(L); μL[Ω] .= -ΩMAX                               # PENALTY UPPER BOUNDS
σ  = zeros(L);  σ[Ω] .=    σΩ                               # PENALTY SCALINGS
fn_penalty, gd_penalty = CtrlVQE.SmoothBounds.functions(λ, μR, μL, σ)

# OPTIMIZATION FUNCTIONS
fn = CtrlVQE.CompositeCostFunction(fn_energy, fn_penalty)
gd = CtrlVQE.CompositeGradientFunction(gd_energy, gd_penalty)

hessian = zeros(size(xi)[1], size(xi)[1])
h = 0.0001
for i in (1:size(xi)[1])
    xin = zeros(size(xi))
    xin[i] += h
    xip = zeros(size(xi))
    xip[i] -= h
    hessian[i,:] = (gd(xin) - gd(xip))/(2*h)
end
print(hessian)

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
g_calls_limit=1
) 

##########################################################################################
#= RUN OPTIMIZATION =#
optimization = Optim.optimize(fn, gd, xi, optimizer, options)

xf = Optim.minimizer(optimization)      # FINAL PARAMETERS
print("gd ", gd, "\n")
exit()
##########################################################################################
#= REPORT RESULTS =#

ff = fn(xf)             # LOSS FUNCTION
Ef = fn_energy(xf)      # CURRENT ENERGY
λf = fn_penalty(xf)     # PENALTY CONTRIBUTION
εE = Ef - FCI           # ENERGY ERROR
cE = 1-εE/(REF-FCI)     # CORRELATION ENERGY

Ωsat = sum((
count(xf[Ω] .≥ ( ΩMAX)),
count(xf[Ω] .≤ (-ΩMAX)),
))

println("""

Optimization Results
--------------------
Energy Error: $εE
% Corr Energy: $(cE*100)

Loss Function: $ff
from  Energy: $Ef
from Penalty: $λf

Saturated Amplitudes: $Ωsat / $(length(Ω))
""")
=#

##########################################################################################
#= PLOT RESULTS =#

#= # EXTRACT REAL/IMAGINARY PARTS OF THE PULSE
_, _, t = CtrlVQE.trapezoidaltimegrid(T, r)
CtrlVQE.Parameters.bind(device, xf)             # ENSURE DEVICE USES THE FINAL PARAMETERS
nD = CtrlVQE.ndrives(device)
α = Array{Float64}(undef, r+1, nD)
β = Array{Float64}(undef, r+1, nD)
for i in 1:nD
Ωt = CtrlVQE.Devices.TransmonDevices.drivesignal(device, i)(t)
α[:,i] = real.(Ωt)
β[:,i] = imag.(Ωt)
end

# SET UP PLOT OBJECT
yMAX = (ΩMAX / 2π) * 1.1        # Divide by 2π to convert angular frequency to frequency.
# Multiply by 1.1 to add a little buffer to the plot.
plot = Plots.plot(;
xlabel= "Time (ns)",
ylabel= "|Amplitude| (GHz)",
ylims = [-yMAX, yMAX],
legend= :topright,
)

# DUMMY PLOT OBJECTS TO SETUP LEGEND THE WAY WE WANT IT
Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="α")
Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="β")

# PLOT AMPLITUDES
for i in axes(nD,2)
Plots.plot!(plot, t, α[:,i]./2π, lw=3, ls=:solid, color=i, label="Drive $i")
Plots.plot!(plot, t, β[:,i]./2π, lw=3, ls=:dot, color=i, label=false)
end

# SHOW PLOT
Plots.gui() =#
~
