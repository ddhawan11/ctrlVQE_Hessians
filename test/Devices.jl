using Test
import .StandardTests

import CtrlVQE: Signals, Devices
import CtrlVQE.Quples: Quple

# TRANSMON DEVICE PARAMETERS
ω̄ = 2π * [4.50, 4.52]
δ̄ = 2π * [0.33, 0.34]
ḡ = 2π * [0.020]
quples = [Quple(1, 2)]
q̄ = [1, 2]
ν̄ = 2π * [4.30, 4.80]
Ω̄ = [
    Signals.Constant( 0.020*2π),
    Signals.Constant(-0.020*2π),
]
m = 3

# TRANSMON DEVICE SELF-CONSISTENCY CHECKS
@testset "TransmonDevice" begin
    device = Devices.TransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)
    StandardTests.validate(device)
end

@testset "FixedFrequencyTransmonDevice" begin
    device = Devices.FixedFrequencyTransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)
    StandardTests.validate(device)
end