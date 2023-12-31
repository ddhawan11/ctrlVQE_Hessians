using Test
import FiniteDifferences: grad, central_fdm
import LinearAlgebra: I

import CtrlVQE: Parameters, Devices, Signals
import CtrlVQE.Devices: LocallyDrivenDevices
import CtrlVQE.Bases: DRESSED, OCCUPATION
import CtrlVQE.Operators: StaticOperator, IDENTITY, COUPLING, STATIC
import CtrlVQE.Operators: Qubit, Channel, Drive, Hamiltonian, Gradient

const t = 1.0
const r = 10
const τ = t / r

const τ̄ = fill(τ, r+1); τ̄[[1,end]] ./=2
const t̄ = range(0.0, t, r+1)
const ϕ̄ = ones(r+1)

function validate(device::Devices.Device)
    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    N = Devices.nstates(device)
    nD= Devices.ndrives(device)
    nG= Devices.ngrades(device)

    # PARAMETERS AND GRADIENT

    L = Parameters.count(device)
    x̄ = Parameters.values(device)
    @test length(x̄) == length(Parameters.names(device)) == L

    Parameters.bind(device, 2 .* x̄)
    @test Parameters.values(device) ≈ 2 .* x̄
    Parameters.bind(device, x̄)

    # TODO (hi): Check the *actual* gradient against finite difference, in another function. (Requires Evolutions.) =#
    ϕ̄_ = repeat(ϕ̄, 1, nG)
    grad = Devices.gradient(device, τ̄, t̄, ϕ̄_)
    @test length(grad) == L
    grad_ = zero(grad); Devices.gradient(device, τ̄, t̄, ϕ̄_; result=grad_)
    @test grad ≈ grad_

    # HAMILTONIAN OPERATORS

    ā = Devices.algebra(device)
    @test Devices.eltype_algebra(device) == eltype(ā)
    @test size(ā) == (N, N, n)

    a0 = Devices.localloweringoperator(device)
    @test Devices.eltype_localloweringoperator(device) == eltype(a0)
    a0_ = zero(a0); Devices.localloweringoperator(device; result=a0_)
    @test a0 ≈ a0_

    G = Devices.staticcoupling(device, ā)
    @test Devices.eltype_staticcoupling(device) == eltype(G)
    G_ = zero(G); Devices.staticcoupling(device, ā; result=G_)
    @test G ≈ G_
    @test G ≈ G'          # NOTE: Sanity check only. We are not testing for correctness!

    h̄ = [Devices.qubithamiltonian(device, ā, q) for q in 1:n]
    for (q, h) in enumerate(h̄)
        @test Devices.eltype_qubithamiltonian(device) == eltype(h)
        h_ = zero(h); Devices.qubithamiltonian(device, ā, q; result=h_)
        @test h ≈ h_
        @test h ≈ h'      # NOTE: Sanity check only. We are not testing for correctness!
    end

    v̄ = [Devices.driveoperator(device, ā, i, t) for i in 1:nD]
    for (i, v) in enumerate(v̄)
        @test Devices.eltype_driveoperator(device) == eltype(v)
        v_ = zero(v); Devices.driveoperator(device, ā, i, t; result=v_)
        @test v ≈ v_
        @test v ≈ v'      # NOTE: Sanity check only. We are not testing for correctness!
    end

    Ā = [Devices.gradeoperator(device, ā, j, t) for j in 1:nG]
    for (j, A) in enumerate(Ā)
        @test Devices.eltype_gradeoperator(device) == eltype(A)
        A_ = zero(A); Devices.gradeoperator(device, ā, j, t; result=A_)
        @test A ≈ A_
        @test A ≈ A'      # NOTE: Sanity check only. We are not testing for correctness!
    end

    # BASIS CONTROL

    āL = Devices.localalgebra(device)
    @test Devices.eltype_algebra(device) == eltype(ā)
    @test size(āL) == (m, m, n)

    for q in 1:n
        aL = āL[:,:,q]
        @test aL ≈ a0
        aG = Devices.globalize(device, aL, q)
        @test aG ≈ ā[:,:,q]
        aG_ = zero(aG); Devices.globalize(device, aL, q; result=aG_)
        @test aG ≈ aG_
    end

    #=
    TODO (lo): Some meaningful consistency check on the dressed basis.
        I'm not sure how much I *care* about the dressed basis, though...

    @time Devices.diagonalize(DRESSED, device)
    @time Devices.diagonalize(OCCUPATION, device)
    @time Devices.diagonalize(OCCUPATION, device, q)

    @time Devices.basisrotation(DRESSED, OCCUPATION, device)
    @time Devices.basisrotation(OCCUPATION, OCCUPATION, device)
    @time Devices.basisrotation(OCCUPATION, OCCUPATION, device, q)
    @time Devices.localbasisrotations(OCCUPATION, OCCUPATION, device)
    =#

    # OPERATOR METHOD ACCURACIES

    @test Devices.operator(IDENTITY, device) ≈ Matrix(I, N, N)
    @test Devices.operator(COUPLING, device) ≈ G
    @test Devices.operator(STATIC, device) ≈ sum(h̄) + G
    for (q, h) in enumerate(h̄); @test Devices.operator(Qubit(q), device) ≈ h; end
    for (i, v) in enumerate(v̄); @test Devices.operator(Channel(i,t), device) ≈ v; end
    for (j, A) in enumerate(Ā); @test Devices.operator(Gradient(j,t), device) ≈ A; end
    @test Devices.operator(Drive(t), device) ≈ sum(v̄)
    @test Devices.operator(Hamiltonian(t), device) ≈ sum(h̄) + G + sum(v̄)

    # OPERATOR METHOD CONSISTENCIES

    ψ0 = zeros(ComplexF64, N); ψ0[N] = 1
    ops = [
        IDENTITY, COUPLING, STATIC,
        Qubit(n), Channel(nD,t), Gradient(nG,t),
        Drive(t), Hamiltonian(t),
    ]

    for op in ops
        H = Devices.operator(op, device)
        @test eltype(op, device) == eltype(H)
        H_ = zero(H); Devices.operator(op, device; result=H_)
        @test H ≈ H_

        U = Devices.propagator(op, device, τ)
        @test U ≈ exp((-im*τ) .* H)
        U_ = zero(U); Devices.propagator(op, device, τ; result=U_)
        @test U ≈ U_

        ψ = copy(ψ0); Devices.propagate!(op, device, τ, ψ)
        @test ψ ≈ U * ψ0

        if op isa StaticOperator
            UT = Devices.evolver(op, device, τ)
            @test UT ≈ U
            UT_ = zero(UT); Devices.evolver(op, device, τ; result=UT_)
            @test UT ≈ UT_

            ψ_ = copy(ψ0); Devices.evolve!(op, device, τ, ψ_)
            @test ψ ≈ ψ_
        end

        E = Devices.expectation(op, device, ψ)
        @test abs(E - (ψ' * H * ψ)) < 1e-8
        F = Devices.braket(op, device, ψ, ψ0)
        @test abs(F - (ψ' * H * ψ0)) < 1e-8

    end


    # LOCAL METHODS

    h̄L = Devices.localqubitoperators(device)
    @test size(h̄L) == (m,m,n)
    h̄L_ = zero(h̄L); Devices.localqubitoperators(device; result=h̄L_)
    @test h̄L ≈ h̄L_

    ūL = Devices.localqubitpropagators(device, τ)
    @test size(ūL) == (m,m,n)
    ūL_ = zero(ūL); Devices.localqubitpropagators(device, τ; result=ūL_)
    @test ūL ≈ ūL_

    ūtL = Devices.localqubitevolvers(device, τ)
    @test size(ūtL) == (m,m,n)
    ūtL_ = zero(ūtL); Devices.localqubitevolvers(device, τ; result=ūtL_)
    @test ūtL ≈ ūtL_

    for q in 1:n
        hL = h̄L[:,:,q]; uL = ūL[:,:,q]; utL = ūtL[:,:,q]
        @test Devices.globalize(device, hL, q) ≈ h̄[q]
        @test uL ≈ exp((-im*τ) .* hL)
        @test uL ≈ utL
    end

    return true
end

function validate(device::LocallyDrivenDevices.LocallyDrivenDevice)
    super = invoke(validate, Tuple{Devices.Device}, device)
    !super && return false

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    nD= Devices.ndrives(device)
    nG= Devices.ngrades(device)

    # QUBIT ASSIGNMENTS

    for i in 1:nD; @test 1 ≤ LocallyDrivenDevices.drivequbit(device,i) ≤ n; end
    for j in 1:nG; @test 1 ≤ LocallyDrivenDevices.gradequbit(device,j) ≤ n; end

    # LOCAL DRIVE METHODS

    ā = Devices.algebra(device)
    v̄ = [Devices.driveoperator(device, ā, i, t) for i in 1:nD]

    v̄L = LocallyDrivenDevices.localdriveoperators(device, t)
    @test size(v̄L) == (m,m,n)
    v̄L_ = zero(v̄L); LocallyDrivenDevices.localdriveoperators(device, t; result=v̄L_)
    @test v̄L ≈ v̄L_

    ūL = LocallyDrivenDevices.localdrivepropagators(device, τ, t)
    @test size(ūL) == (m,m,n)
    ūL_ = zero(ūL); LocallyDrivenDevices.localdrivepropagators(device, τ, t; result=ūL_)
    @test ūL ≈ ūL_

    for q in 1:n
        vL = v̄L[:,:,q]; uL = ūL[:,:,q]
        @test Devices.globalize(device, vL, q) ≈ v̄[q]
        @test uL ≈ exp((-im*τ) .* vL)
    end

    return true
end






function validate(signal::Signals.AbstractSignal{P,R}) where {P,R}

    # TEST PARAMETERS

    L = Parameters.count(signal)
    x̄ = Parameters.values(signal)
    @test eltype(x̄) == P
    @test length(x̄) == length(Parameters.names(signal)) == L

    Parameters.bind(signal, 2 .* x̄)
    @test Parameters.values(signal) ≈ 2 .* x̄
    Parameters.bind(signal, x̄)

    # TEST FUNCTION CONSISTENCY

    ft = signal(t)
    @test eltype(ft) == R

    ft̄ = signal(t̄)
    @test eltype(ft̄) == R
    @test ft̄ ≈ [signal(t_) for t_ in t̄]
    ft̄_ = zero(ft̄); signal(t̄; result=ft̄_)
    @test ft̄ ≈ ft̄_

    # TEST GRADIENT CONSISTENCY

    for i in 1:L
        gt = Signals.partial(i, signal, t)
        @test eltype(gt) == R

        gt̄ = Signals.partial(i, signal, t̄)
        @test eltype(gt̄) == R
        @test gt̄ ≈ [Signals.partial(i, signal, t_) for t_ in t̄]
        gt̄_ = zero(gt̄); Signals.partial(i, signal, t̄; result=gt̄_)
        @test gt̄ ≈ gt̄_
    end

    # CHECK GRADIENT AGAINST THE FINITE DIFFERENCE

    g0 = [Signals.partial(i, signal, t) for i in 1:L]
    @test eltype(g0) == R

    function fℜ(x)
        Parameters.bind(signal, x)
        fx = real(signal(t))
        Parameters.bind(signal, x̄)  # RESTORE ORIGINAL VALUES
        return fx
    end
    gΔℜ = grad(central_fdm(5, 1), fℜ, x̄)[1]

    if R <: Complex
        function fℑ(x)
            Parameters.bind(signal, x)
            fx = imag(signal(t))
            Parameters.bind(signal, x̄)  # RESTORE ORIGINAL VALUES
            return fx
        end
        gΔℑ = grad(central_fdm(5, 1), fℑ, x̄)[1]
        gΔ = Complex.(gΔℜ, gΔℑ)
    else
        gΔ = gΔℜ
    end

    εg = g0 .- gΔ
    @test √(sum(abs2.(εg))./length(εg)) < 1e-5

    # CONVENIENCE FUNCTIONS
    @test typeof(string(signal)) == String

    #= TODO (lo): Test integrate functions. =#

end