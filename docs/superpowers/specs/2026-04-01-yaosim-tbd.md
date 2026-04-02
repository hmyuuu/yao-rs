# yaosim CLI — To Be Discussed

Open discussion topics collected during brainstorming on 2026-04-01. These are **separate from the v1 design spec** — they represent future directions, unresolved questions, and features that need further discussion before being designed.

## 1. Backend Auto-Selection Logic Details

Auto-select simulation backend based on circuit content and init-state type:
- statevec if no noise channels and init state is a vector
- dm if noise channels present or init state is a density matrix
- `--backend` override

**Open questions:**
- What happens if user forces `--backend statevec` on a noisy circuit? Error? Warn and skip noise? Require `--ignore-noise`?
- Should tn be a separate backend or just the export path?

## 2. Fidelity Task

`--task fidelity --reference ref.json` — compute fidelity between output state and a reference.

**Open questions:**
- Requires new library code: `fidelity(state1, state2) -> f64` not in yao-rs today
- Should it support fidelity between density matrices too?
- Output format: single scalar? Include phase info?

## 3. Entanglement Entropy Task

`--task entropy --partition 0,1:2,3` — compute entanglement entropy via partial trace.

**Open questions:**
- Requires new library code: reduced density matrix + von Neumann entropy not in yao-rs
- Which entropy measures? Von Neumann only, or also Renyi?
- How to specify partitions on the CLI ergonomically?

## 4. TN Subcommand (`yaosim tn`)

yaosim embeds omeco as library dependency for TN workflows:
- `yaosim tn export` — circuit to TN JSON (einsum + tensors + sizes)
- `yaosim tn optimize` — find contraction order, report complexity (tc/sc/rwc)
- `yaosim tn contract` — optimize + contract end-to-end

omeco also gets its own standalone CLI for non-quantum TN users.

**Open questions:**
- Exact JSON format for TN export (omeco's Julia-compatible format? custom?)
- Which omeco optimizers to expose (GreedyMethod, TreeSA, both?)
- Slicing support (`--sc-target`) in v1?
- How does `yaosim tn contract` interact with the torch feature flag?
- Should omeco CLI design be a separate spec?

## 5. Verbosity Levels

`--quiet` / default / `--verbose` behavior details.

**Open questions:**
- What exactly does default show vs `--verbose`?
- Should `--quiet` suppress all stderr or just progress?
- Is there a `--porcelain` flag for stable machine output format?

## 6. Parameterized Circuit Mechanism

Currently planned as `{{theta}}` template substitution in the CLI layer.

**Open questions:**
- Template substitution vs library-level `Gate::Rx(Parameter)` support
- File a GitHub issue for yao-rs library-level parameterized circuits after CLI is merged
- How does `--sweep` interact with multi-parameter circuits?

## 7. `circuit_to_expectation` |0> Limitation

yao-rs `circuit_to_expectation(circuit, obs)` hardcodes initial state to |0...0>.

**Open questions:**
- Need a statevec-based expectation path for arbitrary init states: compute U|psi>, then <psi_out|O|psi_out> via dot product
- Should TN path be extended to support arbitrary boundary states?
- How to handle this transparently in the CLI (auto-select path based on init state)?

## 8. Non-Z-Basis Measurement

yao-rs `measure()` is Z-basis projective only.

**Open questions:**
- Support measurement with operator argument (like Yao.jl `Measure(n, operator=X)`)
- Or document "append H gates for X-basis measurement" as the workaround
- Requires new yao-rs library code for non-trivial implementation

## 9. Unitary Task

`--task unitary` — compute the full unitary matrix U of the circuit.

**Open questions:**
- Not in yao-rs today — needs 2^n apply calls (one per basis state)
- Only practical for small circuits (<= ~10 qubits)
- Is this worth implementing or just a convenience over scripted apply?

## 10. Crate Split / Repo Location

yaosim-core vs yao-rs-qasm separation, and whether CLI moves to a separate repo.

**Open questions:**
- Should QASM parser be its own crate for reuse by yao-rs-py?
- Should yaosim live in yao-rs workspace or its own repo?
- How to handle cross-repo dependencies if separated?

## 11. TN Boundary Extension for Arbitrary Overlap

v1 computes arbitrary <phi|U|psi> via statevec path (apply + dot product). For large circuits where statevec is infeasible, the TN path needs extension.

**Current yao-rs limitation:**
- `circuit_to_overlap()` hardcodes both bra and ket to |0...0>
- `circuit_to_einsum_with_boundary(circuit, final_state)` only supports |0> boundaries (hardcoded rank-1 tensors `[1, 0, 0, ...]`)

**Yao.jl reference implementation (`yao2einsum`):**
```julia
yao2einsum(circuit;
    initial_state = Dict(),   # |psi> at input legs
    final_state   = Dict(),   # <phi| at output legs
    optimizer     = TreeSA(),
    mode          = VectorMode())
```

Boundary states accept two forms:
- **Integer**: `Dict(1=>0, 2=>1)` — pins qubits to computational basis states
- **ArrayReg**: `Dict(1=>ArrayReg([0.6, 0.8im]))` — arbitrary single-qubit state vector

Qubits not listed in either dict remain as **open (free) legs** in the tensor network.

Equivalence: `contract(network)[] == zero_state(n)' * (zero_state(n) |> circuit)`

**Yao.jl also provides:**
- `reg'` (adjoint) for bra representation
- `reg_phi' * reg_psi` for inner product (the canonical idiom)
- `fidelity(reg1, reg2)` for F(rho, sigma) including mixed states

**Required changes to yao-rs:**
- Extend `circuit_to_einsum_with_boundary` to accept `Dict<usize, Vec<Complex64>>` for both initial and final states
- Or add a new function `circuit_to_einsum_with_states(circuit, initial_states, final_states)`

## 12. State Round-Tripping

Pipe output state from one `yaosim run` as input to another.

**Open questions:**
- `--output-state state.json` flag that writes state as JSON (dims + amplitudes)
- Loadable by `--init-state state.json`
- Documented JSON schema for state files

## 13. NDJSON vs JSON Array for Sweep Output

When `--sweep` + `--json` produces multiple results.

**Open questions:**
- NDJSON (one JSON object per line) — streamable with jq
- JSON array `[{...}, {...}]` — simpler but not streamable
- Or configurable via flag?

## 14. Exit Code Differentiation

Standard Rust error handling with `Result`/`anyhow`/`thiserror`.

**Open questions:**
- Simple 0/1 may be sufficient
- Differentiated codes (1=usage, 2=IO, 3=simulation) useful for scripting
- Low priority — revisit based on user feedback

## 15. Time Evolution Task

`e^{-iHt}|psi>` via Trotterization — a primary quantum simulation use case.

**Open questions:**
- Not mentioned in current design
- Would require Hamiltonian input + time parameter + Trotter order
- Consider for v2

## 16. Stabilizer Backend

Clifford/stabilizer tableau simulation (Gottesman-Knill theorem). O(n^2) per Clifford gate, exponentially efficient for Clifford-only circuits.

**Open questions:**
- Requires new yao-rs library code (stabilizer tableau representation, Clifford gate application)
- How to handle non-Clifford gates? Error, or fall back to statevec?
- Which tasks are supported? (statevector output doesn't make sense for stabilizer — it tracks Pauli operators, not amplitudes)
- Supported tasks likely: `sample`, `expect` (for Pauli observables), `probs`
- Reference implementations: Stim (fastest), Qiskit Aer stabilizer backend

## 17. Quantum Monte Carlo Trajectory Simulation

An alternative noise simulation method that works on state vectors instead of density matrices.

**How it works:**
1. For each trajectory: at each noise channel, randomly pick one Kraus operator K_i with probability `tr(K_i rho K_i^dagger)`
2. Apply K_i to the pure state vector (collapse to one branch)
3. Repeat for many trajectories
4. Average measurement results across trajectories to reconstruct the effect of noise

**Why it matters:**
- Density matrix simulation scales as O(4^n) — impractical beyond ~12 qubits
- Trajectory simulation scales as O(2^n) per trajectory — same as statevec
- For N trajectories, total cost is O(N * 2^n) which can be cheaper than O(4^n) when N << 2^n
- Naturally parallelizable (each trajectory is independent)
- yao-rs already has `kraus_operators()` for every noise channel — the building blocks exist

**CLI integration:**
- Could be a new backend: `--backend trajectory`
- Requires a `--trajectories <N>` flag (number of Monte Carlo samples)
- Uses `--seed` for reproducibility
- Output is approximate (statistical), unlike the exact dm backend

**Open questions:**
- Not implemented in yao-rs today — needs new library code
- How to report statistical uncertainty in output? (error bars on probabilities/expectation values)
- Should trajectories be parallelized via rayon?
- Reference implementations: Qiskit Aer (automatic trajectory switching), QuTiP (mcsolve)

## 18. Yao.jl / yao-rs / yaosim Feature Comparison

Reference table for tracking feature parity across the three layers.

| Feature | Yao.jl | yao-rs | yaosim (v1 spec) |
|---|---|---|---|
| **State vector sim** | `apply!(reg, circuit)`, `reg \|> circuit` | `apply(circuit, state)`, `apply_inplace` | `--task statevector` |
| **Probabilities** | `probs(reg)` | `probs(state, locs)` | `--task probs [--qubits]` |
| **Measurement sampling** | `measure(reg; nshots=N)`, `measure!(reg)` | `measure(state, locs, nshots, rng)` | `--task sample --shots N` |
| **Non-Z measurement** | `Measure(n, operator=X)` (algebraic eigenspace) | Not implemented | Not in v1 (TBD #8) |
| **Expectation values** | `expect(op, reg)`, `expect(op, reg => circuit)` | `circuit_to_expectation(circuit, obs)` (TN, \|0> only) | `--task expect --obs "Z0Z1"` |
| **Overlap** | `reg1' * reg2` (inner product) | `circuit_to_overlap` (\|0> only) | `--task overlap --bra --ket` (statevec dot product) |
| **Fidelity** | `fidelity(reg1, reg2)` (pure + mixed) | Not implemented | Not in v1 (TBD #2) |
| **Trace distance** | `tracedist(reg1, reg2)` | Not implemented | Not planned |
| **Density matrix** | `DensityMatrix` register, `density_matrix(reg)` | TN-based DM via `circuit_to_einsum_dm` | Auto-detect noise → dm backend |
| **Noise channels** | `KrausChannel`, `MixedUnitaryChannel`, predefined errors | `NoiseChannel` enum (11 variants), `kraus_operators()`, `superop()` | Auto-detect, dm backend |
| **TN export** | `yao2einsum` (VectorMode, DensityMatrixMode, PauliBasisMode) | `circuit_to_einsum`, `circuit_to_einsum_dm` | Not in v1 (TBD #4) |
| **TN boundary states** | `initial_state=Dict(), final_state=Dict()` (arbitrary) | `circuit_to_einsum_with_boundary` (\|0> only) | Statevec path for arbitrary (TBD #11 for TN) |
| **TN contraction** | `contract(tn)` via OMEinsum.jl | `contract(tn)` via omeco + tch (torch feature) | Not in v1 (TBD #4) |
| **Easybuild** | `qft_circuit`, `phase_estimation_circuit`, `hadamard_test_circuit`, `swap_test_circuit` | `qft_circuit`, `variational_circuit`, `hadamard_test_circuit`, `swap_test_circuit` | `yaosim easybuild` subcommand |
| **QASM** | `YaoQASM.jl`, `OpenQASM.jl` (separate, unstable) | Not implemented | QASM 2.0 parser in yaosim-core |
| **JSON** | Not built-in | `circuit_to_json`, `circuit_from_json` | Native JSON input |
| **Visualization** | `vizcircuit` (SVG/PNG/PDF via Compose.jl) | `to_pdf` (PDF via Typst) | `yaosim show --format pdf\|svg` |
| **Parameters** | `dispatch!(circuit, params)`, `parameters(circuit)` | Not implemented | `{{template}}` substitution (TBD #6) |
| **AD/Gradients** | `expect'(op, reg => circuit)`, Zygote integration | Not implemented | Not planned |
| **Qudit support** | Qubits only for named gates | Full qudit (`dims: Vec<usize>`) | Inherited from yao-rs |
| **Circuit adjoint** | `circuit'` (adjoint) | `circuit.dagger()` | Inherited from yao-rs |
| **Stabilizer sim** | Not built-in | Not implemented | Backend listed, not implemented (TBD #16) |
| **Monte Carlo trajectory** | Not built-in | Not implemented | Not in v1 (TBD #17) |

**Key gaps in yao-rs vs Yao.jl:**
- No `dispatch!`-style parameter system (most impactful for variational algorithms)
- No AD/gradient support
- No arbitrary TN boundary states
- No non-Z-basis measurement
- No fidelity/trace distance
- No PauliBasisMode in TN export

**Where yao-rs/yaosim exceeds Yao.jl:**
- Full qudit support (per-site dimensions)
- Built-in JSON serialization
- CLI interface (Yao.jl is library-only)
- QASM parser (Yao.jl's is unstable/separate)
- Unix pipeline composability
