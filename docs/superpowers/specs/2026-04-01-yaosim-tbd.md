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

**Upstream status (2026-04-08):** `ArrayReg::fidelity(&self, other: &ArrayReg) -> f64` is implemented upstream. Returns `|<psi|phi>|^2`. Can be adopted directly.

**Open questions:**
- ~~Requires new library code~~ → Upstream has pure-state fidelity; density matrix fidelity (`F(rho, sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2`) still not implemented
- Output format: single scalar? Include phase info?
- CLI: `--task fidelity --reference ref.json` or `--task fidelity --bra state1.json --ket state2.json`?

## 3. Entanglement Entropy Task

`--task entropy --partition 0,1:2,3` — compute entanglement entropy via partial trace.

**Upstream status (2026-04-08):** `DensityMatrix::partial_tr(&self, traced_locs: &[usize])` and `DensityMatrix::von_neumann_entropy(&self) -> f64` are both implemented upstream. Also `DensityMatrix::purity(&self) -> f64`. The full pipeline (pure state → density matrix → partial trace → entropy) is available.

**Open questions:**
- ~~Requires new library code~~ → Upstream has partial trace + von Neumann entropy; can adopt
- Which entropy measures? Von Neumann only, or also Renyi?
- How to specify partitions on the CLI ergonomically?
- Also expose `--task purity` as a separate quick check?

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

**Upstream status (2026-04-08):** No parameterized circuit support upstream either. Template substitution remains the pragmatic CLI-layer approach.

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

## 19. Extended Operator Set (P0, P1, Pu, Pd)

Upstream yao-rs supports operators beyond Pauli: P0 (`|0⟩⟨0|`), P1 (`|1⟩⟨1|`), Pu (raising `|0⟩⟨1|`), Pd (lowering `|1⟩⟨0|`). Our yaosim spec only supports X, Y, Z, I.

**Why it matters:**
- Projectors P0/P1 enable post-selection probabilities: `<psi|P0(0)|psi>` = probability of qubit 0 being |0⟩
- Raising/lowering operators are standard in condensed matter Hamiltonians (e.g., Heisenberg model: `Pu(i)Pd(j) + Pd(i)Pu(j)`)
- Upstream CLI syntax: `P0(0)`, `Pu(0)Pd(1)` — already implemented and tested

**Open questions:**
- Add to yaosim v1 or defer?
- Site-indexed syntax: `P0_0` or `P0(0)` or `P00` (ambiguous with P0 on qubit 0)?
- Upstream uses parenthesized syntax `P0(0)` — avoids ambiguity. Consider adopting for extended ops while keeping `Z0Z1` for Pauli.

## 20. QASM Parser: Build vs Reuse `openqasm` Crate

Our spec says "build QASM 2.0 parser in yaosim-core". Upstream uses the `openqasm` crate (external) which provides:
- Full QASM 2.0 parsing with `SourceCache` and `Parser`
- Type checking via `program.type_check()`
- Gate linearization to U+CX primitives via `Linearize` (depth=100)
- Bundled `qelib1.inc` with extensible `FilePolicy`
- `GateWriter` trait for custom circuit building

**Upstream approach:** Extend `qelib1.inc` with modern Qiskit gates (swap, sx, p, cp, crx, cry, etc.) via `EXTRA_GATES` string appended to the bundled file.

**Trade-offs:**
- **Reuse `openqasm` crate:** Much less code, battle-tested parser, handles edge cases. Decomposition to U+CX means all gates become Rz/Ry/CX (lossy — original gate names lost).
- **Build our own:** Full control, preserve original gate names (H stays H, not Rz·Ry·Rz), can add noise channel parsing, better error messages with line numbers. More work.

**Open questions:**
- Can we use `openqasm` crate but avoid the Rz/Ry/CX decomposition? (Needs custom `GateWriter` that maps to yao-rs gates directly instead of linearizing)
- If we build our own, how much of `openqasm`'s edge case handling do we need?
- Upstream's approach loses gate identity (H becomes 3 rotation gates) — acceptable for simulation but bad for `yaosim info` and `yaosim show`

## 21. Binary State Format for Pipeline Efficiency

Upstream implements `yao-state-v1`: JSON header line + raw little-endian Complex128 bytes. This is much more efficient than JSON for intermediate state piping.

**Format:**
```
{"format":"yao-state-v1","num_qubits":3,"dims":[2,2,2],"num_elements":8,"dtype":"complex128"}\n
<raw bytes: 8 * 16 = 128 bytes of f64 pairs>
```

**Our spec currently:** JSON state files only, product-state strings for `--init-state`.

**Proposal:** Support both formats:
- `--init-state state.json` — JSON (human-editable, small circuits)
- `--init-state state.bin` — binary (efficient, pipeline use)
- `--output-state state.bin` — binary output for piping
- Detect format by file extension or magic bytes

**Open questions:**
- Should yaosim support a `simulate` subcommand (like upstream) that outputs binary state for pipe chains?
- Or is `yaosim run bell.json --output-state state.bin` sufficient?
- Binary format needs qudit extension (our fork has `dims: Vec<usize>` not just `vec![2; n]`)

## 22. Fetch Subcommand (QASMBench Download)

Upstream implements `yao fetch qasmbench <name>` to download benchmark circuits from the QASMBench GitHub repository.

**Features:**
- `yao fetch qasmbench list` — list all available circuits (queries GitHub API)
- `yao fetch qasmbench list --scale small` — filter by scale
- `yao fetch qasmbench grover` — download by name (auto-detect scale)
- Pipeline: `yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100`

**Open questions:**
- Add to yaosim v1 or defer?
- Extend to other sources beyond QASMBench? (e.g., MQTBench, Queko)
- Cache downloaded circuits locally?
- Requires network access — how to handle offline mode?

## 23. Shell Completions

Upstream implements `yao completions [shell]` using `clap_complete`. Trivial to add with clap derive.

**Usage:** `eval "$(yaosim completions)"` added to `.bashrc`/`.zshrc`.

**Open questions:**
- Add to v1 (trivial) or defer?
- Recommend adding to v1 — near-zero effort with clap_complete, significant UX improvement.

## 24. Direct DensityMatrix Type vs TN-Only DM Path

Upstream has a first-class `DensityMatrix` type with:
- `DensityMatrix::from_reg(reg)` — pure → mixed
- `DensityMatrix::mixed(weights, regs)` — statistical mixture
- `DensityMatrix::partial_tr(traced_locs)` — partial trace
- `DensityMatrix::von_neumann_entropy()` — von Neumann entropy
- `DensityMatrix::purity()` — purity Tr(ρ²)
- `impl Register for DensityMatrix` — apply circuits via column-wise statevec simulation

Our fork only has TN-based DM via `circuit_to_einsum_dm`. This is the tensor network approach (good for large circuits with structure) but cannot compute partial trace, entropy, or purity.

**Trade-off:**
- Direct DM: O(4^n) memory, supports all DM operations (trace, entropy, etc.), exact
- TN DM: memory depends on circuit structure, only computes what the einsum asks for, needs contraction

**Proposal:** Port upstream's `DensityMatrix` for small-circuit DM tasks (entropy, purity, partial trace). Keep TN path for large circuits. CLI auto-selects based on qubit count.

**Open questions:**
- At what qubit count should the CLI switch from direct DM to TN? (~12 qubits = 16M entries)
- Should `--backend dm` always mean direct DM, with TN as a separate `--backend tn-dm`?
- Port upstream code directly or reimplement with our qudit-compatible State type?

## 25. Upstream Architecture Divergence: Qubit-Only vs Qudit

Upstream dropped qudit support on 2026-04-07, refactoring to qubit-only `ArrayReg` backed by `Vec<Complex64>` with `nbits: usize`. Our fork retains `State` with `dims: Vec<usize>` supporting arbitrary per-site dimensions.

**Implications for yaosim:**
- Our qudit support is a differentiator but complicates porting upstream features
- Upstream's `bitbasis` crate and bit-manipulation fast paths assume d=2
- Features like `DensityMatrix`, `expect_arrayreg`, `fidelity` are qubit-only upstream
- Porting them to our fork requires generalizing from `1 << nbits` to `dims.iter().product()`

**Open questions:**
- Is qudit support worth the maintenance cost for yaosim v1?
- Can we have a fast qubit path (like upstream) with qudit fallback?
- Should named gates (X, Y, Z, H, etc.) remain qubit-only while `Custom` gates handle qudits?

## 18. Yao.jl / yao-rs (upstream) / yao-rs (our fork) / yaosim Feature Comparison

Reference table updated 2026-04-09 after reviewing upstream GiggleLiu/yao-rs (commit e5c509d, 2026-04-08).

**Key upstream changes since our fork:**
- Dropped qudit support → qubit-only `ArrayReg` (April 7 refactor)
- Added `openqasm` crate integration for QASM 2.0 import/export
- Added `DensityMatrix` with `partial_tr`, `von_neumann_entropy`, `purity`
- Added `expect_arrayreg()` and `expect_dm()` (statevec-based expectation)
- Added `ArrayReg::fidelity()`
- Added `bitbasis` crate for optimized bit manipulation
- Added full CLI (`yao-cli`) with Unix-pipe architecture
- Added `yao fetch qasmbench` for downloading benchmark circuits

| Feature | Yao.jl | yao-rs (upstream) | yao-rs (our fork) | yaosim (v1 spec) |
|---|---|---|---|---|
| **State vector sim** | `apply!(reg, circuit)` | `apply(&circuit, &reg)` on `ArrayReg` | `apply(circuit, state)` on `State` | `--task statevector` |
| **Register type** | `ArrayReg` (qubit) | `ArrayReg` (qubit-only, `Vec<Complex64>`) | `State` (qudit, `Array1<Complex64>`, `dims`) | Inherited |
| **Probabilities** | `probs(reg)` | `probs()` via `measure` module | `probs(state, locs)` | `--task probs [--qubits]` |
| **Measurement** | `measure(reg; nshots=N)` | `measure_with_postprocess()` (NoPostProcess/ResetTo/RemoveMeasured) | `measure(state, locs, nshots, rng)` | `--task sample --shots N` |
| **Non-Z measurement** | `Measure(n, operator=X)` | Not implemented | Not implemented | Not in v1 (TBD #8) |
| **Expectation (pure)** | `expect(op, reg)` | `expect_arrayreg(reg, op)` — works on any state | `circuit_to_expectation` (TN, \|0⟩ only) | `--task expect --obs` (statevec fallback for non-\|0⟩) |
| **Expectation (mixed)** | `expect(op, dm)` | `expect_dm(dm, op)` | Not implemented | Auto via dm backend |
| **Operators** | X, Y, Z, I | X, Y, Z, I, **P0, P1, Pu, Pd** | X, Y, Z, I | Pauli only (TBD #19) |
| **Overlap** | `reg1' * reg2` | `circuit_to_overlap` (\|0⟩ only) | `circuit_to_overlap` (\|0⟩ only) | `--task overlap --bra --ket` (statevec dot) |
| **Fidelity** | `fidelity(reg1, reg2)` (pure + mixed) | `ArrayReg::fidelity()` (pure only) | Not implemented | TBD #2 (can adopt upstream) |
| **Trace distance** | `tracedist(reg1, reg2)` | Not implemented | Not implemented | Not planned |
| **Density matrix** | `DensityMatrix` register | `DensityMatrix` with `partial_tr`, `von_neumann_entropy`, `purity`, `mixed()` | TN-based via `circuit_to_einsum_dm` | Auto-detect noise → dm backend |
| **Noise channels** | `KrausChannel`, predefined | `NoiseChannel` (11 variants) | `NoiseChannel` (11 variants) | Auto-detect, dm backend |
| **TN export** | `yao2einsum` (3 modes) | `circuit_to_einsum`, `circuit_to_einsum_dm` | `circuit_to_einsum`, `circuit_to_einsum_dm` | Not in v1 (TBD #4) |
| **TN contraction** | `contract(tn)` via OMEinsum.jl | `contract(tn)` via omeco + tch | `contract(tn)` via omeco + tch | Not in v1 (TBD #4) |
| **Easybuild** | `qft_circuit`, etc. | `qft_circuit`, `variational_circuit`, `phase_estimation_circuit`, `rand_supremacy2d`, `rand_google53`, etc. | Same set | `yaosim easybuild` subcommand |
| **QASM** | `YaoQASM.jl` (unstable) | `openqasm` crate: `from_qasm()`, `to_qasm()` | Not implemented | Originally "build in yaosim-core" (TBD #20) |
| **JSON** | Not built-in | `circuit_to_json`, `circuit_from_json` (no noise) | Same | Native JSON input |
| **Visualization** | `vizcircuit` (SVG/PNG/PDF) | `to_pdf` (Typst) | `to_pdf` (Typst) | `yaosim show --format pdf\|svg` |
| **Parameters** | `dispatch!(circuit, params)` | Not implemented | Not implemented | `{{template}}` substitution (TBD #6) |
| **Parameter sweep** | Not built-in | Not implemented | Not implemented | `--sweep` flag (**unique to yaosim**) |
| **AD/Gradients** | `expect'(op, reg => circuit)` | Not implemented | Not implemented | Not planned |
| **Qudit support** | Qubits only for named gates | **Dropped** (qubit-only since April 7) | Full qudit (`dims: Vec<usize>`) | Inherited from our fork |
| **Circuit adjoint** | `circuit'` | `circuit.dagger()` | `circuit.dagger()` | Inherited |
| **CLI** | None (library-only) | `yao` CLI (Unix-pipe: simulate\|measure\|probs\|expect\|run) | None | `yaosim` CLI (task-oriented) |
| **CLI operator syntax** | N/A | `Z(0)Z(1)`, `0.5*Z(0)Z(1) + X(0)` | N/A | `Z0Z1`, `0.5 X0Z1 + X0X1` |
| **State I/O** | N/A | Binary `yao-state-v1` (JSON header + raw bytes) | N/A | JSON state files (TBD #21) |
| **Fetch benchmarks** | N/A | `yao fetch qasmbench` | N/A | Not in v1 (TBD #22) |
| **Shell completions** | N/A | `yao completions` | N/A | Not in v1 (TBD #23) |
| **Stabilizer sim** | Not built-in | Not implemented | Not implemented | TBD #16 |
| **Monte Carlo trajectory** | Not built-in | Not implemented | Not implemented | TBD #17 |

**Key gaps in our fork vs upstream:**
- No `DensityMatrix` type (partial trace, entropy, purity, mixed states)
- No `expect_arrayreg` / `expect_dm` (statevec-based expectation for arbitrary states)
- No `ArrayReg::fidelity()`
- No QASM support
- No `bitbasis` crate (optimized bit manipulation)

**Where our fork / yaosim exceeds upstream:**
- Full qudit support preserved (per-site dimensions)
- Task-oriented CLI design (more discoverable than pipe architecture)
- Composable add-ons (`--with-probs --sample N --obs` in one command)
- Human-readable output by default (histograms, top-k, formatted tables)
- Parameter sweep (`--sweep`)
- Overlap with arbitrary states (`--bra`/`--ket`)
- Dual observable syntax (site-indexed + dense positional)
