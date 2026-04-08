# Benchmark Suite: yao-rs vs Yao.jl

**Date:** 2026-04-08
**Goal:** Validate correctness and compare performance of yao-rs against the reference Yao.jl implementation across 3 benchmark tasks covering single gates, QFT circuits, and noisy density-matrix simulation.

## Parameters

- **Qubit range:** 4–25 (tasks 1–2), 4–10 (task 3, density matrix)
- **Ground truth:** Julia scripts run manually by the developer; JSON output committed to `benchmarks/data/`
- **Correctness tolerance:** 1e-10 for state vector comparisons, 1e-8 for derived quantities (entropy, expectation values)

---

## Task 1: Single Gate Benchmarks

### Purpose

Validate and benchmark individual gate application against Yao.jl, organized by the dispatch path in `instruct_qubit.rs`.

### Groups

| Group | Gates | Dispatch path |
|-------|-------|--------------|
| 1q uncontrolled | X, H, T, Rx(0.5), Rz(0.5) | `instruct_x`, `instruct_1q`, `instruct_1q_diag` |
| 2q controlled | CNOT, CRx(0.5) | `instruct_x_controlled`, `instruct_1q_controlled` |
| Multi-controlled | Toffoli (CCX) | `instruct_1q_controlled` with 2 control qubits |

### Initial State

Deterministic formula (no RNG). Each amplitude is a complex number constructed from the basis-state index `k`:
```
state[k] = Complex64(cos(0.1 * k), sin(0.2 * k))   (unnormalized)
state = state / norm(state)                           (normalized)
```

Both Julia and Rust compute this identically from the index `k = 0, 1, ..., 2^n - 1`.

### Gate Placement

Matches Yao.jl benchmark convention:
- 1q gates: applied to qubit 2 (0-indexed)
- 2q controlled: control=2, target=3
- Toffoli: controls=(2,3), target=1

### Qubit Sweep

4, 5, 6, ..., 25 qubits for all gates.

### Ground Truth Format

Three JSON files:

**`single_gate_1q.json`:**
```json
{
  "X": {"4": [re0, im0, re1, im1, ...], "5": [...], ...},
  "H": {"4": [...], ...},
  "T": {"4": [...], ...},
  "Rx_0.5": {"4": [...], ...},
  "Rz_0.5": {"4": [...], ...}
}
```

**`single_gate_2q.json`:**
```json
{
  "CNOT": {"4": [...], ...},
  "CRx_0.5": {"4": [...], ...}
}
```

**`single_gate_multi.json`:**
```json
{
  "Toffoli": {"4": [...], ...}
}
```

### Correctness Test

For each gate and qubit count:
1. Construct the deterministic initial state
2. Build a single-gate circuit using `put()` or `control()`
3. Apply circuit to state
4. Compare resulting state vector element-wise against JSON data (tolerance 1e-10)

### Performance Benchmark

Three Criterion groups (`bench_gates_1q`, `bench_gates_2q`, `bench_gates_multi`), each parameterized over `(gate, nqubits)`.

---

## Task 2: QFT Circuit

### Purpose

Validate and benchmark the full QFT circuit, which exercises H gates and all controlled-Phase rotations.

### Circuit

Standard QFT (matching `easybuild::qft_circuit`):
- For each qubit i: H(i), then controlled-Phase(2pi/2^(j+1)) with control=i+j for j in 1..(n-i)
- No final SWAP layer (matches Yao.jl convention)

### Initial State

Product state `|1>` = `|000...01>` (qubit 0 is set). This gives nontrivial output:
```
state[k] = (1/sqrt(2^n)) * exp(2*pi*i*k / 2^n)
```

This is analytically known, providing a secondary verification independent of Julia.

### Qubit Sweep

4, 5, 6, ..., 25 qubits.

### Ground Truth Format

**`qft.json`:**
```json
{
  "4": [re0, im0, re1, im1, ...],
  "5": [...],
  ...
  "25": [...]
}
```

### Correctness Test

1. Construct `|1>` state via `ArrayReg::product_state`
2. Build QFT circuit via `qft_circuit(n)`
3. Apply circuit
4. Compare against Julia ground truth (tolerance 1e-10)
5. Also cross-check against analytical formula as a secondary verification

### Performance Benchmark

One Criterion group (`bench_qft`) parameterized over `nqubits`.

---

## Task 3: Noisy Circuit — Density Matrix Evolution

### Purpose

Validate density matrix evolution with noise channels against Yao.jl. Tests `DensityMatrix`, `NoiseChannel` Kraus operators, partial trace, entropy, and `expect_dm`.

### Circuit

Fixed structure for each qubit count:
1. H on all qubits
2. CNOT chain: control(i, i+1, X) for i in 0..(n-1)
3. Depolarizing noise (p=0.01) on each qubit
4. Rz(0.3) on all qubits
5. Amplitude damping (gamma=0.05) on each qubit

### Qubit Sweep

4, 5, 6, 7, 8, 9, 10 (density matrix is 2^n x 2^n, so 10 qubits = 1M entries).

### Implementation Note

`DensityMatrix::apply` currently handles `Gate` elements but skips `Channel` elements. This benchmark requires implementing Kraus-based channel evolution for density matrices:

```
rho -> sum_i K_i * rho * K_i^dagger
```

This is a feature gap to be filled as part of this work. The Julia script applies channels via `Yao.apply!(dm, channel)`.

### Quantities Validated

For each qubit count, the Julia script exports:

| Quantity | Description | Tolerance |
|----------|------------|-----------|
| `density_matrix` | Full dm elements (n <= 6 only) | 1e-10 |
| `trace` | tr(rho) — must be 1.0 | 1e-12 |
| `purity` | tr(rho^2) — decreases with noise | 1e-8 |
| `reduced_dm` | partial_tr over last qubit | 1e-10 |
| `entropy` | von_neumann_entropy of reduced state | 1e-6 |
| `expect_ising` | <H_ising> where H = sum Z_i Z_{i+1} + 0.5 sum X_i | 1e-8 |

### Ground Truth Format

**`noisy_circuit.json`:**
```json
{
  "4": {
    "density_matrix": [re00, im00, re01, im01, ...],
    "trace": 1.0,
    "purity": 0.987...,
    "reduced_dm": [re00, im00, ...],
    "entropy": 0.123...,
    "expect_ising": {"re": -1.23, "im": 0.0001}
  },
  "5": { ... },
  ...
}
```

For n > 6, `density_matrix` and `reduced_dm` fields are omitted (too large); only scalar quantities are compared.

### Performance Benchmark

One Criterion group (`bench_noisy_dm`) parameterized over `nqubits`, measuring total evolution time (gates + channels).

---

## File Organization

```
benchmarks/
  julia/
    Project.toml                  # Yao.jl, YaoBlocks, BenchmarkTools deps
    generate_ground_truth.jl      # Generates ground truth JSON + Julia timings
  data/
    single_gate_1q.json           # Task 1 ground truth
    single_gate_2q.json
    single_gate_multi.json
    qft.json                      # Task 2 ground truth
    noisy_circuit.json            # Task 3 ground truth
    timings.json                  # Julia performance timings (ns)
  compare.py                      # Python script: reads Julia timings + Criterion output, prints table

benches/
  apply.rs                        # Existing — keep unchanged
  gates.rs                        # Task 1 performance
  qft.rs                          # Task 2 performance
  density.rs                      # Task 3 performance

tests/suites/
  benchmark_validation.rs         # Correctness tests for all 3 tasks
```

### Julia Script Structure

`generate_ground_truth.jl` is a single self-contained script:

1. Defines the deterministic initial state formula
2. Task 1: loops over gates and qubit counts, applies each gate, exports state vectors
3. Task 2: loops over qubit counts, builds QFT, applies to |1>, exports state vectors
4. Task 3: loops over qubit counts, builds noisy circuit, evolves density matrix, exports all quantities
5. Writes all 5 JSON files to `benchmarks/data/`

### Rust Test Structure

`benchmark_validation.rs` loads JSON files via `include_str!` or at runtime, deserializes, and runs parameterized tests:

- `test_single_gate_1q_{gate}_{nqubits}` — one test per (gate, nqubits) combo
- `test_qft_{nqubits}` — one test per qubit count
- `test_noisy_dm_{nqubits}` — one test per qubit count, checking all quantities

### Running

```bash
# Generate ground truth + Julia timings (manual, one-time)
cd benchmarks/julia && julia --project=. generate_ground_truth.jl

# Correctness validation
cargo test --all-features benchmark_validation

# Performance benchmarks (Rust)
cargo bench --bench gates
cargo bench --bench qft
cargo bench --bench density

# Compare performance (reads Julia timings + Criterion output)
python3 benchmarks/compare.py
```

---

## Performance Comparison: Rust vs Julia

### Julia Timing

The Julia script also benchmarks each task using `BenchmarkTools.@benchmark`, taking the **minimum time** (matching Yao.jl's convention). Timings are exported alongside ground truth:

**`timings.json`:**
```json
{
  "single_gate_1q": {
    "X": {"4": 123.4, "5": 234.5, ...},
    "H": {"4": ..., ...},
    ...
  },
  "single_gate_2q": {
    "CNOT": {"4": ..., ...},
    "CRx_0.5": {"4": ..., ...}
  },
  "single_gate_multi": {
    "Toffoli": {"4": ..., ...}
  },
  "qft": {"4": ..., "5": ..., ...},
  "noisy_dm": {"4": ..., "5": ..., ...}
}
```

All times are in **nanoseconds** (minimum of benchmark samples).

### Rust Timing

Criterion writes results to `target/criterion/`. Each benchmark produces `estimates.json` with mean/median/min timings.

### Comparison Report

A small Python script (`benchmarks/compare.py`) reads:
1. `benchmarks/data/timings.json` (Julia)
2. `target/criterion/*/new/estimates.json` (Rust)

Outputs a markdown table to stdout:

```
| Task          | Gate/Circuit | Qubits | Julia (ns) | Rust (ns) | Speedup |
|---------------|-------------|--------|------------|-----------|---------|
| single_gate   | X           | 4      | 123        | 45        | 2.7x    |
| single_gate   | X           | 8      | 456        | 120       | 3.8x    |
| ...           | ...         | ...    | ...        | ...       | ...     |
| qft           | QFT         | 20     | 1234567    | 456789    | 2.7x    |
| noisy_dm      | full        | 8      | 9876543    | 3456789   | 2.9x    |
```

### Julia Script Timing Section

The Julia script uses warmup + benchmark for each task:

```julia
using BenchmarkTools

# Per gate/circuit:
result = @benchmark apply!(reg_copy, circuit) setup=(reg_copy = copy(reg))
timing_ns = minimum(result).time  # nanoseconds
```

---

## Feature Gap: Density Matrix Channel Evolution

Task 3 requires extending `DensityMatrix` to apply `Channel` circuit elements. The implementation:

1. In `DensityMatrix::apply()`, match on `CircuitElement::Channel(pc)` instead of skipping it
2. Extract Kraus operators via `pc.channel.kraus_operators()`
3. Apply Kraus evolution: `rho_new = sum_i K_i rho K_i^dag` on the subsystem specified by `pc.locs`
4. This is analogous to the gate application but with multiple Kraus operators summed

This is the only new feature required. All other code paths already exist.
