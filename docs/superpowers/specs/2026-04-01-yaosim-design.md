# yaosim CLI — Design Specification

**Date:** 2026-04-01
**Status:** Draft

A command-line tool for quantum circuit simulation built on the yao-rs library. yaosim fills an unoccupied niche: there is no polished quantum circuit simulation CLI in the ecosystem (Qiskit, Cirq, QuEST, qsim are all library-only). yaosim brings quantum simulation to the Unix pipeline.

---

## 1. Project Structure

**Name:** `yaosim`

**Workspace layout:** Two new crates added to the yao-rs workspace:

- **`yaosim-core/`** (library crate): QASM 2.0 parser, Pauli expression parser, output formatting, circuit analysis (depth, gate count). Reusable by yao-rs-py or other consumers.
- **`yaosim/`** (binary crate): Thin CLI layer using `clap` (derive). Depends on `yaosim-core` + `yao-rs`.

**CLI framework:** `clap` with derive macros.

**Feature flags** (inherited from yao-rs):
- `typst` — enables `viz` subcommand (PDF/SVG rendering)
- `torch` — enables TN contraction
- `parallel` — enables rayon parallelism

---

## 2. Subcommands

```
yaosim run <circuit>        # Simulate circuit, compute tasks
yaosim easybuild <template>  # Generate built-in circuit JSON to stdout
yaosim convert <input>      # Convert between JSON and OpenQASM 2.0
yaosim show <input>          # Render circuit diagram (PDF/SVG)
yaosim info <input>         # Circuit statistics
```

All subcommands accept `-` for stdin where a circuit file is expected.

---

## 3. `yaosim run` — Simulation

### 3.1 Common Arguments

| Argument | Short | Description | Default |
|---|---|---|---|
| `<circuit>` | — | Circuit file (.json/.qasm) or `-` for stdin | required |
| `--task <T>` | `-t` | Primary computation task | `statevector` |
| `--init-state <S>` | `-i` | Product state `"0,1,0"` or state file (.json) | \|0...0> |
| `--backend <B>` | `-b` | `statevec\|tn\|dm` | auto-detect |
| `--circuit <tmpl>` | `-c` | Built-in circuit shorthand | — |
| `--sweep "<spec>"` | — | Parameter sweep (see section 3.5) | — |
| `--json` | `-j` | JSON output to stdout | — |
| `--csv` | — | CSV output to stdout | — |
| `--quiet` | `-q` | Suppress human-readable banners, data only | — |
| `--verbose` | `-v` | Full output, no truncation | — |
| `-o <file>` | — | Write output to file | stdout |
| `--seed <u64>` | — | RNG seed for reproducibility | random |

### 3.2 Tasks

#### `statevector` (default)

Compute the output state U|psi>.

| Argument | Description | Default |
|---|---|---|
| `--threshold <f>` | Hide amplitudes with \|a\|^2 < threshold | `0.01` |
| `--top <k>` | Show only top-k amplitudes by magnitude | `10` |

**Shortcut:** `--applyto <state>` expands to `--task statevector --init-state <state>`

#### `probs`

Compute probability distribution P(x) = \|<x\|psi>\|^2 over all basis states, or marginal probabilities on a subset.

| Argument | Description | Default |
|---|---|---|
| `--qubits <list>` | Compute marginal on these qubits (e.g., `0,1`) | all qubits |
| `--threshold <f>` | Hide entries with P < threshold | `0.001` |

#### `sample`

Z-basis projective measurement — sample N bitstrings from the output state.

| Argument | Description | Default |
|---|---|---|
| `--shots <N>` | Number of measurement shots | `1024` |
| `--qubits <list>` | Measure subset of qubits | all qubits |

**As primary task:** `--task sample --shots <N>` runs sampling as the main output.
**As add-on:** `--sample <N>` on any other task adds a sampling section to the output.

#### `expect`

Compute expectation value <psi|O|psi> for a Pauli observable.

| Argument | Description | Default |
|---|---|---|
| `--obs <expr\|file>` | Inline Pauli expression or JSON file path | required |

**As primary task:** `--task expect --obs "<expr>"` runs expectation as the main output.
**As add-on:** `--obs "<expr>"` on any other task adds an expectation section to the output.

#### `overlap`

Compute transition amplitude <bra|U|ket>.

| Argument | Description | Default |
|---|---|---|
| `--bra <state>` | Bra state: product state `"0,1,0"` or state file | \|0...0> |
| `--ket <state>` | Ket state (input to circuit): product state or file | \|0...0> |

`--init-state` sets `--ket` if `--ket` is not explicitly provided.

**Shortcut:** `--overlapwithzero` expands to `--task overlap` (both bra and ket default to |0...0>).

**Implementation:** For v1, arbitrary <phi|U|psi> is computed via the statevec path: `apply(circuit, ket)` then inner product with bra (`sum of conj(phi_i) * psi_i`). This works because `State::data` is public. The TN path (`circuit_to_einsum_with_boundary`) only supports |0> boundaries — extending it to support arbitrary boundary states (matching Yao.jl's dictionary-based interface) is tracked in the TBD doc.

### 3.3 Composable Add-ons

These flags add extra outputs on top of any primary task, without conflict:

| Add-on | Description |
|---|---|
| `--with-probs` | Also show probability distribution |
| `--sample <N>` | Also run N measurement samples |
| `--obs "<expr>"` | Also compute expectation value of observable (when not primary task) |

Example:
```bash
yaosim run bell.json --task statevector --with-probs --sample 1000 --obs "Z0Z1"
# Output: state vector + probabilities + 1000 samples + expectation value
```

### 3.4 Backend Selection

**Available backends:**

| Backend | Description | Status |
|---|---|---|
| `statevec` | Dense state vector simulation via `apply()` | Implemented |
| `dm` | Density matrix via tensor network (`circuit_to_einsum_dm`) | Implemented |
| `stabilizer` | Clifford/stabilizer tableau (Gottesman-Knill, O(n^2) per gate) | Not yet implemented |

**Auto-selection logic:**
- No noise channels → `statevec`
- Noise channels present → `dm`
- `--backend` flag overrides auto-detection
- `--backend stabilizer` on a non-Clifford circuit → error with message listing non-Clifford gates
- Backend selection messages appear on stderr (e.g., `[backend] noise channels detected -- switching to dm`)

### 3.5 Parameter Sweep

Circuit files use `{{param_name}}` template placeholders in parameter positions. The `--sweep` flag specifies how to substitute them.

**Template format in circuit JSON:**
```json
{"type": "gate", "gate": "Ry", "params": ["{{theta_0}}"], "targets": [0]}
```

**Sweep syntax:**

Single parameter:
```bash
--sweep "theta_0=0.0:0.1:3.14"
# Sweeps theta_0 from 0.0 to 3.14 in steps of 0.1
```

Multiple parameters tied together (same sweep values):
```bash
--sweep "theta_0=theta_1=theta_2=0.0:0.1:3.14"
# All three parameters take the same value at each sweep point
```

Multiple independent parameters:
```bash
--sweep "theta_0=0.0:0.1:3.14" --sweep "phi=0.0:0.5:6.28"
# Grid sweep over all combinations
```

**Format:** `param=start:step:end` where start, step, end are floats. Multiple `=`-chained parameter names share the same sweep range.

If any `{{param}}` placeholders remain unresolved after sweep substitution, the CLI errors and lists the unresolved names.

---

## 4. `yaosim easybuild` — Built-in Circuits

Outputs circuit JSON to stdout.

```
yaosim easybuild qft --nqubits <N>
yaosim easybuild variational --nqubits <N> --nlayers <L> --topology ring|square
yaosim easybuild hadamard-test --gate <gate>
yaosim easybuild swap-test --nqubits <N>
```

Wraps yao-rs `easybuild` module: `qft_circuit`, `variational_circuit`, `hadamard_test_circuit`, `swap_test_circuit`, `pair_ring`, `pair_square`.

Also available as shorthand on `run`:
```bash
yaosim run --circuit qft --nqubits 4
```

---

## 5. `yaosim convert` — Format Conversion

```bash
yaosim convert input.qasm --to json [-o output.json]
yaosim convert input.json --to qasm [-o output.qasm]
```

Format inferred from input file extension. Output format specified by `--to`.

---

## 6. `yaosim show` — Circuit Visualization

```bash
yaosim show circuit.json --format pdf              # PDF to stdout
yaosim show circuit.json --format svg              # SVG to stdout
yaosim show circuit.json --format pdf -o diagram.pdf   # PDF to file
yaosim show circuit.json --format svg -o diagram.svg   # SVG to file
yaosim show circuit.json --format svg | open -f -a Safari  # pipe to viewer
```

| Argument | Short | Description | Default |
|---|---|---|---|
| `<circuit>` | — | Circuit file (.json/.qasm) or `-` for stdin | required |
| `--format <F>` | `-f` | Output format: `pdf` or `svg` | required |
| `-o <file>` | — | Write to file instead of stdout | stdout |

`--format` is required — no inference from file extension. Output goes to stdout by default (binary PDF or text SVG), enabling pipe chains. Requires `typst` feature flag; graceful error message if not compiled with it.

Uses yao-rs Typst-based rendering (embedded Typst compiler + Quill circuit visualization package).

---

## 7. `yaosim info` — Circuit Analysis

```bash
yaosim info circuit.json [--json]
```

Output: qubit count, gate count (by type), circuit depth, noise channel count, parameter placeholders (if template).

---

## 8. Input Formats

### 8.1 JSON

Existing yao-rs `circuit_to_json`/`circuit_from_json` format:

```json
{
  "num_qubits": 2,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0], "control_configs": [true]}
  ]
}
```

Noise channels:
```json
{"type": "channel", "noise": "depolarizing", "params": {"n": 1, "p": 0.05}, "locs": [0]}
```

### 8.2 OpenQASM 2.0

Parser built in `yaosim-core`. Extensible architecture for future QASM 3.0 / other DSLs.

**Supported features:**

| QASM 2.0 Feature | Support | Mapping |
|---|---|---|
| `qreg n[k]` | Accept | `Circuit::new(dims: vec![2; k])`. Multiple qregs concatenated into flat index space. |
| `creg c[k]` | Accept (track) | Parse and store as metadata for measure/if bookkeeping. |
| `include "qelib1.inc"` | Accept (bundled) | Hardcoded mapping table (see below). Reject unknown includes. |
| Standard gates | Accept | See gate mapping table. |
| `gate mygate(params) q { ... }` | Inline-expand | Recursively substitute calls. Depth-limit ~64. |
| `barrier q` | Accept as Annotation | `CircuitElement::Annotation`. No-op in execution. |
| `measure q -> c` | Accept (partial) | Emit marker. Mid-circuit measurement with classical feedback out of scope. |
| `reset q` | Reject with error | No yao-rs equivalent for mid-circuit reset. |
| `if (creg==val) gate` | Warn and skip | Classical feedback has no yao-rs execution model. |
| `opaque` | Warn and skip | No matrix available. |
| Math in params | Parse subset | `pi`, basic arithmetic (`+`,`-`,`*`,`/`), `cos`, `sin`, `sqrt`. |

**qelib1.inc gate mapping:**

| QASM gate | yao-rs equivalent |
|---|---|
| `x`, `y`, `z`, `h`, `s`, `t` | `Gate::X/Y/Z/H/S/T` |
| `sdg` | `Gate::Phase(-pi/2)` |
| `tdg` | `Gate::Phase(-pi/4)` |
| `rx(t)`, `ry(t)`, `rz(t)` | `Gate::Rx/Ry/Rz(t)` |
| `p(t)` / `u1(t)` | `Gate::Phase(t)` |
| `cx` / `CX` | `control([ctrl], [target], Gate::X)` |
| `cz` | `control([ctrl], [target], Gate::Z)` |
| `swap` | `Gate::SWAP` |
| `sx` | `Gate::SqrtX` |
| `u2(phi, lam)`, `u3(theta, phi, lam)` | `Gate::Custom` (evaluate matrix numerically) |
| `id` | Skip (identity gate) |

**Critical: `rz(t)` = `diag(e^{-it/2}, e^{it/2})` vs `p(t)`/`u1(t)` = `diag(1, e^{it})`. These must not be conflated — wrong mapping silently shifts rotation angles.**

**Error messages** must include source line number, offending token, and one-sentence hint.

### 8.3 stdin

All subcommands accept `-` to read circuit from stdin, enabling pipe chains:
```bash
yaosim easybuild qft --nqubits 4 | yaosim run - --task probs
```

---

## 9. Observable Specification

### 9.1 Inline Pauli Expression

Two syntaxes accepted:

**Site-indexed (OpenFermion-style, recommended):**
```
"0.5 X0Z1 + X0X1"
"Z0 + Z1 + Z2"
"Z0Z1"
```
Operator letter immediately followed by site index. Terms separated by `+` or `-`. Optional real coefficient before each term.

**Dense positional (Qiskit-style, for small circuits):**
```
"0.5 XZ + XX"
"ZZI"
```
String length must equal number of qubits. Left-to-right = qubit 0 (NOT Qiskit's right-to-left convention). Valid chars: `I`, `X`, `Y`, `Z`.

**Parser outputs `OperatorPolynomial`** matching yao-rs's internal representation.

**Qudit guard:** Pauli operators (X, Y, Z) are only valid for d=2 sites. Parser errors if applied to a qudit circuit with d > 2 on the referenced site.

### 9.2 JSON File

For complex Hamiltonians, pass a file path to `--obs`. The file must match the `OperatorPolynomial` serde format:

```json
{
  "coeffs": [{"re": 0.5, "im": 0.0}, {"re": 1.0, "im": 0.0}],
  "opstrings": [
    {"ops": [[0, "X"], [1, "Z"]]},
    {"ops": [[0, "X"], [1, "X"]]}
  ]
}
```

### 9.3 Ecosystem Context

Research findings on Pauli string formats across the ecosystem:

| Framework | Style | Example | Qubit ordering |
|---|---|---|---|
| **OpenFermion** | Site-indexed sparse | `X0 Z3 Y4` | Explicit indices |
| **Qiskit** | Dense positional | `XIZIY` | Right-to-left (qubit 0 = rightmost) |
| **STIM** | Dense positional | `+IXYZ` | Left-to-right |
| **PennyLane** | Python objects | `PauliWord({0:"X"})` | No string syntax |
| **Cirq** | Python objects | `cirq.X(q0)*cirq.Z(q1)` | No string parser |
| **NetKet** | Dense positional | `ZZII` + weights | Left-to-right |

No framework parses inline sum expressions like `"0.5*X0Z1 + X0X1"` from a single string — yaosim's parser is novel in this regard. All frameworks represent sums as collections of (string, coefficient) pairs.

For qudit operators (d > 2), no major framework has a standard string syntax. The standard operator basis for qudits is Weyl-Heisenberg (clock-shift) operators, but no tools expose a string format for them.

---

## 10. Output Design

### 10.1 Principles

- **Human-readable** (default): Compact summary, top-k entries, ASCII histograms
- **`--json`**: Structured JSON to stdout
- **`--csv`**: Tabular data to stdout
- **stderr**: Progress, timing, warnings — never mixed with data on stdout
- **`--quiet`**: Suppress banners, data only to stdout
- Float formatting: deterministic canonical format across platforms

### 10.2 Primary Task: `statevector`

**Default (compact):**
```
Circuit: 2 qubits | 2 gates | depth 2

State Vector (2/4 nonzero, threshold=0.01):
  |00>  0.7071+0.0000i  (50.00%)
  |11>  0.7071+0.0000i  (50.00%)
```

**`--verbose`:**
```
Circuit: 2 qubits | 2 gates | depth 2
Initial state: |00>

State vector (non-zero amplitudes):
  |00>  0.7071+0.0000i  (p = 0.500)
  |01>  0.0000+0.0000i  (p = 0.000)
  |10>  0.0000+0.0000i  (p = 0.000)
  |11>  0.7071+0.0000i  (p = 0.500)

Norm: 1.0000
```

### 10.3 Primary Task: `probs`

```
Circuit: 2 qubits | 2 gates | depth 2

Probability Distribution:
  |00>  0.500000  ########################################  50.00%
  |01>  0.000000                                            0.00%
  |10>  0.000000                                            0.00%
  |11>  0.500000  ########################################  50.00%

Nonzero: 2/4 basis states
```

**With `--qubits 0` (marginal):**
```
Marginal Probability (qubit 0):
  |0>  0.5000
  |1>  0.5000
```

### 10.4 Primary Task: `sample`

```
Circuit: 2 qubits | 2 gates | depth 2

Measurement Samples (1000 shots):
  |00>   503  ######################  50.3%
  |11>   497  #####################   49.7%
```

### 10.5 Primary Task: `expect`

```
Circuit: 2 qubits | 2 gates | depth 2

Expectation Value:
  <Z0Z1> = 1.000000
```

**Multi-term observable with `--verbose`:**
```
Expectation Values (per term):
  <Z0Z1> = 1.000000  (coeff: 0.5, contribution: 0.500000)
  <X0X1> = 1.000000  (coeff: 1.0, contribution: 1.000000)

Total: <H> = 1.500000
```

### 10.6 Primary Task: `overlap`

**Default (<0|U|0>):**
```bash
$ yaosim run bell.json --task overlap
```
```
Circuit: 2 qubits | 2 gates | depth 2

Overlap <0|U|0>:
  Amplitude: 0.7071+0.0000i
  |Amplitude|^2: 0.500000
```

**Arbitrary bra/ket:**
```bash
$ yaosim run bell.json --task overlap --bra "1,1" --ket "0,0"
```
```
Circuit: 2 qubits | 2 gates | depth 2

Overlap <11|U|00>:
  Amplitude: 0.7071+0.0000i
  |Amplitude|^2: 0.500000
```

**Shortcut:**
```bash
$ yaosim run bell.json --overlapwithzero
# Equivalent to: --task overlap (bra=|0>, ket=|0>)
```

### 10.7 Composable Add-ons

When add-ons are used, extra sections appear below the primary output:

```
yaosim run bell.json --task statevector --with-probs --sample 1000
```

```
Circuit: 2 qubits | 2 gates | depth 2

State Vector (2/4 nonzero, threshold=0.01):
  |00>  0.7071+0.0000i  (50.00%)
  |11>  0.7071+0.0000i  (50.00%)

Probabilities:
  |00>  0.5000
  |11>  0.5000

Measurement Samples (1000 shots):
  |00>   503  ######################  50.3%
  |11>   497  #####################   49.7%
```

### 10.8 JSON Output

```bash
yaosim run bell.json --task probs --sample 1000 --json
```

```json
{
  "circuit": {"qubits": 2, "gates": 2, "depth": 2},
  "probs": {"00": 0.5, "11": 0.5},
  "samples": {"shots": 1000, "seed": 42, "counts": {"00": 503, "11": 497}}
}
```

### 10.9 Composable Flags + Task Interaction Rules

- `--task` selects the primary computation (default: `statevector`)
- `--sample <N>`, `--with-probs`, `--obs "<expr>"` add extra outputs — they never conflict with `--task`
- `--sample <N>` as an add-on is distinct from `--task sample --shots <N>` as primary:
  - Primary `sample`: formatted prominently with histograms and stats
  - Add-on `--sample`: compact section appended below primary output

---

## 11. `yaosim easybuild` Output Showcase

```bash
$ yaosim easybuild qft --nqubits 4 | head -20
{
  "num_qubits": 4,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "Phase", "params": [1.5707963], "targets": [1], "controls": [0], "control_configs": [true]},
    ...
  ]
}

$ yaosim easybuild variational --nqubits 4 --nlayers 2 --topology ring
# Outputs variational ansatz JSON with Ry rotations + CNOT entanglers
```

---

## 12. `yaosim convert` Output Showcase

```bash
$ yaosim convert ghz.qasm --to json
{
  "num_qubits": 3,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0], "control_configs": [true]},
    {"type": "gate", "gate": "X", "targets": [2], "controls": [1], "control_configs": [true]}
  ]
}

$ yaosim convert bell.json --to qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
```

---

## 13. `yaosim show` Output Showcase

```bash
# To file
$ yaosim show bell.json --format pdf -o bell.pdf
Rendering circuit diagram...
  Format: PDF (via Typst)
  Output: bell.pdf (18.4 KB)

# To stdout (pipe to viewer)
$ yaosim show bell.json --format svg > bell.svg

# Pipe from easybuild
$ yaosim easybuild qft --nqubits 4 | yaosim show - --format svg -o qft.svg

# Direct to viewer on macOS
$ yaosim show bell.json --format pdf | open -f -a Preview
```

---

## 14. `yaosim info` Output Showcase

```bash
$ yaosim info circuit.json
Circuit: circuit.json
  Qubits     : 4
  Gates      : 10  (H: 4, Phase: 6)
  Depth      : 6
  Noise      : 0 channels
  Parameters : none

$ yaosim info noisy_bell.json
Circuit: noisy_bell.json
  Qubits     : 2
  Gates      : 2  (H: 1, X: 1 controlled)
  Depth      : 2
  Noise      : 2 channels (depolarizing: 2)
  Parameters : none

$ yaosim info circuit.json --json
{"qubits":4,"gates":{"total":10,"H":4,"Phase":6},"depth":6,"noise_channels":0,"parameters":[]}
```

---

## 15. Comprehensive CLI Showcase

### 15.1 Beginner: First Bell State

```bash
# Write a Bell state circuit
$ cat bell.json
{
  "num_qubits": 2,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0], "control_configs": [true]}
  ]
}

# Simulate — default task is statevector
$ yaosim run bell.json
Circuit: 2 qubits | 2 gates | depth 2

State Vector (2/4 nonzero, threshold=0.01):
  |00>  0.7071+0.0000i  (50.00%)
  |11>  0.7071+0.0000i  (50.00%)

# See probabilities
$ yaosim run bell.json --task probs
Probability Distribution:
  |00>  0.500000  ########################################  50.00%
  |11>  0.500000  ########################################  50.00%

# Measure 1000 times
$ yaosim run bell.json --task sample --shots 1000
Measurement Samples (1000 shots):
  |00>   503  ######################  50.3%
  |11>   497  #####################   49.7%

# Generate PDF diagram
$ yaosim show bell.json --format pdf -o bell.pdf

# Try built-in QFT
$ yaosim run --circuit qft --nqubits 4 --verbose
Circuit: 4 qubits | 10 gates | depth 6  (QFT-4)
Initial state: |0000>

State vector (non-zero amplitudes):
  |0000>  0.2500+0.0000i  (p = 0.0625)
  |0001>  0.2500+0.0000i  (p = 0.0625)
  |0010>  0.2500+0.0000i  (p = 0.0625)
  ... (16 amplitudes, all equal — uniform superposition)

# Inspect the QFT circuit
$ yaosim easybuild qft --nqubits 4 | yaosim info -
Circuit: stdin
  Qubits: 4 | Gates: 10 (H: 4, Phase: 6) | Depth: 6
```

### 15.2 Researcher: VQE Parameter Sweep

```bash
# Generate variational template
$ yaosim easybuild variational --nqubits 4 --nlayers 2 --topology ring -o vqe.json

# Inspect
$ yaosim info vqe.json
Circuit: vqe.json
  Qubits     : 4
  Gates      : 24  (Ry: 16, X: 8 controlled)
  Depth      : 10
  Parameters : {{theta_0}} .. {{theta_7}}

# Sweep theta, output CSV
$ yaosim run vqe.json \
    --sweep "theta_0=theta_1=theta_2=theta_3=theta_4=theta_5=theta_6=theta_7=0.0:0.1:3.14" \
    --task expect --obs "1.0 Z0Z1 + 1.0 Z1Z2 + 1.0 Z2Z3 + 1.0 Z3Z0" \
    --csv
theta,<H>
0.00,4.000000
0.10,3.921569
...
1.40,-1.713248
...
3.14,2.560612

# Render the circuit for a paper
$ yaosim show vqe.json --format svg -o vqe_circuit.svg
```

### 15.3 Engineer: QASM Interop

```bash
# Import from Qiskit-exported QASM
$ cat ghz.qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];

# Inspect
$ yaosim info ghz.qasm
Circuit: ghz.qasm
  Qubits: 3 | Gates: 3 (h: 1, cx: 2) | Format: OpenQASM 2.0

# Simulate directly from QASM
$ yaosim run ghz.qasm --task probs
Probability Distribution:
  |000>  0.500000  ########################################  50.00%
  |111>  0.500000  ########################################  50.00%

# Convert to JSON
$ yaosim convert ghz.qasm --to json -o ghz.json

# 10000 shots, JSON output
$ yaosim run ghz.json --task sample --shots 10000 --json -o results.json

# Round-trip back to QASM
$ yaosim convert ghz.json --to qasm -o ghz_roundtrip.qasm
```

### 15.4 Researcher: Noisy Simulation

```bash
# Circuit with noise channels
$ cat noisy_bell.json
{
  "num_qubits": 2,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0], "control_configs": [true]},
    {"type": "channel", "noise": "depolarizing", "params": {"n": 1, "p": 0.05}, "locs": [0]},
    {"type": "channel", "noise": "depolarizing", "params": {"n": 1, "p": 0.05}, "locs": [1]}
  ]
}

# Auto-detects noise -> dm backend
$ yaosim run noisy_bell.json --task probs --obs "Z0Z1"
[backend] noise channels detected -- switching to dm

Probability Distribution:
  |00>  0.4763
  |01>  0.0237
  |10>  0.0237
  |11>  0.4763

Expectation Value:
  <Z0Z1> = 0.9050

# Compare with clean version
$ yaosim run bell.json --task probs --obs "Z0Z1"
[backend] using statevec

Probability Distribution:
  |00>  0.5000
  |11>  0.5000

Expectation Value:
  <Z0Z1> = 1.0000
```

### 15.5 Automation: Pipeline and Scripting

```bash
# Generate + simulate in one pipe
$ yaosim easybuild qft --nqubits 4 | yaosim run - --task probs --json --quiet

# Size sweep
$ for n in $(seq 2 8); do
    yaosim easybuild qft --nqubits $n \
      | yaosim run - --task probs --json --quiet 2>/dev/null \
      | jq "{qubits: $n, nonzero: ([.probs | to_entries[] | select(.value > 0.001)] | length)}"
  done

# Batch QASM -> JSON conversion
$ for f in circuits/*.qasm; do
    yaosim convert "$f" --to json -o "json/$(basename ${f%.qasm}).json"
  done

# Regression test
$ diff <(jq --sort-keys . baseline.json) \
       <(yaosim run bell.json --task probs --json --quiet | jq --sort-keys .)

# Expectation value comparison
$ yaosim run bell.json --task expect --obs "Z0Z1" --json --quiet
{"expect": {"Z0Z1": 1.0}}
```

### 15.6 Observable Syntax Examples

```bash
# Site-indexed (OpenFermion-style)
$ yaosim run bell.json --task expect --obs "Z0Z1"
$ yaosim run bell.json --task expect --obs "0.5 Z0Z1 + 0.3 X0X1"
$ yaosim run bell.json --task expect --obs "Z0 + Z1 + Z2"

# Dense positional (Qiskit-style, left-to-right = qubit 0)
$ yaosim run bell.json --task expect --obs "ZZ"
$ yaosim run bell.json --task expect --obs "0.5 ZZ + 0.3 XX"

# From JSON file (for complex Hamiltonians)
$ yaosim run circuit.json --task expect --obs hamiltonian.json

# Composable: statevector + expectation
$ yaosim run bell.json --obs "Z0Z1"
# Output: state vector (primary) + expectation value (add-on)
```

### 15.7 Initial State Specification

```bash
# Default: |0...0>
$ yaosim run bell.json

# Product state
$ yaosim run circuit.json --init-state "1,0,1"

# From file (arbitrary state vector)
$ yaosim run circuit.json --init-state state.json

# Shortcut: apply circuit to specific state
$ yaosim run circuit.json --applyto "1,0,0"
```

### 15.8 All Tasks with JSON Output

```bash
# statevector
$ yaosim run bell.json --task statevector --json
{"statevector": {"00": {"re": 0.7071, "im": 0.0}, "11": {"re": 0.7071, "im": 0.0}}}

# probs
$ yaosim run bell.json --task probs --json
{"probs": {"00": 0.5, "11": 0.5}}

# probs marginal
$ yaosim run bell.json --task probs --qubits 0 --json
{"marginal_probs": {"qubits": [0], "0": 0.5, "1": 0.5}}

# sample
$ yaosim run bell.json --task sample --shots 1000 --seed 42 --json
{"samples": {"shots": 1000, "seed": 42, "counts": {"00": 503, "11": 497}}}

# expect
$ yaosim run bell.json --task expect --obs "Z0Z1" --json
{"expect": {"Z0Z1": 1.0}}

# overlap (default: <0|U|0>)
$ yaosim run bell.json --task overlap --json
{"bra": "00", "ket": "00", "overlap": {"re": 0.7071, "im": 0.0}, "abs_squared": 0.5}

# overlap with explicit states
$ yaosim run bell.json --task overlap --bra "1,1" --ket "0,0" --json
{"bra": "11", "ket": "00", "overlap": {"re": 0.7071, "im": 0.0}, "abs_squared": 0.5}
```

---

## 16. Error Handling

Standard Rust error handling with `Result<T, E>`:
- `thiserror` in library crates (`yaosim-core`, `yao-rs`)
- `anyhow` in the binary crate (`yaosim`) only
- Do not leak `ndarray` or `tch` error types into public APIs

**Error message format:**
```
error: <what went wrong>
  hint: <how to fix>
```

**Key scenarios:**
- Invalid JSON/QASM: parse error with source line number and offending token
- Feature not compiled: `error: feature 'typst' not compiled; rebuild with --features typst`
- Dimension mismatch: report expected vs actual shape
- Invalid Pauli expression: show offending substring + valid syntax example
- Pauli on qudit site: `error: Pauli operators require d=2, site 0 has d=3`
- Unresolved template params: list all unresolved `{{param}}` names
- Large circuit warning: warn at >= 24 qubits with memory estimate

---

## 17. Testing Strategy

- **CLI tests**: `assert_cmd` + `predicates` for subprocess-level testing. `insta` for snapshot testing of human-readable output.
- **yaosim-core tests**: Own test data in `yaosim-core/tests/data/`. QASM test corpus curated from OpenQASM repo.
- **Stochastic tests**: `--seed` flag for deterministic `sample` output in tests.
- **Feature-gated tests**: `#[cfg(feature = "typst")]` / `#[cfg(feature = "torch")]`.
- **CI**: `make check-yaosim` target added to `make check-all`.

---

## 18. Deliverables

1. `yaosim-core/` crate — QASM 2.0 parser, Pauli expression parser, output formatting, circuit analysis
2. `yaosim/` crate — CLI binary with all subcommands
3. Comprehensive showcase documentation (section 15 of this spec, expanded into mdBook under `docs/src/cli/`)
4. Tests for all subcommands, parsers, and output formats

**Post-merge follow-up:** File a GitHub issue for parameterized circuit support in yao-rs library (to replace `{{template}}` mechanism).
