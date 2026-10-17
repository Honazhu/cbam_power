# Montenegro–Italy Electricity Trade Around the EU CBAM

*An open research workflow integrating electricity market data, geospatial weather controls, RDiT and synthetic counterfactual analysis.*

**Release candidate:** `v0.1.0-rc1` (Python package `0.1.0rc1`). **FINAL RELEASE CODE NUMERICAL REPLICATION: VERIFIED** for the authorized V3/V3.2 workflow in the recorded environment; see the scope below.

**Does CBAM reduce carbon-intensive electricity imports, and reduce their responsiveness to price spreads?** This coursework project studies the Montenegro–Italy corridor around **1 January 2026**, alongside four other Italy-bound corridors. Electricity is a covered CBAM sector, and that date marks the start of the definitive regime. [European Commission](https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/cbam-definitive-regime_en)

The research follows the existing project presentation: construct hourly positive net exports; estimate Basic and calendar-FE RDiT; compare Montenegro with a fixed Greece/Austria synthetic benchmark; and add corridor-specific bilateral ERA5 controls. The two hypotheses concern a local fall in imports and lower responsiveness to destination–origin price spreads. The observed outcome is a commercial-flow proxy, not a direct measure of embedded emissions.

![Three research branches: actual-corridor RDiT, the fixed synthetic benchmark and gap, and ERA5-based weather-adjusted RDiT](docs/figures/workflow.svg)

The diagram preserves the presentation's research logic. Commercial-flow files are frozen **Energy-Charts mirrors with ENTSO-E upstream provenance**. The presentation's “relative gap” means **actual minus synthetic in MW**, not a percentage. Full definitions and terminology reconciliation are in the [methodology](docs/methodology.md). This repository makes that existing design reproducible; public examples use artificial data, while empirical inputs, frozen numerical baselines and results remain private.

## Run the public example

Use **Python 3.12** and **uv**. The validated reference environment used Python **3.12.14** on macOS arm64. Run these commands from the repository root:

```bash
uv sync --locked --no-editable --extra weather --extra learning --group test
make test
make demo RUN_ROOT=../cbam-power-demo-runs RUN_ID=demo_01
```

`make test` runs public/artificial tests. `make demo` validates invented inputs, aggregates market and weather observations, assembles a panel, applies fixed artificial donor weights, estimates an RDiT with HAC inference, checks the known artificial effect and renders a report. It writes to the separate run directory. Use a fresh `RUN_ID` for a new run.

Every demo result is labelled **SYNTHETIC DATA — NOT EMPIRICAL RESEARCH RESULTS**. The example requires no private research data or provider credentials. Installation resolves software dependencies; computation itself does not download research data. The Makefile rebuilds a regular wheel so source edits cannot silently leave an old installed package in use.

## Explore the research workflow

| Notebook | What it explains |
|---|---|
| [01 · Data and weather](notebooks/01_data_and_weather.ipynb) | Flow signs, hourly aggregation, UTC/Rome time and spatial weather weights |
| [02 · RDiT and HAC](notebooks/02_rdit_and_hac.ipynb) | Local trends, fixed effects, triangular weights and elapsed-hour inference |
| [03 · Synthetic control](notebooks/03_synthetic_control.ipynb) | Fixed donor weights, completeness, gap signs and support limitations |
| [04 · Robustness and validation](notebooks/04_robustness_and_validation.ipynb) | Polynomial interactions, shape checks and scientific regression tests |

The notebooks call package functions and retain reviewed artificial figures, tables and numerical checkpoints for direct GitHub reading. Each notebook was executed twice from a fresh kernel; the publication gate accepts registered PNG and text outputs, and rejects errors, private metadata and active HTML. To independently rerun copies in a separate output directory:

```bash
uv sync --locked --no-editable --extra weather --extra learning --group test
make notebooks RUN_ROOT=../cbam-power-demo-runs RUN_ID=learning_01
```

Read the [methodology](docs/methodology.md) for equations, estimands, data lineage and limitations. [Data access and release boundaries](docs/data_access.md) explains provider permissions and authorized local replication.

Open the [offline research website](docs/index.html) for the same maintained methodology with rendered equations, numbered teaching figures, navigation and the prototype API reference. Rebuild with `make site`; `make site-check` detects stale generated content. The website uses native MathML and local assets, without JavaScript or a CDN. CI prepares a Pages-ready directory from the exact registered source and artificial-material allowlist, preserving the repository layout and the `docs/index.html` entry (`/docs/` when hosted). It does not upload or deploy that directory; GitHub Pages is not published.

## What is implemented and verified?

The package retains the **V3 formal baseline**, the **V3.1 presentation layer** and the **V3.2 polynomial robustness extension** as distinct research objects. It includes the complete model schedule, all 16 frozen Montenegro synthetic versions, full covariance matrices and explicit unestimated statuses.

The completed private **PR5/G5 validation** compared independently rebuilt outputs with frozen references in the locked macOS arm64 / Python 3.12.14 environment: **1,892 estimation records**, **77,574 coefficient rows**, **3,612,010 covariance elements**, and **90 polynomial models**. Sample identities, missing sets and model statuses were also checked. These are verification coverage counts, not empirical findings. Passing public tests or running the demo does not reproduce those private comparisons. Other platforms and future package revisions require their own validation.

**FINAL RELEASE CODE NUMERICAL REPLICATION: VERIFIED.** On 17 October 2026, `v0.1.0-rc1` (package `0.1.0rc1`) completed one new, independent authorized run from inputs through weather, panel and fixed-weight synthetic construction, the complete V3 estimation and verification, and V3.2 estimation and verification. All eight stages passed in locked macOS arm64 / Python 3.12.14. The complete Python source identity is `34ea74ef89c32e8810459760c5415e77228276ddf2498e6974f4d4b2173e0945`. Inputs and benchmarks were read-only; historical project, prior-run and external-network access were denied. The private receipt binds the exact source, configuration, lockfile, environment and comparisons. This certifies the formal Montenegro workflow on those bytes and that environment; the generic methods prototype remains supported by artificial tests only. It does not certify other countries, future revisions or platforms. GitHub CI and cross-platform numerical replication have not been executed.

| Mode | Inputs and purpose | Output boundary |
|---|---|---|
| Public learning | Invented observations; inspect algorithms and run software tests | Clearly labelled artificial outputs in a separate run directory |
| Authorized replication | Exact private input versions and read-only frozen references | Private runs with input, code, environment and validation receipts |

Full replication uses `CBAM_INPUT_ROOT`, a separate `CBAM_OUTPUT_ROOT`, and the supplementary verification references described in [data access](docs/data_access.md). `make reproduce` intentionally performs the full calculation; it fails on missing inputs and never falls back to the demo. Reporting reads saved results and numeric sidecars without fitting models.

## Interpretation and limitations

The local cutoff jump, same-season level difference, price-spread slope change and actual-minus-synthetic gap answer different questions. RDiT requires care with time trends, serial dependence, anticipation and coincident shocks. The 2025 placebo and bandwidth/functional-form checks are part of that assessment. [Hausman and Rapson](https://www.nber.org/papers/w23602)

The frozen Montenegro synthetic benchmark has weak prefit and support and remains descriptive. Inference conditions on its frozen weights; there is no bootstrap of weight uncertainty. Local bilateral weather is available for the main window, while incomplete annual weather and other blocked specifications retain their recorded status. A successful software replication does not strengthen these identification assumptions.

## Code, data and citation

Start with [`src/cbam_power/`](src/cbam_power/), the frozen design in [`config/`](config/), the input and validation metadata in [`manifests/`](manifests/), and [`tests/`](tests/). Official source references and the legacy-to-package mapping are collected in the [methodology](docs/methodology.md).

**Author:** Hona Zhu. **Contributors, in contribution order:** Mohammad Amir, Jasper Wu and John Sirin. Developed as a coursework demonstration for **Natural Resource and Energy Economics, Stockholm School of Economics**. Project members developed the research and engineering code with AI assistance.

The rights holder has confirmed MIT release of the original code and teaching material; see [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff). This permission does not cover third-party data, inherited source material or restricted derivatives. Those remain subject to the separate [data-access and publication boundary](docs/data_access.md); security checks do not authorize redistribution of referenced payloads.
