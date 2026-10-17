# Data access and release boundaries

This repository candidate separates runnable research code from authorized inputs
and research outputs. Local access, successful replication and a provider's open
data policy are different from approval to publish a particular file. `UNVERIFIED`
never means `APPROVED_PUBLIC`. Hona Zhu has confirmed the authority to license original project software and
original teaching materials under [MIT](../LICENSE). This grant does not cover
third-party data, fonts, maps or documents.

## Public demo and authorized replication

`make demo` uses artificial inputs and a reduced specification. Its output must
carry **SYNTHETIC DATA — NOT EMPIRICAL RESEARCH RESULTS**. It does not use the
private empirical inputs or calibrate artificial values to frozen results.
`make test` runs public/artificial tests, not the full authorized test suite.
Demo execution alone does not approve publication of the repository's code.

Full replication requires the exact frozen inputs supplied by a researcher who
has the necessary provider permissions. Keep three locations separate:

| Location | Content | Access |
|---|---|---|
| I: `CBAM_INPUT_ROOT` | `closure/` raw and frozen upstream inputs | Read-only |
| B: `CBAM_INPUT_ROOT/baselines/` | Frozen panels, results, weights and verification references | Read-only |
| A: `CBAM_OUTPUT_ROOT/<run_id>/` | Newly computed outputs, logs and receipts | Private; independent of I/B and the candidate repository |

A must not overlap I, B or the repository, including ancestor/descendant and
symlink relationships. Keep credentials, source payloads, private run receipts,
baselines, empirical tables/figures/PDFs, and real-data or unreviewed executed
notebook outputs outside the public candidate. The reviewed artificial notebook
outputs described below are explicitly permitted. A default-deny `.gitignore` is an additional guard, not release approval.

```bash
export CBAM_INPUT_ROOT="/path/to/authorized/inputs"
export CBAM_OUTPUT_ROOT="/path/to/separate/runs"
export CBAM_REFERENCE_ROOT="/path/to/private/verification-references"

uv sync --locked --no-editable --extra weather --group test
uv run --locked --no-editable --extra weather --group test \
  cbam-power validate-inputs --mode authorized --run-id input_check
```

Use a regular wheel installation; after source edits, reinstall with
`uv sync --locked --no-editable --extra weather --group test --reinstall-package cbam-power`.
For an intentionally requested new full computation, `make reproduce` also needs
an explicit private `RUN_ROOT` and a fresh `RUN_ID`. A licensing review does not
require repeating that computation.

`manifests/inputs.toml` is the runtime index. Logical IDs resolve exact versions,
sizes and SHA-256 identities. Legacy paths are provenance, never fallback paths.
The resolver does not download, repair, choose a newer version, fill observations
or change expected hashes. Missing or changed required inputs fail with
`BLOCKED_INPUT_MISSING` or `BLOCKED_INPUT_HASH`. F assets support computation;
K assets support verification; P assets are optional presentation references;
H assets record history. K/P/H values must not enter estimation as observations
or expected coefficients.

Six existing polynomial sidecars are additionally needed for full verification:
`polynomial_daily_bins_p1`–`p3` and `polynomial_plot_curves_p1`–`p3`. Store unchanged
authorized copies in `CBAM_REFERENCE_ROOT` as `<asset_id>.csv`; the frozen manifest
pins their identities. They are comparison references only. Optional source maps,
reports and fonts are not automatically materialized or published.

## Provider permissions and inherited provenance

The following distinguishes evidence checked on **2026-10-16** from the exact
terms governing the frozen acquisition. Current terms do not silently replace
historical agreements. The release registry records decisions separately from
the immutable scientific input manifest.

| Source | Evidence and public boundary |
|---|---|
| Energy-Charts Italy-reporter flows | The frozen OpenAPI and [current API](https://api.energy-charts.info/) state a CC BY 4.0 default with exceptions. Credit Energy-Charts/Fraunhofer ISE and identify upstream ENTSO-E provenance and transformations. Exact upstream item eligibility remains `UNVERIFIED`; no provider-level finding automatically approves every artifact. |
| Restricted Energy-Charts prices | **Seven frozen payloads** explicitly restrict external use of both raw and derived data: ME 2024/2025/2026, IT-Centre-South 2024/2025/2026, and GR 2024. Their exception overrides the API default. Keep them and affected derivatives private unless specific separate permission is documented. |
| SMARD and qualifying mirrors | [SMARD market-data terms](https://www.smard.de/en/datennutzung) allow attributed CC BY 4.0 reuse; retain `Bundesnetzagentur \| SMARD.de`, the licence and modification notice. This does not authorize other exchange prices or mixed-source panels. |
| GME and HEnEx | [GME terms](https://www.mercatoelettrico.org/en-us/Home/TermsOfUse) and [HEnEx terms](https://www.enexgroup.gr/web/guest/terms-of-use) do not establish a general public-repository grant. Historical accepted terms, additional permissions and some derived-publication scopes remain `UNVERIFIED`. Attribution alone is insufficient. |
| Eurostat generation | Six frozen `nrg_cb_pem` JSON files identify ESTAT/Eurostat. [Current statistical reuse policy](https://ec.europa.eu/eurostat/en/help/copyright-notice) permits reuse with source acknowledgement and applicable notices. Exact acquisition-time terms were not archived. This permission does not extend to mixed-source weights, panels or price-based results. |
| Copernicus ERA5 | The [current ERA5 single-level catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) links CC BY 4.0. Cite the dataset/DOI, producer and acquisition source, and identify modifications. The exact accepted September 2026 licence revision is `UNVERIFIED`; the 50 NetCDF headers do not contain it. |
| GISCO Countries/NUTS | [Administrative](https://ec.europa.eu/eurostat/web/gisco/geodata/administrative-units) and [statistical boundary](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units) terms specify noncommercial use and the notice `© EuroGeographics for the administrative boundaries` in map legends and introductory publication pages. Do not apply Eurostat's statistical policy to geography. Exact frozen-vintage redistribution and derived-weight/aggregate permissions remain `UNVERIFIED`. |

V1 AT/CH/ME observations remain inherited frozen inputs. Their actual Italian
destination prices use GME when official observations exist; the IT-CSUD mirror
served validation and is not an alternative licence for those GME values.
The package does not claim to rebuild the entire earlier provider-to-V1 chain.
Source and variable lineage must survive aggregation: restricting publication
does not authorize changing the scientific source, model or sample.

Regional weather combines ERA5 with frozen GISCO-derived masks and area weights.
It therefore needs a separate artifact-level decision; ERA5 openness alone does
not settle the geometry-derived component. The formal calculation retains eight
geographies; a historical FR geometry does not add FR to formal analysis.
Complete third-party HTML, PDF source documents, embedded fonts and historical
map/side-task outputs also require separate review.

## Release checks

Release decisions follow file → variable → artifact lineage, including all
transitive dependencies. A review must identify the rights holder, evidence,
scope, destination, required notices and exact content identity. Neither
aggregation nor publication of only coefficients or a screenshot creates an
exemption. A documented artifact-specific exemption must state its precise scope.

The release API uses `RightsRegistry.authorize` for decisions and `stage_public`
for controlled staging. The command equivalents are:

```bash
cbam-power release-check --artifact <registered_id> \
  --registry <reviewed_rights.toml> --context public_repository
cbam-power export-public --registry <reviewed_rights.toml> \
  --source-root <reviewed_source_root> --staging <new_public_staging_dir> \
  --selection <private_selection.json>
```

The current content allowlist supports artificial text, tables, SVG and HTML
reports. Real data-bearing formats require an additional explicit content-policy
review, as well as adequate licences and artifact approvals. This engineering
restriction is not a legal conclusion that every regression statistic is prohibited.

The selection maps relative source paths to registered node IDs. It is not a
rights grant. Unknown, restricted or unverified dependencies must block export.
Real `report --mode authorized --audience public` requires explicit approval of
the saved result and each rendered artifact; otherwise it fails. Local reports
stay private and read saved results without estimation. Do not use a demo label
to export real observations or results.

The recorded PR5 G5 acceptance concerns numerical replication in the tested
locked environment. Its code/input identities, frozen contracts, numerical
tolerances and private evidence remain protected. G5 is not a publication
permission. The confirmed MIT software grant remains separate from empirical-artifact
approvals. Dependency licences and notices must also be preserved when their
covered software is redistributed.


## Secure first-release candidate

The first release surface is complete project source, tests, scientific design
configuration, provenance/schema metadata, and artificial demo generators.
All real raw/processed data, frozen numerical baseline payloads, synthetic
series, empirical results, unapproved figures and real-data notebook outputs
remain private, including datasets whose provider permits attributed reuse.
The three manifest files contain identifiers, hashes, schemas, request metadata
and validation rules; they do not bundle the files they reference. Full input
identities stay unchanged, so no public/private manifest override is needed.

`prepare_candidate` creates a new **security-review candidate** from the exact
source allowlist after scanning every byte. This is not publication approval:
`stage_public` still requires the separate ownership and artifact permissions.
Do not add `.env`, `.cdsapirc`, connection credentials, caches, environments,
logs or raw payloads. Scanner errors contain risk codes, not matched secret
values. A clean scan cannot replace review of future files or Git history.

For an isolated environment outside the source tree, set
`UV_PROJECT_ENVIRONMENT` to that separate environment, use a clean process
environment and an offline dependency cache, and set an explicit private
`RUN_ROOT` when running `make demo`. `make test` selects public/artificial tests;
authorized tests still require the exact private inputs and must not silently
fall back to demo. Keep installed module hashes equal to the reviewed source.

Project members developed the research and engineering code with AI assistance.
AI assistance is not itself a reason to seek an OpenAI open-source grant.
Hona Zhu confirmed the required authority for the MIT grant, including the
absence of outstanding contributor or institutional permissions. Third-party
source obligations remain separate. Security clearance alone is not authority
to publish, and this preparation does not perform a GitHub release.

## Teaching materials and final release preparation

Original project software and original teaching materials are licensed under
[MIT](../LICENSE), copyright 2026 Hona Zhu. Preserve the copyright and permission
notice in copies or substantial portions. The grant does not relicense third-party
data or dependencies, and does not approve empirical results for publication.
Author and contributor roles are recorded in the [README](../README.md); the
[CFF citation](../CITATION.cff) names Hona Zhu as author. This is a Natural Resource
and Energy Economics coursework demo at Stockholm School of Economics, with no
claim of institutional endorsement.

The four notebooks use invented observations, fixed random seeds and existing
package functions. Their reviewed artificial PNG figures, text tables and numerical
checkpoints are saved for direct reading. Byte-pinned sources and output identities
are registered together with artificial provenance. `make notebooks` reruns every
notebook in two independent fresh kernels, compares outputs and checkpoints, and
writes new copies only to the separate run root. Stored outputs are never accepted
as a substitute for execution. The publishing policy rejects errors/tracebacks,
active HTML, credential metadata and temporary execution guards. The two statistical illustrations are labelled
synthetic; the three method diagrams use original schematic geometry, not provider
boundaries. Their content identities and derivation are registered separately.

`make release-check` reviews the exact source and artificial-figure allowlist;
`PACKAGE_ROOT=<external distribution directory>` additionally checks actual wheel
and source distributions. This safety check is not legal publication approval.
The proposed GitHub workflow exercises public tests, the artificial pipeline,
notebooks and packaging. It does not require credentials or private inputs and
does not certify a new empirical replication or a new operating system.

The offline research website is built by `make site` from the single maintained
`methodology.md`, reviewed figures and API source signatures. `make site-check`
rejects stale HTML. It loads no remote assets or scripts. The general methods API
is an independently tested artificial-data prototype, outside the historical G5
and PR7 full numerical certifications; the frozen replication engine is unchanged.

CI also prepares a static Pages-ready directory through `stage_public`, using the
exact registered source/artificial allowlist and retaining the MIT notice. The
repository-relative layout is preserved, with entry `docs/index.html`; preparation
verifies copied identities and HTML freshness. Its receipt stays outside that
directory. No full-workspace copy, artifact upload or Pages deployment is performed,
and this step requests no additional write or deployment permission.
