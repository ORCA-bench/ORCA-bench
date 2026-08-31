# Repo setup

Non-default settings this leaderboard needs. One-time, mostly.

This `leaderboard/` folder is a self-contained project (own `pyproject.toml`,
benchmark-neutral `leaderboard` package) vendored from
[harbor-index](https://github.com/harbor-framework/harbor-index) without
touching ORCA-bench's root project. The per-benchmark constants live in exactly
three places:

- `src/leaderboard/core/hub.py` — `DATASET`, `DATASET_REF`
- `src/leaderboard/ci/static_analysis.py` — `MIN_TRIALS_PER_TASK` (the expected
  task count is derived from the dataset registry, not hardcoded)
- `src/leaderboard/ci/submit.py` — `LEADERBOARD_PACKAGE`, `LEADERBOARD_NAME`

Plus `src/leaderboard/display_names.json`, which seeds the display-name map.

The workflows live in `.github/workflows/leaderboard-*.yml` rather than in here
because GitHub only executes workflow files directly in that directory.

## What this port leaves out

harbor-index also ships an LLM **trajectory judge** (`/judge` and `/apply`
comment commands, `ci/judge.py`, `ci/gen_analysis_tasks.py`, and the
`leaderboard-judge.yml` / `leaderboard-apply.yml` workflows). The *trajectory*
judge is **not** ported: it runs judge agents on Modal against per-task agent
images on Docker Hub, which ORCA-bench's privileged docker-in-docker otel-demo
tasks don't have. So no `ANTHROPIC_API_KEY` / Modal secrets are needed, and the
`disqualified_trials` / `credited_trials` override lists are maintained by hand
on the bot PR instead of by `/apply`. The metric still honors them.

ORCA-bench does ship its own `leaderboard-judge.yml`, but it answers a different
question. It runs `run_llm_judge.py` on the runner — a plain API call per trial,
no sandbox — to score the held-out split, whose tasks ship without
`expected.json` or `rubrics/` and so cannot be scored in-container (see
`build_harbor_tasks.py`). Same `/judge` trigger and the same write-access gate;
different work.

## Get your `HARBOR_API_KEY`

`harbor auth login` (GitHub OAuth in the browser) mints a long-lived personal
`sk-harbor-…` key and stores it in `~/.harbor/credentials.json`:

```bash
uv tool install harbor            # if you don't have the CLI yet
harbor auth login
jq -r .api_key ~/.harbor/credentials.json   # the raw key, for CI secrets etc.
```

harbor itself reads the stored key automatically; a `HARBOR_API_KEY` env var
takes precedence when set — that's how CI supplies it.

## The dataset

Published as `orca-bench/orca-bench` (755 tasks), pinned in
[`core/hub.py`](src/leaderboard/core/hub.py) as
`sha256:2add497ac2f93468dba1b83977c295a784c89031754e6a35520195345ad62ada`.

**The dataset is public**, so anyone can fetch it and run the benchmark. CI
still passes `HARBOR_API_KEY` because the submission flow writes (cloning
trials, creating rows), not because reading needs it.

If it is ever made private again, the failure mode is worth knowing: an
*unauthenticated* caller gets

```
ValueError: Tag 'latest' not found for dataset 'orca-bench/orca-bench'
```

which reads like the dataset doesn't exist. It does — you are just not logged
in. Check with `harbor auth status` before concluding anything from that error.
Note also that a credentials file written by an older harbor is rejected by
0.20 ("Found credentials from an older Harbor version"), which silently
downgrades you to anonymous.

## The leaderboard on the hub

**Created** — `orca-bench/orca-bench` → leaderboard `orca-bench`, title
"ORCA-bench", **visibility `public`** (matching the dataset), id
`dcb96003-d991-4f98-97b6-3813c35daa81`. This matches `LEADERBOARD_PACKAGE` /
`LEADERBOARD_NAME` in [`ci/submit.py`](src/leaderboard/ci/submit.py).

The board on the retired uppercase package (`orca-bench/ORCA-bench`, id
`1b72818f-bd2e-4051-a3d1-634fe44808d7`) is superseded and should be deleted; it
holds no rows.

Visibility must not run ahead of the dataset's: a public leaderboard over a
private dataset shows scores nobody can reproduce or submit against. Both are
public now. A private board is hidden rather than refused — an anonymous read
returns `404 leaderboard not found`, not a permission error — so test visibility
by reading it with no credentials rather than by trusting the field.

### The package slug must stay lowercase

Every endpoint validates a `package` selector against a lowercase-only pattern —

```
/^[a-z0-9][a-z0-9_-]*\/[a-z0-9][a-z0-9_.-]*$/
```

The dataset used to be published as `orca-bench/ORCA-bench`, whose uppercase
fails that pattern, so nothing that talked to the API could select it by slug.
That forced a `package_id` UUID indirection in two places — `harbor hub
leaderboard create` and `leaderboard-row-create` in `ci/submit.py`. Both are
gone now that the package is `orca-bench/orca-bench`: `ci/submit.py` sends
`package` + `name` directly, and `tests/test_submit.py` guards the slug against
regressing to something the hub would reject.

The definition is checked in as [`leaderboard.json`](leaderboard.json) and can
be passed to `create` / `update` as-is:

```bash
uv run harbor hub leaderboard create --config leaderboard.json --json
```

The `metadata_schema` / `metrics_schema` are the contract merged submissions
must satisfy. `metadata` = the display fields, plus the `pr` markdown link cell
(`[#N](…)` → the promoted bot PR) that CI stamps in at promotion.

`metrics` is three published metrics, all over **incident** tasks only — the
panels of `plot_rca_and_hallucination.py` — plus the whole-submission resource
totals (`total_tokens` = input + output, `total_cost_usd` — optional, omitted
when the trials don't report the underlying telemetry):

| Key | Column | Subset |
| --- | --- | --- |
| `rca_accuracy_incident_medium` | RCA Accuracy (Medium) ↑ | incident ∩ difficulty=medium |
| `rca_accuracy_incident_hard` | RCA Accuracy (Hard) ↑ | incident ∩ difficulty=hard |
| `hallucinate_any_incident` | Hallucination Rate ↓ | all incident tasks |

Each is stored **three ways**, because a leaderboard cell renders exactly one
field and the cells show `mean ± se`:

| Field | Type | Purpose |
| --- | --- | --- |
| `<key>` | number | the mean; ranks the board, machine-readable |
| `<key>_stderr` | number | the standard error, machine-readable |
| `<key>_display` | string | `"26.61 ± 3.00"` — the only one with a column |

`rank_by` uses the numeric mean (`rca_accuracy_incident_medium`), never a
display field: the Hub compares strings lexicographically, so ranking on
`"9.5"` vs `"26.6"` would invert the board.

Task labels come from each task's `task.toml` `[metadata]` at the pinned refs
(see `core/task_groups.py`). A task is **control** iff its `events` list is
empty, and the difficulty tiers come from the `difficulty` field — *not*
`granularity`, which holds a second ladder (`easy`/`hard`/`universal`) for the
same tiers. Control tasks carry a difficulty too, which is why the tier metrics
intersect with "not control" rather than partitioning on difficulty alone. No
metric covers control tasks: they have no root cause, so `rca_accuracy` is 0
there by construction and `hallucinate_any` is not emitted at all.

> **Push schema changes to the live hub *before* a row that uses them merges.**
> `metrics_schema` sets `additionalProperties: false`, so `leaderboard-row-create`
> rejects a row carrying an unknown key — and that happens at merge time, after
> the PR is already merged. `tests/test_metrics_schema.py` keeps the checked-in
> schema and the computed metrics in step, but it cannot see the live hub:
> run `harbor hub leaderboard update --config leaderboard.json` first.

Verify it, or export the live definition if you need to amend it:

```bash
uv run harbor hub leaderboard list
uv run harbor hub leaderboard export orca-bench/orca-bench/orca-bench
uv run harbor hub leaderboard update --config leaderboard.json
```

The account you are logged in as must be allowed to manage the `orca-bench`
org's leaderboards.

The token / cost columns are totals over every trial in the submission
(disqualified trials included — they still consumed resources). The `pr` cell
holds a markdown link string; the `markdown` column type tells the renderer to
render it as a link.

## Re-pinning `DATASET_REF`

`src/leaderboard/core/hub.py` pins the exact dataset version submissions must
run. CI rejects any job whose dataset ref doesn't match, so re-pin it in
lockstep with every republish:

```bash
uv run python -c "import asyncio; \
  from harbor.registry.client.package import PackageDatasetClient; \
  print(asyncio.run(PackageDatasetClient().get_dataset_metadata('orca-bench/orca-bench@latest')).version)"
```

(The field is `.version`, not `.ref` — it holds the `sha256:` digest. There is
no `harbor hub dataset` subcommand.)

## Secrets

`Settings → Secrets and variables → Actions`, or `gh secret set <NAME> --repo <owner>/<repo>`:

| Secret | Used by | For |
| --- | --- | --- |
| `HARBOR_API_KEY` | check, promote, merge, close, judge | hub/registry access, trial cloning, leaderboard submit |
| `OPENAI_API_KEY` | judge | the LLM judge itself |
| `OPENAI_BASE_URL` | judge | optional; non-OpenAI inference endpoint |

`OPENAI_API_KEY` is why `/judge` is gated to write-access commenters: `promote`
runs on any fork PR that passes static analysis, so an automatic trigger would
hand contributor-controlled spend — one judge call per trial, at effort `high` —
to whoever opens a valid submission.

**Set.** Sourced from `~/.harbor/credentials.json` (the `api_key` field, present
once you have logged in with harbor ≥ 0.20):

```bash
uv run --no-project python -c "import json,os,sys; \
  sys.stdout.write(json.load(open(os.path.expanduser('~/.harbor/credentials.json')))['api_key'])" \
  | gh secret set HARBOR_API_KEY --repo ORCA-bench/ORCA-bench
```

## Label: `lb-submission`

The `static-analysis` job labels every intake submission PR `lb-submission`,
and the `promote` job labels the bot PR it opens the same way — so submission
PRs are filterable from everything else in the repo. **Created** with:

```bash
gh label create lb-submission --repo ORCA-bench/ORCA-bench \
  --description "Leaderboard submission PR" --color 0E8A16
```

## Org setting: let Actions open the bot PR

**Enabled** at both the org and repo level. The `promote` job
(`leaderboard-check.yml`) opens the bot PR with `GITHUB_TOKEN`, which requires,
at the **org** level:

> Org → Settings → Actions → General → Workflow permissions → ☑ **Allow GitHub Actions to create and approve pull requests**
>
> https://github.com/organizations/ORCA-bench/settings/actions

This is org-locked: setting it on the repo alone, while the org forbids it,
returns HTTP 409 (`The organization does not allow GitHub Actions to create or
approve pull requests`). Set the org first, then the repo. Both need the
`admin:org` scope, which the default `gh` token does not carry:

```bash
gh auth refresh -h github.com -s admin:org
gh api --method PUT orgs/ORCA-bench/actions/permissions/workflow \
  -F default_workflow_permissions=read -F can_approve_pull_request_reviews=true
gh api --method PUT repos/ORCA-bench/ORCA-bench/actions/permissions/workflow \
  -F default_workflow_permissions=read -F can_approve_pull_request_reviews=true
```

`default_workflow_permissions` deliberately stays `read` — each job escalates
only what it needs via its own `permissions:` block.

If this ever gets turned off, static analysis still runs and comments on
submission PRs, but `promote` fails when it tries to open the bot PR.

## Leave these at their (non-obvious) defaults

- **`GITHUB_TOKEN` default stays read-only** (the org enforces this; you can't set a write default per-repo). No need to — each job escalates only what it needs via `permissions:`. E.g. `promote` uses `contents: write` to push `submission/pr-N`, which works fine under the read-only default.
- **No required status checks on `main`.** Bot PRs are created by `GITHUB_TOKEN`, so their checks don't re-fire (GitHub loop prevention) — a required check would block every bot PR from merging. The `promote` job re-runs the checks against the exact file before opening the PR, and that is the validation. (Currently satisfied by default: branch protection isn't available on this private repo without GitHub Pro. Re-check this if the repo goes public or the plan changes.)
