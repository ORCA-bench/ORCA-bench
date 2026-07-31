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
`leaderboard-judge.yml` / `leaderboard-apply.yml` workflows). Those are **not**
ported: they run judge agents on Modal against per-task agent images on Docker
Hub, which ORCA-bench's privileged docker-in-docker otel-demo tasks don't have.

Consequences: no `ANTHROPIC_API_KEY` / Modal secrets are needed, and the
`disqualified_trials` / `credited_trials` override lists are maintained by hand
on the bot PR instead of by `/apply`. The metric still honors them.

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

Published as `orca-bench/ORCA-bench` (755 tasks), pinned in
[`core/hub.py`](src/leaderboard/core/hub.py) as
`sha256:39149656128e6279e88d3b123374b6e9222bb71a98b55213ac6355a6f156a67a`.

**The dataset is currently private.** That has one confusing consequence worth
knowing before you debug anything: an *unauthenticated* caller gets

```
ValueError: Tag 'latest' not found for dataset 'orca-bench/ORCA-bench'
```

which reads like the dataset doesn't exist. It does — you are just not logged
in. Check with `harbor auth status` before concluding anything from that error.
Note also that a credentials file written by an older harbor is rejected by
0.20 ("Found credentials from an older Harbor version"), which silently
downgrades you to anonymous.

While the dataset stays private, CI needs `HARBOR_API_KEY` to see it at all,
and external contributors cannot run the benchmark. Make the dataset public
before opening submissions up.

## The leaderboard on the hub

**Created** — `orca-bench/ORCA-bench` → leaderboard `orca-bench`, title
"ORCA-bench", **visibility `private`** (matching the dataset), id
`1b72818f-bd2e-4051-a3d1-634fe44808d7`. This matches `LEADERBOARD_PACKAGE` /
`LEADERBOARD_NAME` in [`ci/submit.py`](src/leaderboard/ci/submit.py).

Make it public in the same change that makes the dataset public — a public
leaderboard over a private dataset shows scores nobody can reproduce.

### Gotcha: `package` vs `package_id`

**The hub never accepts `orca-bench/ORCA-bench` as an API selector.** Every
endpoint validates a `package` slug against a lowercase-only pattern —

```
/^[a-z0-9][a-z0-9_-]*\/[a-z0-9][a-z0-9_.-]*$/
```

— which the uppercase in `ORCA-bench` fails. The slug is fine for display and
for hub URLs, but anything that talks to the API must select the package by id
(`package_id`, or `leaderboard_id` where supported). This bit twice: once on
`leaderboard create`, and again on `leaderboard-row-create` in `ci/submit.py`,
which is why `LEADERBOARD_PACKAGE_ID` exists alongside `LEADERBOARD_PACKAGE`.

Renaming the dataset package to lowercase would remove the need for the
indirection entirely; `tests/test_submit.py` asserts the slug is *not* hub-valid,
so that test fails the day it is renamed, flagging the workaround as removable.

The definition is checked in as [`leaderboard.json`](leaderboard.json), but it
**cannot be passed to `create` as-is** — the CLI reports:

```
Error: invalid leaderboard.json:
  package: String should match pattern '^*/*$'
```

Swap `package` for the dataset's `package_id` UUID
(`6d2769e3-f6e9-4496-9830-ea8997cb9e6d`) — supply one or the other, never both:

```bash
uv run python -c "import json; d=json.load(open('leaderboard.json')); \
  d.pop('package'); d['package_id']='6d2769e3-f6e9-4496-9830-ea8997cb9e6d'; \
  json.dump(d, open('/tmp/lb.json','w'), indent=2)"
uv run harbor hub leaderboard create --config /tmp/lb.json --json
```

`leaderboard.json` keeps the human-readable `package` slug because that is what
`ci/submit.py` and the hub URL use; only the `create`/`update` CLI needs the
UUID substitution.

The `metadata_schema` / `metrics_schema` are the contract merged submissions
must satisfy. `metrics` = `accuracy` + `accuracy_stderr` (required) plus the
whole-submission resource totals (`total_tokens` = input + output,
`total_cost_usd` — optional, omitted when the trials don't report the underlying
telemetry). `metadata` = the display fields, plus the `pr` markdown link cell
(`[#N](…)` → the promoted bot PR) that CI stamps in at promotion.

Note `accuracy` is the **mean graded reward** as a percentage, not a pass rate —
ORCA-bench verifiers emit a normalized reward in `[0, 1]`.

Verify it, or export the live definition if you need to amend it (`update`
takes the same `package_id` substitution as `create`):

```bash
uv run harbor hub leaderboard list
uv run harbor hub leaderboard export orca-bench/ORCA-bench/orca-bench
uv run harbor hub leaderboard update --config /tmp/lb.json
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
  print(asyncio.run(PackageDatasetClient().get_dataset_metadata('orca-bench/ORCA-bench@latest')).version)"
```

(The field is `.version`, not `.ref` — it holds the `sha256:` digest. There is
no `harbor hub dataset` subcommand.)

## Secrets

`Settings → Secrets and variables → Actions`, or `gh secret set <NAME> --repo <owner>/<repo>`:

| Secret | Used by | For |
| --- | --- | --- |
| `HARBOR_API_KEY` | check, promote, merge, close | hub/registry access, trial cloning, leaderboard submit |

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
