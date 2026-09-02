# Steps to publish Harbor jobs to Harbor Hub

1. Create new dataset:

- Tasks: update the Harbor tasks and follow the provided hint to publish them to Harbor hub

```bash
uv run python build_harbor_tasks.py -od out-0804 -dd data-0418 \
  --templates-dir harbor-template --force
```

- Dataset(s): re-run the build with `--split` to write `split.json` plus four
  dataset manifests, then publish each one. Names derive from `--org`, so the
  full build and the public split can never claim the same name — a collision
  that would otherwise republish the public dataset with the private tasks
  included.

| Dataset | Tasks | Contents |
|---|---|---|
| `orca-bench/orca-bench` | 755 | public split |
| `orca-bench/orca-bench-private-internal` | 324 | held-out split, answers included |
| `orca-bench/orca-bench-private` | 324 | the same tasks with the answers removed |
| `orca-bench/orca-bench-verified` | 40 | verified subset of the public split |

```bash
uv run python build_harbor_tasks.py -od out-0804 -dd data-0418 \
  --templates-dir harbor-template --force --split \
  --verified-json sampled_tasks.json \
  --dataset-author "Albert Gong <ag2435@cornell.edu>"

# Writes out-0804/harbor/split.json, the answer-free tasks/<task_id>-hidden/
# dirs, and out-0804/harbor/datasets/{public,private-internal,private,verified}/
uv run harbor add out-0804/harbor/tasks --scan
uv run harbor publish out-0804/harbor/datasets/public --private
uv run harbor publish out-0804/harbor/datasets/private-internal --private
uv run harbor publish out-0804/harbor/datasets/private --private
uv run harbor publish out-0804/harbor/datasets/verified --private
```

> [!IMPORTANT]
> `orca-bench/orca-bench-private` is the split to hand to submitters. Its tasks
> ship without `tests/expected.json`, `tests/rubrics/`,
> `tests/check_prediction.py` or `solution/`, and their
> `task.toml` `[metadata]` keeps only `category`, `tags`, `user_facing_issue`,
> `current`, `reported_styled` and `difficulty` — every other field,
> `incident_time`, `flag` and `events` included, names or dates the root
> cause. (`reported_styled` is the phrasing the prompt already shows; the ISO
> `reported` field is `incident_time + offset_minutes`, so it is not kept.) `harbor run` puts the task directory on the machine that
> runs it, so publishing `-private-internal` to submitters would hand over the
> answers for all 324 tasks.
>
> The hidden tasks are not scored in-container. Harbor requires every task to
> have a test script and to leave a reward file, so their `tests/test.sh` copies
> `report.md` into the verifier dir and writes an empty rewards map (`{}`). The
> trial finishes clean, with no metrics, instead of recording a 0 that every
> mean would average in. Score these trials out-of-band with `run_llm_judge.py`,
> which reads each trial's report and its oracle twin's rubric straight from the
> Hub:
>
> ```bash
> uv run python run_llm_judge.py -od out-0804 --job <hidden-job-uuid>
> ```
>
> One thing to confirm on the first real run: how the Hub stores an empty
> rewards map. `trial_metric` returns `None` (excluded, as intended) when the
> row's `evals` is empty, but raises `MissingRewardMetricError` if the Hub
> instead materializes a group with an empty metrics list. Nothing hits that
> path today — `submission_trials` filters on `source == orca-bench/orca-bench`,
> so private trials never reach it — but a leaderboard extended to the private
> split would need to handle it.
>
> Do not put a placeholder metric in that map. `trial_metric` raises
> `MissingRewardMetricError` when a trial reports metrics but not the one asked
> for, and the Hub takes a row's own reward scalar from the alphabetically first
> numeric entry — which already mis-scored this leaderboard once, when
> `hallucinate_any` sorted ahead of `reward`.

> [!NOTE]
> `split.json` records the disjoint partition under `splits` (`public`,
> `private-internal`) and the overlapping selections under `views` (`verified` is
> a subset of public; `private-hidden` repackages private-internal under new
> package names). `convert_job.py` assigns `routing[task_id]` while looping over
> `splits`, so an entry sharing a task_id there would silently reroute those
> trials to whichever dataset iterated last.

- Create a leaderboard for each dataset by running the following command:

```bash
```

> [!NOTE]
> The leaderboard code is pinned to a specific dataset revision at INSERT.
> Updating the dataset superficially (e.g., modifying the description) moves the revision number but doesn't affect the leaderboard.

2. Update `config.json` and `result.json` inside a jobs directory to the latest Harbor tasks and dataset.

The script does three things in one pass: it associates `task_name` for each trial's task with a `package_name` in `split.json` and **filters out anything not published on the
Hub**, rsyncs only the surviving trial dirs to `--dst`, then rewrites
`config.json`/`result.json` to the package task and writes the per-trial
`lock.json` that `harbor upload` requires (harbor 0.6.3 never wrote one).

```bash
uvx --from harbor python convert_job.py \
  --src /root/benchmark/src/sre-agent/examples/otel-demo/jobs-sub \
  --dst /mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo/jobs-sub-backfilled-scores-2 \
  --split /root/benchmark/src/ORCA-bench/out-0804/harbor/split.json
```

> [!NOTE]
> Trials whose task is in no split — the `-02-medium` tasks — have no published
> Hub task to point at and are excluded before the copy. Of 1,449 trials per job,
> 1,079 convert (755 public + 324 private) and 370 are dropped.

3. Trials scored before the verifier emitted `reward.json` carry a bare-float
`verifier/reward.txt` plus a `verifier/details.json` summary with no rubric
detail. Rewrite each `verifier/` dir into the current format by recovering
the judge result from the `judge-*.json` that `run_llm_judge.py` wrote for that
trial, so the verdicts are reused as-is and no LLM calls are made.

```bash
uv run python backfill_verifier_outputs.py \
  --jobs-dir /mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo/jobs-sub-backfilled-scores \
  --scores-dir /mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo/out-leaderboard
```

Each paired trial gets `reward.json` + `reward-details.json`, and its
`result.json` has `verifier_result.rewards` repointed at the same scores; the
stale `reward.txt` / `details.json` are removed. `report.md`, `log.txt` and
`test-stdout.txt` are left untouched — they are the real container transcript.

`result.json` has to move with `reward.json`: `harbor upload` rebuilds each Hub
trial from `result.json` alone and never re-reads the `verifier/` files, which
upload as opaque artifacts. Updating only `reward.json` would publish the
pre-backfill reward to the Hub — and to the leaderboard reading it — while the
new score stayed visible only by downloading the trial. A trial whose
`result.json` is missing fails before anything is written, so it can't end up
half-converted.

The run is idempotent. Before finishing it validates every `reward.json` against
harbor's `VerifierResult`, then parses each `result.json` with harbor's
`TrialResult` — the same step `harbor upload` takes — and fails if the two
disagree.

> [!NOTE]
> On the five job dirs above: 5396 trials, 5395 converted (3740 `llm_judge`,
> 975 `no_incident_llm_judge`, 680 `empty_report`). The one skip,
> `d1-n1-loadgeneratorfloodhomepage__q6Twx3F`, died in `_setup_agent` (see its
> `exception.txt`), so it produced no report and was never judged — it has no
> score file to pair with and its `verifier/` dir stays empty.

4. Split jobs into public and private job directories.

Dataset names are read from `split.json`, not hardcoded, so this keeps working
when a dataset is republished under a different name. `--out` takes
`<split>=<path>` pairs and is repeatable, so extra splits (e.g. `verified`) need
no code change.

Trial membership comes from each trial's `lock.json` `task.source`, which step 3
set to the published dataset name. Each output tree gets real job-level
`config.json`/`result.json` — they must differ per split (scoped dataset list,
trial count) — while every trial dir is a **relative symlink** back to `--src`,
so no trial payload is duplicated.

```bash
uv run python split_jobs.py \
  --src /mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo/jobs-sub-backfilled-scores-2 \
  --split out-0804/harbor/split.json \
  --out public=/mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo/jobs-sub-backfilled-scores-public-2 \
  --out private=/mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo/jobs-sub-backfilled-scores-private-2
```

`harbor upload` traverses the symlinks correctly: the link is a *middle* path
component (`<trial>/agent`, `<trial>/config.json`), so `tarfile` recurses
normally instead of storing a link stub. Verified — archives contain real files.

> [!NOTE]
> The trees cost ~23 MB on disk for ~98 GB of exposed content. Links are
> relative, so the three trees can be moved together but not individually.

5. Re-ID jobs and trials before uploading anything that was uploaded before.

**This step is mandatory whenever the content changed but the trials did not.**
Harbor's storage upload is skip-if-exists — `upload_bytes` swallows the
"already exists" error — and object paths are `jobs/{job_id}/job.tar.gz` and
`trials/{trial_id}/trial.tar.gz`. Re-uploading with unchanged ids therefore
writes nothing and the Hub keeps serving the **old** archive, silently. Deleting
the Hub job does not help: it removes the DB rows but leaves the blobs, so a
deleted job's storage still shadows a re-upload.

Both ids must change. Re-IDing only the trials leaves the job-level archive stale.

```bash
for tree in public private; do
  for job in /mnt/.../jobs-sub-backfilled-scores-$tree-2/*/; do
    uv run python reid_trials.py --job-dir "$job" --tag "out0804-$tree"
  done
done
```

> [!IMPORTANT]
> Use a **distinct tag per tree**. Ids are UUIDv5 over stable names, and the
> public and private trees share job directory names — a single shared tag gives
> both the same job UUID, and the two would merge into one Hub job. Trial ids are
> safe either way (trial dir names are disjoint), but job ids are not.

> [!TIP]
> Each republish needs a **new** tag. The derivation is deterministic, so reusing
> a tag regenerates the same ids, which land on storage paths that already exist
> — straight back into the silent-skip failure above.

To confirm before uploading, check that no id already exists on the Hub —
querying the `trial` table alone is not enough, since a deleted job leaves blobs
behind. Prefer ids that have never been used at all.

6. Upload the job directories to Harbor Hub.

`harbor upload` takes one job dir at a time, so loop over the split trees. Use
`uvx harbor` rather than a repo venv unless that venv has a current harbor —
upload needs a recent version and the credentials in `~/.harbor/credentials.json`
(check with `uvx harbor auth status`).

```bash
B=/mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo
for tree in public private; do
  for job in "$B/jobs-sub-backfilled-scores-$tree-2"/*/; do
    echo "=== $tree $(basename "$job") ==="
    uvx harbor upload "$job"
  done
done
```

Each job runs in two phases: the per-trial uploads (fast, 10 concurrent by
default, `-c` to change), then finalization, which tars **the whole job dir** into
one `job.tar.gz` and ships it. Finalization dominates — it is single-threaded
gzip over tens of GB. Budget roughly 10–45 min per 755-trial public job and
5–17 min per 324-trial private job; ~2.5–3 h for ten jobs totalling ~97 GB.

Visibility is private by default on new uploads. Passing neither `--public` nor
`--private` on a re-upload leaves the server-side value unchanged.

```bash
uvx harbor upload "$job" --public     # publish
uvx harbor upload "$job" --private    # unpublish
```

> [!CAUTION]
> Think before making private-split jobs public — they are trajectories over the
> held-out tasks, and publishing them exposes reasoning about held-out incidents.

The command is an **idempotent sweep**: trials already present are skipped, missing
ones are filled in, and the job is finalized only if `archive_path` is still NULL.
Re-running after any failure converges, so wrapping each job in a small retry loop
costs nothing and covers transient storage errors (a `544` from Supabase killed one
job's finalization on an earlier run).

> [!WARNING]
> Do not trust the CLI summary alone. The progress bar reported `324/324` on the
> job that hit the `544`, but only 277 trial rows had actually landed. Verify
> against the Hub afterwards.

Verify each job by querying the Hub directly: the `trial` table row count for the
`job_id` should equal the number of trial dirs on disk, the `job` row's
`archive_path` should be non-NULL (NULL means finalization never completed), and
every row's `task_content_hash` should match the digest in the published dataset
manifest. Spot-check freshness by downloading a trial and diffing its `verifier/`
against local:

```bash
uvx harbor trial download <trial_id>
```

> [!NOTE]
> If a trial's `verifier/` on the Hub differs from local, the ids were reused and
> storage silently skipped the write — go back to step 5.

Superseded jobs from an earlier publish can be removed once the new ones verify.
`delete` accepts several ids at once:

```bash
uvx harbor hub job delete <job_id> [<job_id> ...] --yes
```

7. Submit the jobs to the leaderboard.

`lb submit` runs filter → metadata → open-prs in one go, opening one PR per
`(agent, agent version, model, reasoning effort)`. Run it from `leaderboard/`:

```bash
cd leaderboard
uv run lb submit https://hub.harborframework.com/jobs/<job_id> [<job_id> ...]
```

Each PR is checked, then promoted to a bot PR carrying the computed metrics;
merging the bot PR posts the row. See [SUBMIT.md](leaderboard/SUBMIT.md) for the
review pipeline and what the checks enforce.

8. Attach the leaderboard to the new dataset revision.

A leaderboard is bound to specific dataset **version ids**, not to the package.
Publishing a revision mints a new version id and nothing attaches it to an
existing board, so the new revision's page shows no leaderboard until you do.

`harbor hub leaderboard update` cannot fix this — its updatable set is only
`title`, `metadata_schema`, `metrics_schema`, `columns`, `rank_by` and
`visibility`, and `dataset_version_ids` appears in neither the create nor the
update config. (`leaderboard-migrate`, despite the name, is the atomic
definition+rows path, not version migration.) The `leaderboard-update` edge
function does accept the field, so go through it directly.

First collect the version id of every revision the board should appear on:

```bash
uv run python -c "
import asyncio
from harbor.registry.client.package import PackageDatasetClient
c = PackageDatasetClient()
for ref in ('1', '2', 'latest'):
    m = asyncio.run(c.get_dataset_metadata(f'orca-bench/orca-bench@{ref}'))
    print(f'{ref:8} {m.dataset_version_id}  {m.version}')
"
```

Then set the list, passing **every** id the board should be attached to. Treat
the field as replacing the list rather than appending to it: both ids were
passed together when this was first done, so whether a shorter list detaches the
omitted revisions is untested — including them all costs nothing and avoids
finding out the hard way.

```bash
uv run python -c "
import asyncio
from harbor.hub.leaderboards import LeaderboardClient

LEADERBOARD_ID = 'dcb96003-d991-4f98-97b6-3813c35daa81'
VERSION_IDS = ['<v1_version_id>', '<v2_version_id>']

async def go():
    c = LeaderboardClient()
    board = await c.get(leaderboard_id=LEADERBOARD_ID)
    await c.update_definition({
        'leaderboard_id': LEADERBOARD_ID,
        # Optimistic-concurrency guard: fails if the board changed meanwhile.
        'expected_updated_at': board.updated_at,
        'dataset_version_ids': VERSION_IDS,
    })
asyncio.run(go())
"
```

Verify the binding, and that the rows and columns came through untouched:

```bash
uv run harbor hub leaderboard show orca-bench/orca-bench/orca-bench --json \
  | jq '.leaderboard | {dataset_version_ids, columns: (.columns | length)}'
uv run harbor hub leaderboard row list orca-bench/orca-bench/orca-bench
```

> [!NOTE]
> Updating description increments the revision number, so we often need to link the leaderboard from a prior version (e.g., v1) to later versions.

> [!WARNING]
> Attaching the board to a new revision is display only — it does not re-pin the
> benchmark. `DATASET_REF` in
> [`core/hub.py`](leaderboard/src/leaderboard/core/hub.py) still points at
> whichever revision it did before, so existing rows were scored against that one
> and new submissions are still validated against it. If the new revision is
> meant to be canonical, re-pin `DATASET_REF` as well — but that invalidates
> every uploaded trial, since the per-trial digest check compares each trial's
> task ref against the pinned version's. Only worth doing when the task content
> actually changed; a metadata-only republish leaves scores comparable.