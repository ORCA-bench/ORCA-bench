# OpenSearch wipe recovery (provenance)

> \[!NOTE\]
> These scripts are **archived for provenance only** — they are *not* part of the
> intended dataset-construction pipeline (see the warning at the top of the main
> [`../README.md`](../README.md#constructing-dataset)).

During our data collection the OpenSearch instance was hit by `DELETE INDEX` /
`DELETE BY QUERY` ransomware events that wiped portions of the daily Jaeger /
OTel indices captured in our scheduled snapshots. The **released Harbor tasks**
were built on data recovered by the *post hoc* procedure documented here:

1. Detect each wipe from the snapshot size curve and emit a restore manifest —
   [`build_restore_manifest.py`](build_restore_manifest.py)
2. Replay the missing data from peak snapshots —
   [`restore_from_manifest.sh`](restore_from_manifest.sh)
3. Consolidate the recovered `*-lifetime*` indices back into daily indices —
   [`consolidate_lifetimes.py`](consolidate_lifetimes.py) /
   [`create_aliases.py`](create_aliases.py)
4. Persist point-in-time-correct `*-consolidated` snapshots that the (then)
   Harbor entrypoint restored — [`per_snapshot_consolidate.py`](per_snapshot_consolidate.py)

The full runbook is in
[`RESTORE_HISTORICAL_INDICES.md`](RESTORE_HISTORICAL_INDICES.md).

> \[!IMPORTANT\]
> The runbook and scripts assume they are run from the **repository root**
> (paths like `./load_snapshot.sh`, `./check_health.sh`, `data-XXXX/…` are
> relative to the repo root, not this subfolder). The current
> `harbor-template` entrypoint restores the plain scheduled snapshot, not
> `${SNAPSHOT_NAME}-consolidated`; restoring consolidated state was specific to
> the recovered data behind the released tasks.
