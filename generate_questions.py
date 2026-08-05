#!/usr/bin/env python3
"""Generate user-facing issue questions from the incident schedule JSON.

Reads ground-truth rubrics from ``output_dir/json/`` so the LLM knows how
each feature flag behaves and avoids generating overlapping issues.

Examples::

    uv run python generate_schedule_readme.py \
        --schedule schedules/incident_schedule_dev_quick_single.json \
        -od out-test-0210
"""

import json
import os
from pathlib import Path

from check_prediction import format_rubric
from dotenv import load_dotenv
from openai import OpenAI
from utils import get_base_parser

prompt_template = """\
**OpenTelemetry Demo** is composed of microservices written in different
programming languages that talk to each other over gRPC and HTTP; and a load
generator which uses [Locust](https://locust.io/) to fake user traffic.

Here is the full architecture:

```mermaid
graph TD
subgraph Service Diagram
accounting(Accounting):::dotnet
ad(Ad):::java
cache[(Cache<br/>&#40Valkey&#41)]
cart(Cart):::dotnet
checkout(Checkout):::golang
currency(Currency):::cpp
email(Email):::ruby
flagd(Flagd):::golang
flagd-ui(Flagd-ui):::elixir
fraud-detection(Fraud Detection):::kotlin
frontend(Frontend):::typescript
frontend-proxy(Frontend Proxy <br/>&#40Envoy&#41):::cpp
image-provider(Image Provider <br/>&#40nginx&#41):::cpp
load-generator([Load Generator]):::python
payment(Payment):::javascript
product-catalog(Product Catalog):::golang
quote(Quote):::php
recommendation(Recommendation):::python
shipping(Shipping):::rust
queue[(queue<br/>&#40Kafka&#41)]:::java
react-native-app(React Native App):::typescript
postgresql[(Database<br/>&#40PostgreSQL&#41)]

accounting ---> postgresql

ad ---->|gRPC| flagd

checkout -->|gRPC| currency
checkout -->|gRPC| cart
checkout -->|TCP| queue

cart --> cache
cart -->|gRPC| flagd

checkout -->|gRPC| payment
checkout --->|HTTP| email
checkout -->|gRPC| product-catalog
checkout -->|HTTP| shipping

fraud-detection -->|gRPC| flagd

frontend -->|gRPC| ad
frontend -->|gRPC| currency
frontend -->|gRPC| cart
frontend -->|gRPC| checkout
frontend -->|HTTP| shipping
frontend ---->|gRPC| recommendation
frontend -->|gRPC| product-catalog

frontend-proxy -->|gRPC| flagd
frontend-proxy -->|HTTP| frontend
frontend-proxy -->|HTTP| flagd-ui
frontend-proxy -->|HTTP| image-provider

payment -->|gRPC| flagd

queue -->|TCP| accounting
queue -->|TCP| fraud-detection

recommendation -->|gRPC| flagd
recommendation -->|gRPC| product-catalog

shipping -->|HTTP| quote

Internet -->|HTTP| frontend-proxy
load-generator -->|HTTP| frontend-proxy
react-native-app -->|HTTP| frontend-proxy
end

classDef dotnet fill:#178600,color:white;
classDef cpp fill:#f34b7d,color:white;
classDef elixir fill:#b294bb,color:black;
classDef golang fill:#00add8,color:black;
classDef java fill:#b07219,color:white;
classDef javascript fill:#f1e05a,color:black;
classDef kotlin fill:#560ba1,color:white;
classDef php fill:#4f5d95,color:white;
classDef python fill:#3572A5,color:white;
classDef ruby fill:#701516,color:white;
classDef rust fill:#dea584,color:black;
classDef typescript fill:#e98516,color:black;
```

Here is the ground-truth rubric for this incident:

{current_rubric}
{other_rubrics_section}
Please generate the exact issues a user would face and perceive. \
Only emit issues a real human user could actually notice from the UI — \
a visibly broken element, a clearly slow page (multiple seconds), an \
error message they can see, missing or blank content, etc. Do NOT \
emit issues that are only visible in metrics, logs, or numerical \
comparisons. Counter-example: "page latency rose from 10 ms to 190 ms" \
is NOT perceivable — a sub-200ms change is invisible to humans and \
only shows up in dashboards.

For each affected feature area, emit one or more **hard** descriptions \
naming the broad feature area the user is impacted in (one or a few \
alternate phrasings per area). For each distinct user-visible issue you \
identify, additionally emit a **medium** and an **easy** variant of that \
same underlying issue. Granularity levels:

- **hard**: a single short clause from the customer's perspective \
  describing the BROAD feature area affected (e.g., "ads", "cart", \
  "product page", "recommendations", "checkout"), NOT the specific \
  symptom or action. MAX 7 WORDS. Must start with a user-perspective \
  subject such as "users are…", "customers cannot…", "shoppers see…". \
  No product names, page names, error codes, messages, users, regions, \
  or locales. Describe what the user actually sees on the surface they \
  interact with (e.g., "product page", "checkout", "cart"); do NOT name \
  underlying technical components or resources the user does not see \
  (e.g., "image", "API", "service", "cache"). Examples: "users are \
  having issues with ads", "users are having issues with cart", "users \
  cannot view product pages", "users are unable to pay", "users are \
  having payment issues", "customers cannot complete payment", \
  "shoppers are unable to pay", "users say product page not loading \
  properly". Emit at least ONE hard per affected feature area. \
  You MAY emit a small number (typically 1–3) of alternate phrasings \
  for the same feature area to capture different customer voices — \
  vary the subject ("users"/"customers"/"shoppers") and the verbs. \
  Never emit one per individual symptom.
- **medium**: a single short, vague clause. No product names, page names, error \
  codes, messages, users, regions, or locales. \
  Example: "product page is having issues".
- **easy**: name the product / page / feature, the concrete symptom, the \
  specific error or error-message string the user sees, and (when applicable) \
  the affected user, region, or locale. \
  Example: "Product page for National Park Foundation Explorascope is having \
  X issue and is showing Y errors with Z message for W user".

The medium / easy variants for the same underlying issue must \
describe the SAME root cause (the same feature flag) and must be emitted \
together as a pair (one issue's medium/easy before moving on to the next \
issue). The hard variants describe higher-level feature impact and are \
NOT bound 1:1 to those pairs — emit them once per affected feature area, \
independently of the pairs (you may place them before, after, or \
interleaved with the pairs). Use the lowercase granularity tokens \
exactly: "easy", "medium", "hard".
{prior_questions_section}
Return your answer as JSON matching the provided schema.
"""

OTHER_RUBRICS_HEADER = """\
The following other feature flags are also present in the system. \
Do NOT generate issues that are caused by these other flags:

"""

PRIOR_QUESTIONS_HEADER = """\
The following questions have already been generated for this same feature \
flag in prior incidents. Produce DIFFERENT phrasings — vary the subject \
("users", "customers", "shoppers"), the verbs, and the sentence structure. \
Do not repeat any of these wordings:

"""

ISSUES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["granularity", "question"],
                "properties": {
                    "granularity": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "question": {"type": "string"},
                },
            },
        },
    },
}


def load_schedule(json_path: str) -> dict:
    """Load the incident schedule JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_flagd_config(flagd_path: str) -> dict:
    """Load the flagd configuration file."""
    try:
        with open(flagd_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _load_rubrics(json_dir: Path) -> dict[str, dict]:
    """Load all rubric JSON files, keyed by event_id (filename stem).

    Multiple rubrics can share a ``feature_flag`` (e.g. paymentFailure-10%,
    paymentFailure-50%, paymentFailure-100% are distinct events of the same
    flag), so keying by flag would silently collapse them. Each rubric file
    is named after the schedule event it documents, so the filename stem is
    the natural key.

    Args:
        json_dir: Path to ``output_dir/json/`` containing rubric JSON files.

    Returns:
        A dict mapping event_id to ``{"flag": str, "markdown": str}``.

    """
    rubrics: dict[str, dict] = {}
    for jf in sorted(json_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        rubrics[jf.stem] = {
            "flag": data["feature_flag"],
            "markdown": format_rubric(data, include_frontend=True),
        }
    return rubrics


def _collect_prior_questions_for_flag(
    questions_dir: Path,
    flag: str,
    exclude_path: Path | None = None,
) -> list[tuple[str, str]]:
    """Return (granularity, question) pairs from prior-event files for ``flag``.

    Scans every per-event JSON in ``questions_dir`` and collects questions
    whose bundle ``answer`` matches ``flag``. ``exclude_path`` (typically
    the current event's output path under ``--force`` re-generation) is
    skipped so an event never sees its own prior content.

    Args:
        questions_dir: Directory containing per-event question JSON files.
        flag: Feature flag name to filter on.
        exclude_path: If provided, this file is skipped during scanning.

    Returns:
        List of (granularity, question) tuples from prior question files.

    """
    prior: list[tuple[str, str]] = []
    for jf in sorted(questions_dir.glob("*.json")):
        if exclude_path is not None and jf == exclude_path:
            continue
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("answer") != flag:
            continue
        for q in data.get("questions", []):
            prior.append((q["granularity"], q["question"]))
    return prior


def generate_questions(
    client: OpenAI,
    schedule_data: dict,
    flagd_data: dict,
    rubrics: dict[str, dict],
    *,
    model: str,
    effort: str,
    questions_dir: Path,
    prompts_dir: Path,
    schedule_name: str,
    event_ids: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Generate user-facing issue questions from the schedule data.

    Writes one JSON file per event into ``questions_dir`` named
    ``{schedule_name}-{event_id}-{model}-{effort}.json``. If that file
    already exists the event is skipped (letting re-runs resume
    incrementally), unless ``force`` is set. Each call's full prompt is
    also dumped to ``prompts_dir`` as ``{same_stem}.md`` for offline
    debugging — written BEFORE the LLM call so a failed call still
    leaves the prompt on disk.

    Args:
        client: OpenAI client for generating questions from events.
        schedule_data: The incident schedule data loaded from JSON.
        flagd_data: The flagd configuration data loaded from JSON.
        rubrics: Dict mapping event_id to ``{"flag", "markdown"}``.
        model: LLM model name to use for generation.
        effort: Reasoning effort level for LLM calls.
        questions_dir: Directory to write per-event JSON files into.
        prompts_dir: Directory to write per-event prompt markdown into.
        schedule_name: Stem of the schedule JSON filename (e.g.
            ``"incident_schedule_dev_quick_single"``). Included in each
            per-event filename so multiple schedules can coexist in one
            questions dir.
        event_ids: If set, only generate questions for events whose ``id`` is
            in this list. All other events are skipped.
        force: If True, regenerate an event even when its output file already
            exists.

    Returns:
        The list of per-event JSON paths that were written or already existed.

    """
    written: list[Path] = []
    events = schedule_data.get("schedule", [])
    flagd_flags = flagd_data.get("flags", {}) if flagd_data else {}

    for event in events:
        flag = event.get("flag", "")
        value = event.get("value", "")
        event_id = event.get("id", "")

        if event_ids is not None and event_id not in event_ids:
            continue

        # Skip events where the value is the default variant (i.e., normal state)
        flag_info = flagd_flags.get(flag, {})
        default_variant = flag_info.get("defaultVariant", "")
        if value == default_variant:
            continue

        out_path = questions_dir / f"{schedule_name}-{event_id}-{model}-{effort}.json"
        if out_path.exists() and not force:
            print(f"{event_id}: already generated, skipping ({out_path.name})")
            written.append(out_path)
            continue

        # Build rubric context for the prompt
        if event_id not in rubrics:
            print(
                f"{event_id} ({flag}): no rubric at output_dir/json/{event_id}.json, "
                f"skipping"
            )
            continue
        current_rubric = rubrics[event_id]["markdown"]

        # Include one representative rubric per OTHER flag, deduped across
        # variants of the same flag (e.g. paymentFailure-10%/50%/100% share
        # one entry in the "other rubrics" section).
        seen_flags = {flag}
        other_rubric_parts: list[str] = []
        for r in rubrics.values():
            if r["flag"] in seen_flags:
                continue
            seen_flags.add(r["flag"])
            other_rubric_parts.append(r["markdown"])
        if other_rubric_parts:
            other_rubrics_section = (
                OTHER_RUBRICS_HEADER + "\n---\n".join(other_rubric_parts) + "\n"
            )
        else:
            other_rubrics_section = ""

        prior_pairs = _collect_prior_questions_for_flag(
            questions_dir, flag, exclude_path=out_path
        )
        if prior_pairs:
            by_gran: dict[str, list[str]] = {}
            for gran, q in prior_pairs:
                by_gran.setdefault(gran, []).append(q)
            lines = [
                f"- [{gran}] {q}"
                for gran in ("hard", "medium", "easy")
                for q in by_gran.get(gran, [])
            ]
            prior_questions_section = (
                "\n" + PRIOR_QUESTIONS_HEADER + "\n".join(lines) + "\n"
            )
        else:
            prior_questions_section = ""

        input_prompt = prompt_template.format(
            current_rubric=current_rubric,
            other_rubrics_section=other_rubrics_section,
            prior_questions_section=prior_questions_section,
        )
        prompt_path = prompts_dir / f"{out_path.stem}.md"
        prompt_path.write_text(input_prompt)
        response = client.responses.create(
            model=model,
            input=input_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "user_visible_issues",
                    "schema": ISSUES_SCHEMA,
                    "strict": True,
                },
                "verbosity": "medium",
            },
            reasoning={"effort": effort, "summary": "auto"},
            tools=[],
            store=True,
            include=[
                "reasoning.encrypted_content",
                "web_search_call.action.sources",
            ],
        )
        reasoning_summary = [
            item.summary[0].text
            for item in response.output
            if item.type == "reasoning" and item.summary
        ]
        issues = json.loads(response.output_text)["issues"]
        print(f"{event_id} ({flag})")
        questions = []
        for i, issue in enumerate(issues):
            granularity = issue["granularity"]
            problem = issue["question"]
            print(f"* [{granularity}] {problem}")
            questions.append(
                {
                    "id": f"{event_id}-{i:02d}-{granularity}",
                    "question": problem,
                    "granularity": granularity,
                }
            )
        print()

        bundle = {
            "event": event,
            "answer": flag,
            "model": model,
            "effort": effort,
            "prompt": input_prompt,
            "response": response.output_text,
            "reasoning_summary": reasoning_summary,
            "questions": questions,
        }
        out_path.write_text(json.dumps(bundle, indent=2))
        written.append(out_path)

    return written


def main() -> None:
    """Main entry point."""
    parser = get_base_parser()
    parser.add_argument(
        "--flagd",
        type=Path,
        default=Path("opentelemetry-demo/src/flagd/demo.flagd.json"),
        help="Path to flagd config file (default: opentelemetry-demo/src/flagd/demo.flagd.json)",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        default=None,
        help="Only generate questions for the given event id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate events whose per-event question file already exists.",
    )
    args = parser.parse_args()

    json_path = args.schedule
    flagd_path = args.flagd
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading schedule from: {json_path}")
    schedule_data = load_schedule(json_path)

    print(f"Loading flagd config from: {flagd_path}")
    flagd_data = load_flagd_config(flagd_path)

    # Load ground-truth rubrics
    json_dir = output_dir / "json"
    if not json_dir.is_dir():
        print(f"Error: {json_dir} does not exist")
        raise SystemExit(1)
    print(f"Loading rubrics from: {json_dir}")
    rubrics = _load_rubrics(json_dir)
    print(f"  Loaded {len(rubrics)} rubric(s): {', '.join(sorted(rubrics))}")

    print("Generating questions...")
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )
    effort = args.effort or "high"
    if args.event_id is not None:
        print(f"Filtering to event id(s): {', '.join(args.event_id)}")

    questions_dir = output_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "question_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    written = generate_questions(
        client,
        schedule_data,
        flagd_data,
        rubrics,
        model=args.model,
        effort=effort,
        questions_dir=questions_dir,
        prompts_dir=prompts_dir,
        schedule_name=json_path.stem,
        event_ids=args.event_id,
        force=args.force,
    )

    print(f"Done! {len(written)} per-event file(s) in {questions_dir}")


if __name__ == "__main__":
    load_dotenv()
    main()
