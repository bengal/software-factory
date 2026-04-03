# Software Factory

A dark software factory built on the [Attractor](https://github.com/strongdm/attractor) pipeline specification. Humans write specs during the day; the factory processes them overnight; humans review the results in the morning.

## How it works

```
     DAYTIME (humans)                         NIGHTTIME (factory)
     ================                         ===================

     1. Review last night's report            start
     2. Accept/reject commits                   |
     3. Resolve quarantined items             ingest
     4. Write new specs or revise old ones      |
     5. Place specs in specs/ directory       triage & order
     6. Start the factory                       |
                                              [for each item:]
                                              understand → plan → implement
                                                → verify → package → commit
                                                |
                                              report
                                                |
                                              exit

                          (next morning, repeat)
```

The factory runs a pipeline that processes each spec through: **understand -> plan -> implement -> verify**, with a diagnose/fix retry loop on failure and quarantine for items that can't be fixed. Two pipeline variants are included: sequential (default, processes items one at a time with a git commit after each) and parallel.

## Setup

```bash
# Install
pip install -e .

# Set your LLM API key
export ANTHROPIC_API_KEY=sk-ant-...
# or for Claude via Vertex AI:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=us-east5
# or
export OPENAI_API_KEY=sk-...
# or
export GEMINI_API_KEY=...
```

## Usage

### 1. Create a factory config in your project

```bash
cd /path/to/your-project

cat > factory-config.json << 'EOF'
{
    "specs_dir": "specs/",
    "output_dir": "output/",
    "provider": "anthropic",
    "model": "claude-sonnet-4-5",
    "verify": {
        "test_command": "pytest -x -q",
        "lint_command": "ruff check .",
        "typecheck_command": "",
        "build_command": ""
    },
    "limits": {
        "max_retries": 5,
        "implement_timeout": 1800,
        "verify_timeout": 600,
        "total_timeout": 28800,
        "max_tokens": 2000000
    }
}
EOF
```

| Limit | Default | Description |
|---|---|---|
| `max_retries` | 3 | Max verify→diagnose→fix cycles per work item |
| `implement_timeout` | 300 | Seconds before the implement stage is killed |
| `verify_timeout` | 120 | Seconds before each verify command is killed |
| `total_timeout` | 1800 | Seconds before the entire factory run is killed |
| `max_tokens` | 0 | Total LLM token budget (input + output) for the run. 0 = unlimited. 2M is a reasonable safety limit for small projects. |

### 2. Write specs

Each spec is a markdown file in `specs/`:

```markdown
# specs/add-user-search.md

## Goal

Add a search endpoint that lets users search by name or email.

## Requirements

- GET /api/users/search?q=<query>
- Search matches partial name or email (case-insensitive)
- Return max 50 results, sorted by relevance

## Constraints

- Do not modify existing user endpoints
- Use the existing database connection pool

## Acceptance Criteria

- [ ] Endpoint returns matching users as JSON
- [ ] Empty query returns 400
- [ ] Search is case-insensitive
```

All sections are optional but recommended. The more precise the spec, the better the output.

### Incremental runs

The factory tracks which specs have been successfully processed. On subsequent runs,
unchanged specs are skipped automatically. This is based on file checksums stored in
`output/.factory/manifest.json`.

- **Unchanged spec** → skipped (already committed)
- **Revised spec** → re-processed; the factory sees the existing code and implements the delta
- **New spec** → processed normally
- **Deleted spec** → ignored (previous commits remain in git history)
- **Quarantined spec** → retried on next run (not recorded in manifest until it succeeds)

To force a full re-run, delete the manifest:

```bash
rm output/.factory/manifest.json
```

### 3. Run the factory

**Directly:**

```bash
factory run
```

With overrides:

```bash
factory run --specs-dir specs/ --output-dir output/ --provider anthropic --model claude-sonnet-4-5
```

**In a container (recommended):**

The factory executes LLM-generated code, so running it in a sandbox is strongly recommended.
The `run-factory.sh` script builds and runs a Podman container automatically:

```bash
cd /path/to/your-project
ANTHROPIC_API_KEY=sk-ant-... /path/to/software-factory/scripts/run-factory.sh
```

The script auto-detects the project type from manifest files (`Cargo.toml` → Rust,
`pyproject.toml` → Python, `go.mod` → Go, `CMakeLists.txt`/`Makefile` → C/C++) and
builds a container with the appropriate toolchain. You can also specify a preset explicitly:

```bash
/path/to/software-factory/scripts/run-factory.sh --preset rust
```

For Claude via Vertex AI:

```bash
GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/sa-key.json \
GOOGLE_CLOUD_PROJECT=my-project \
GOOGLE_CLOUD_LOCATION=us-east5 \
/path/to/software-factory/scripts/run-factory.sh
```

The service account key file is mounted read-only inside the container.

Extra arguments are passed through to `factory run`:

```bash
/path/to/software-factory/scripts/run-factory.sh --pipeline pipelines/dark_factory.dot
```

**Custom container image:**

For projects that need specific dependencies (e.g., `dnsmasq` for integration tests),
create a `Containerfile.factory` in the project root:

```dockerfile
FROM software-factory-base

RUN dnf install -y gcc dnsmasq iproute && dnf clean all
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
```

The script will use this instead of a preset when it exists.

### 4. Review results

The factory writes to `output/`:

```
output/
  report.md                          # Morning briefing - start here
  work-items/
    add-user-search/
      status.json                    # SUCCESS or QUARANTINED
      understanding.md               # Factory's interpretation of the spec
      plan.md                        # Implementation plan
      diff.patch                     # The produced code changes
      verification_result.md         # Test/lint/build output
    fix-login-bug/
      status.json                    # QUARANTINED
      understanding.md
      plan.md
      diagnosis.md                   # Why it failed, what was tried
      verification_result.md
```

With the sequential pipeline (default), successful work items are committed directly.
Review the commits with `git log`. Quarantined items have artifacts in `output/` for diagnosis.

## Other commands

```bash
# Validate a pipeline DOT file without running it
factory validate pipelines/dark_factory.dot

# Run any custom DOT pipeline
factory run-pipeline my-pipeline.dot
```

## Pipeline stages (sequential, default)

| Stage | Type | Purpose |
|---|---|---|
| **start** | boundary | Entry point |
| **ingest** | tool | Reads `.md` files from `specs/` into structured work items |
| **triage & order** | LLM | Analyzes dependencies, orders items so each builds on the previous |
| **next_item** | conditional | Loop: pick next work item or finish |
| **understand** | LLM | Deep analysis of the spec against the codebase |
| **plan** | LLM | Creates a concrete implementation plan |
| **implement** | LLM | Writes the code |
| **verify** | tool | Runs tests, linter, type checker, build |
| **check_result** | conditional | Routes to package (pass) or diagnose (fail) |
| **diagnose** | LLM | Analyzes verification failures |
| **fix** | LLM | Applies fixes, loops back to verify (up to 5 retries) |
| **quarantine** | tool | Saves state for failed items, loops to next item |
| **package** | tool | Generates diff.patch, copies artifacts |
| **commit** | tool | Commits the work item's changes to git |
| **report** | LLM | Generates the morning briefing |
| **exit** | boundary | Done |

A parallel pipeline variant is also available at `pipelines/dark_factory.dot`, selectable
via `--pipeline` or the `dotfile` config field.

## Architecture

This project implements four specifications from the [Attractor](https://github.com/strongdm/attractor) project:

- **Unified LLM Client** (`src/attractor/llm/`) - Provider-agnostic LLM interface supporting Anthropic, OpenAI, and Gemini
- **Coding Agent Loop** (`src/attractor/agent/`) - Autonomous coding agent with tools (file I/O, shell, grep, glob)
- **Attractor Pipeline Engine** (`src/attractor/engine/`) - DOT-based pipeline execution with handlers, checkpointing, retry logic, and validation
- **Dark Factory Pipeline** (`src/attractor/factory/`) - The specific pipeline and handlers for the overnight software factory

## Design principles

- **No human-in-the-loop during execution.** Zero interactive prompts. Start it and walk away.
- **Quarantine over failure.** A broken work item gets shelved, not killed. The pipeline continues.
- **Artifacts over side-effects.** The factory produces patches and reports. It does not push, deploy, or modify shared infrastructure.
- **Generous limits.** The factory runs overnight. Token budgets, timeouts, and retries are set high.
