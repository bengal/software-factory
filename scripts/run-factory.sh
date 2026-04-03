#!/bin/sh
#
# run-factory.sh — Run the software factory in a Podman container.
#
# Usage:
#   /path/to/software-factory/scripts/run-factory.sh [--preset PRESET] [factory args...]
#
# Run from the project directory you want to process.
#
# Image resolution (first match wins):
#   1. Containerfile.factory in project root   — full custom image
#   2. --preset NAME                           — use presets/NAME.containerfile
#   3. Auto-detect from project files          — Cargo.toml→rust, pyproject.toml→python, etc.
#   4. Base image                              — factory + git + python only
#
# LLM provider credentials (passed as env vars):
#
#   Anthropic (direct):
#     ANTHROPIC_API_KEY
#
#   Claude via Vertex AI:
#     GOOGLE_APPLICATION_CREDENTIALS    — path to service account JSON key (optional if ADC configured)
#     ANTHROPIC_VERTEX_PROJECT_ID       — GCP project ID
#     CLOUD_ML_REGION                   — e.g. us-east5, europe-west1
#     If no GOOGLE_APPLICATION_CREDENTIALS is set, the script auto-mounts
#     ~/.config/gcloud/application_default_credentials.json (from gcloud auth application-default login)
#
#   OpenAI:
#     OPENAI_API_KEY
#
#   Gemini:
#     GEMINI_API_KEY

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FACTORY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(pwd)"
BASE_IMAGE="software-factory-base"
PROJECT_IMAGE="software-factory-project"

# -- Parse script-level args (before passing the rest to factory) --

PRESET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --preset=*)
            PRESET="${1#--preset=}"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# -- Build base image --

echo "Building base image '$BASE_IMAGE'..."
podman build -t "$BASE_IMAGE" -f "$FACTORY_DIR/Containerfile" "$FACTORY_DIR"

# -- Determine project Containerfile --

if [ -f "$PROJECT_DIR/Containerfile.factory" ]; then
    echo "Using project Containerfile.factory"
    CONTAINERFILE="$PROJECT_DIR/Containerfile.factory"
elif [ -n "$PRESET" ]; then
    CONTAINERFILE="$FACTORY_DIR/presets/${PRESET}.containerfile"
    if [ ! -f "$CONTAINERFILE" ]; then
        echo "Error: preset '$PRESET' not found at $CONTAINERFILE" >&2
        echo "Available presets:" >&2
        ls "$FACTORY_DIR/presets/"*.containerfile 2>/dev/null | sed 's/.*\///;s/\.containerfile//' | sed 's/^/  /' >&2
        exit 1
    fi
    echo "Using preset: $PRESET"
else
    # Auto-detect project type
    if [ -f "$PROJECT_DIR/Cargo.toml" ]; then
        PRESET="rust"
    elif [ -f "$PROJECT_DIR/pyproject.toml" ] || [ -f "$PROJECT_DIR/setup.py" ]; then
        PRESET="python"
    elif [ -f "$PROJECT_DIR/go.mod" ]; then
        PRESET="go"
    elif [ -f "$PROJECT_DIR/CMakeLists.txt" ] || [ -f "$PROJECT_DIR/meson.build" ] || [ -f "$PROJECT_DIR/Makefile" ]; then
        PRESET="c"
    fi

    if [ -n "$PRESET" ]; then
        echo "Auto-detected project type: $PRESET"
        CONTAINERFILE="$FACTORY_DIR/presets/${PRESET}.containerfile"
    else
        echo "No project type detected, using base image"
        CONTAINERFILE=""
    fi
fi

# -- Build project image --

if [ -n "$CONTAINERFILE" ]; then
    echo "Building project image '$PROJECT_IMAGE'..."
    podman build -t "$PROJECT_IMAGE" -f "$CONTAINERFILE" "$FACTORY_DIR"
    RUN_IMAGE="$PROJECT_IMAGE"
else
    RUN_IMAGE="$BASE_IMAGE"
fi

# -- Collect env vars to pass through --

ENV_ARGS=""
VOLUME_ARGS=""

# Logging
for var in LOG_LEVEL; do
    eval "val=\${$var:-}"
    if [ -n "$val" ]; then
        ENV_ARGS="$ENV_ARGS -e $var"
    fi
done

# Direct API keys
for var in ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY; do
    eval "val=\${$var:-}"
    if [ -n "$val" ]; then
        ENV_ARGS="$ENV_ARGS -e $var"
    fi
done

# Vertex AI / Google Cloud
for var in GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION CLOUDSDK_CORE_PROJECT CLOUDSDK_COMPUTE_REGION ANTHROPIC_VERTEX_PROJECT_ID CLOUD_ML_REGION CLAUDE_CODE_USE_VERTEX; do
    eval "val=\${$var:-}"
    if [ -n "$val" ]; then
        ENV_ARGS="$ENV_ARGS -e $var"
    fi
done

# Mount Google Cloud credentials
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    # Explicit service account key file
    if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        VOLUME_ARGS="$VOLUME_ARGS -v $GOOGLE_APPLICATION_CREDENTIALS:/run/secrets/gcloud-key.json:ro,Z"
        ENV_ARGS="$ENV_ARGS -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcloud-key.json"
    else
        echo "Warning: GOOGLE_APPLICATION_CREDENTIALS set but file not found: $GOOGLE_APPLICATION_CREDENTIALS" >&2
    fi
else
    # Fall back to Application Default Credentials (gcloud auth application-default login)
    ADC_FILE="${HOME}/.config/gcloud/application_default_credentials.json"
    if [ -f "$ADC_FILE" ]; then
        echo "Mounting Application Default Credentials"
        VOLUME_ARGS="$VOLUME_ARGS -v $ADC_FILE:/run/secrets/gcloud-key.json:ro,Z"
        ENV_ARGS="$ENV_ARGS -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcloud-key.json"
    fi
fi

if [ -z "$ENV_ARGS" ]; then
    echo "Warning: no API keys or credentials found in environment" >&2
fi

# -- Run --

echo "Running factory on $PROJECT_DIR..."
# shellcheck disable=SC2086
exec podman run --rm \
    --privileged \
    -v "$PROJECT_DIR:/workspace:Z" \
    $VOLUME_ARGS \
    $ENV_ARGS \
    "$RUN_IMAGE" \
    "$@"
