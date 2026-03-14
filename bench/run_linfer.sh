#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-./models/model.lnf}
TOKENS=${2:-200}

cargo run -p linfer-cli -- bench "$MODEL_PATH" --tokens "$TOKENS"
