#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  cat <<'EOF'
No virtualenv at ./venv.

Once (Python 3.10–3.13, not 3.14+):

  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

Then run ./photoscanner.sh again.
EOF
  exit 1
fi

exec "$PYTHON" "$ROOT/src/photoscanner.py" "$@"
