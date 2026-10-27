#!/usr/bin/env bash
# Fail if a live credential is anywhere in the tracked tree or git history.
# Runs in CI and before every push.
set -uo pipefail

# gsk_ = Groq, lsv2_pt_ = LangSmith, apikey_<hex> = TypeSafe, gh[pous]_ = GitHub
PATTERN='gsk_[A-Za-z0-9]{20,}|lsv2_(pt|sk)_[A-Za-z0-9]{20,}|apikey_[0-9a-f]{20,}|gh[pousr]_[A-Za-z0-9]{20,}'
status=0

echo "==> tracked files"
if git ls-files -z | xargs -0 grep -nEI "$PATTERN" 2>/dev/null; then
  echo "FAIL: credential found in a tracked file"; status=1
else
  echo "ok"
fi

echo "==> .env must never be tracked"
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "FAIL: .env is tracked by git"; status=1
else
  echo "ok"
fi

echo "==> git history"
if git rev-list --all 2>/dev/null | head -1 | grep -q .; then
  if git log --all -p 2>/dev/null | grep -nEI "$PATTERN" | head -5; then
    echo "FAIL: credential found in git history"; status=1
  else
    echo "ok"
  fi
else
  echo "ok (no commits yet)"
fi

[ $status -eq 0 ] && echo "PASS: no credentials detected" || echo "FAILED"
exit $status
