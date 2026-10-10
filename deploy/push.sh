#!/usr/bin/env bash
# Push the WHOLE working tree to the Pi and swap it in. Run from the repo root.
#
#   bash deploy/push.sh                      # deploy HEAD to the default host
#   PI=user@host bash deploy/push.sh
#   bash deploy/push.sh --verify             # only compare live vs HEAD, change nothing
#
# This is the staging half that deploy/swap_in.sh was missing. Without it the
# only way code reached the Pi was hand-copying individual files, which is how
# ml/config/taxonomy.yaml came to hold a threshold whose inference.py was never
# deployed (A35). Deploy the tree, not the file you happened to edit.
#
# var/ is never touched in either direction -- it holds the database, published
# media and the capture spool, which belong to the deployment and not to a build.
set -euo pipefail

PI="${PI:-grilledbiscuits@192.168.1.120}"
STAGE="${STAGE:-\$HOME/birdcam-next}"   # expanded on the Pi, not here
LIVE=/opt/smartfeeder

say() { printf '[push %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

COMMIT=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain -- capture ml web deploy | head -20)

if [ "${1:-}" = --verify ]; then
  live=$(ssh -o BatchMode=yes "$PI" "cat $LIVE/DEPLOYED_COMMIT 2>/dev/null || echo UNKNOWN")
  say "live: $live"
  say "HEAD: $COMMIT"
  [ "$live" = "$COMMIT" ] && { say "MATCH"; exit 0; }
  say "MISMATCH -- do not quote field figures against this tree"
  exit 1
fi

if [ -n "$DIRTY" ]; then
  say "working tree is dirty; the stage would not match $COMMIT:"
  printf '%s\n' "$DIRTY"
  say "commit or stash first, or set ALLOW_DIRTY=1 to stage anyway"
  [ "${ALLOW_DIRTY:-}" = 1 ] || exit 1
  COMMIT="$COMMIT-dirty"
fi

say "staging $COMMIT to $PI:$STAGE"
# shellcheck disable=SC2029  # $STAGE is meant to expand on the Pi
ssh -o BatchMode=yes "$PI" "mkdir -p $STAGE"
for d in capture ml/src ml/config web deploy; do
  [ -d "$d" ] || continue
  ssh -o BatchMode=yes "$PI" "mkdir -p $STAGE/$(dirname "$d")"
  rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' \
    "$d/" "$PI:$STAGE/$d/"
done
for d in ml/data/export ml/reports; do
  [ -d "$d" ] || continue
  ssh -o BatchMode=yes "$PI" "mkdir -p $STAGE/$d"
  rsync -a "$d/" "$PI:$STAGE/$d/"
done
printf '%s\n' "$COMMIT" | ssh -o BatchMode=yes "$PI" "cat > $STAGE/DEPLOYED_COMMIT"

# capture.yaml is gitignored and per-deployment, so it is NOT in the staged tree.
# swap_in.sh syncs capture/ with --delete, which would remove the live config,
# fail --check and roll back on every run. Carry the live copy into the stage.
say "carrying the live capture.yaml into the stage (gitignored, per-deployment)"
ssh -o BatchMode=yes "$PI" "
  set -e
  if [ -f $LIVE/capture/config/capture.yaml ]; then
    mkdir -p $STAGE/capture/config
    sudo -n cat $LIVE/capture/config/capture.yaml > $STAGE/capture/config/capture.yaml
    echo '  carried the live capture.yaml into the stage'
  else
    echo '  NO live capture.yaml -- first deploy? copy capture.example.yaml and edit it'
    exit 1
  fi"

say "swapping in on the Pi (runs under nohup; swap_in.sh rolls back on failure)"
ssh -o BatchMode=yes "$PI" \
  "cd $STAGE && nohup bash deploy/swap_in.sh $STAGE > /tmp/swap_in.log 2>&1; tail -40 /tmp/swap_in.log"

say "verify with:  bash deploy/push.sh --verify"
