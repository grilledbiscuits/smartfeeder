#!/usr/bin/env bash
# Swap a staged build into /opt/smartfeeder, validate it, and roll back on failure.
#
#   bash deploy/swap_in.sh ~/birdcam-next
#
# Run ON THE PI. Order is the design:
#
#   1. back up the live code, model, reports and config (never var/, never .venv)
#   2. stop the service
#   3. copy the staged build in, keeping root ownership on code the service
#      only needs to READ
#   4. run `python -m capture --check` AS THE SERVICE USER before starting
#   5. start, then watch the log; any ERROR in the first window rolls back
#
# `--check` builds every component -- config cross-checks, ONNX session, the
# graph/calibration pairing guard -- without touching the camera or GPIO. A
# build that fails there would fail on start, and it is far better found before
# the service is down than after.
#
# var/ is deliberately untouched in both directions: it holds the dashboard
# database, published media and the capture spool, which belong to the running
# deployment, not to a build.

set -euo pipefail

STAGE="${1:?usage: swap_in.sh STAGED_BUILD_DIR}"
LIVE=/opt/smartfeeder
SERVICE=birdcam-capture
BACKUP="$HOME/birdcam-backup-$(date +%Y%m%d_%H%M%S)"
PY="$LIVE/.venv/bin/python"
WATCH_SECONDS="${WATCH_SECONDS:-45}"

say() { printf '[swap_in %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

sync_code() {  # src dst -- code the service reads; root-owned, world-readable
  sudo -n rsync -a --delete --chown=root:root --chmod=D755,F644 \
    --exclude='__pycache__/' --exclude='*.pyc' "$1/" "$2/"
}

rollback() {
  say "ROLLING BACK from $BACKUP"
  sudo -n /usr/bin/systemctl stop "$SERVICE" || true
  for d in capture ml/src ml/config web deploy; do
    [ -d "$BACKUP/$d" ] && sync_code "$BACKUP/$d" "$LIVE/$d"
  done
  rsync -a "$BACKUP/ml/data/export/" "$LIVE/ml/data/export/"
  rsync -a "$BACKUP/ml/reports/" "$LIVE/ml/reports/"
  sudo -n /usr/bin/systemctl start "$SERVICE"
  say "rolled back; service is $(systemctl is-active "$SERVICE")"
}

[ -d "$STAGE/capture" ] || { say "no staged build at $STAGE"; exit 1; }
sudo -n true || { say "passwordless sudo is required"; exit 1; }

# The stage must say which commit it is. Without this the live tree cannot be
# compared to anything, and on 2026-10-10 that cost a whole run: a threshold was
# added to ml/config/taxonomy.yaml while the inference.py that reads it stayed
# behind, so the knob was a silent no-op all day and a clip published that it
# would have held back. A missing config KEY raises CaptureConfigError; a key
# whose code is missing fails silently and reads as a tuning result. See A35.
[ -f "$STAGE/DEPLOYED_COMMIT" ] || {
  say "$STAGE has no DEPLOYED_COMMIT -- stage it with deploy/push.sh, not by hand"
  exit 1
}
STAGE_COMMIT=$(cat "$STAGE/DEPLOYED_COMMIT")
say "staging commit: $STAGE_COMMIT"
[ -f "$LIVE/DEPLOYED_COMMIT" ] && say "live commit:    $(cat "$LIVE/DEPLOYED_COMMIT")"

# Config the Pi owns deliberately. swap_in.sh replaces ml/config wholesale, so
# any local edit there is reverted -- that has happened before, reverting the
# Zosterops virens Tier A promotion. Warn loudly; the backup below keeps them.
for f in ml/config/species.yaml ml/config/taxonomy.yaml; do
  if [ -f "$LIVE/$f" ] && [ -f "$STAGE/$f" ] && ! diff -q "$LIVE/$f" "$STAGE/$f" >/dev/null; then
    say "WARNING: $f differs from the stage and WILL be replaced"
    say "         live copy is kept in the backup; re-apply after the swap if intended"
  fi
done

say "backing up live build to $BACKUP"
mkdir -p "$BACKUP/ml/data" "$BACKUP/ml"
for d in capture web deploy; do rsync -a "$LIVE/$d" "$BACKUP/"; done
rsync -a "$LIVE/ml/src" "$LIVE/ml/config" "$LIVE/ml/reports" "$BACKUP/ml/"
rsync -a "$LIVE/ml/data/export" "$BACKUP/ml/data/"

say "stopping $SERVICE"
sudo -n /usr/bin/systemctl stop "$SERVICE"
trap 'say "failed mid-swap"; rollback; exit 1' ERR

say "installing staged build"
for d in capture ml/src ml/config web deploy; do
  [ -d "$STAGE/$d" ] && sync_code "$STAGE/$d" "$LIVE/$d"
done
rsync -a "$STAGE/ml/data/export/" "$LIVE/ml/data/export/"
rsync -a "$STAGE/ml/reports/" "$LIVE/ml/reports/"
printf '%s\n' "$STAGE_COMMIT" | sudo -n tee "$LIVE/DEPLOYED_COMMIT" >/dev/null

say "validating as the service user"
if ! (cd "$LIVE" && sudo -n -u birdcam "$PY" -m capture --check \
        --config "$LIVE/capture/config/capture.yaml"); then
  trap - ERR
  say "--check FAILED"
  rollback
  exit 1
fi

say "starting $SERVICE"
sudo -n /usr/bin/systemctl start "$SERVICE"
trap - ERR

say "watching the log for ${WATCH_SECONDS}s"
MARK=$(date '+%Y-%m-%d %H:%M:%S')
sleep "$WATCH_SECONDS"
if [ "$(systemctl is-active "$SERVICE")" != active ]; then
  say "service is not active after start"
  rollback
  exit 1
fi
if awk -v m="$MARK" '$1" "$2 >= m' "$LIVE/var/capture/capture.log" 2>/dev/null \
     | grep -E ' (ERROR|CRITICAL) ' ; then
  say "errors logged after start"
  rollback
  exit 1
fi

say "deployed $STAGE_COMMIT. backup kept at $BACKUP"
say "to roll back by hand:  bash $LIVE/deploy/rollback.sh $BACKUP"
