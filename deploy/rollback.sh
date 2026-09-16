#!/usr/bin/env bash
# Restore a backup taken by swap_in.sh.   bash deploy/rollback.sh ~/birdcam-backup-YYYYMMDD_HHMMSS
set -euo pipefail
BACKUP="${1:?usage: rollback.sh BACKUP_DIR}"; LIVE=/opt/smartfeeder; SERVICE=birdcam-capture
[ -d "$BACKUP/capture" ] || { echo "no backup at $BACKUP"; exit 1; }
sudo -n /usr/bin/systemctl stop "$SERVICE" || true
for d in capture ml/src ml/config web deploy; do
  [ -d "$BACKUP/$d" ] && sudo -n rsync -a --delete --chown=root:root --chmod=D755,F644 \
    --exclude='__pycache__/' "$BACKUP/$d/" "$LIVE/$d/"
done
rsync -a "$BACKUP/ml/data/export/" "$LIVE/ml/data/export/"
rsync -a "$BACKUP/ml/reports/" "$LIVE/ml/reports/"
sudo -n /usr/bin/systemctl start "$SERVICE"
echo "restored $BACKUP; $SERVICE is $(systemctl is-active "$SERVICE")"
