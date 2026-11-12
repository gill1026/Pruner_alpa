#!/bin/bash
# Resume Grounded-SAM masking (gsam_batch.py) after a Thor reboot.
# Registered as @reboot in aisl's crontab; removes its own cron line once the run prints DONE.
cd /home/aisl/gil || exit 1
PY=/home/aisl/son/alpamayo_thor/venv/bin/python

remove_cron() { crontab -l 2>/dev/null | grep -v 'gil/resume_mask.sh' | crontab -; }
is_done() { grep -aq '^DONE' logs/mask_all.log 2>/dev/null; }

echo "[$(date '+%F %T')] resume_mask start $*"
[ "$1" = "--boot" ] && sleep 120  # let GPU driver / filesystems settle

if is_done; then remove_cron; echo "already done, cron removed"; exit 0; fi

if pgrep -f 'gsam_batch.py' >/dev/null; then
  echo "already running, waiting for it"
  while pgrep -f 'gsam_batch.py' >/dev/null; do sleep 300; done
else
  [ -f logs/mask_all.log ] && mv logs/mask_all.log "logs/mask_all.$(date +%Y%m%d_%H%M%S).log"
  echo "starting gsam_batch.py"
  "$PY" -u gsam_batch.py --out /home/aisl/gil/data > logs/mask_all.log 2>&1 < /dev/null
fi

if is_done; then remove_cron; echo "[$(date '+%F %T')] DONE, cron removed"; fi
