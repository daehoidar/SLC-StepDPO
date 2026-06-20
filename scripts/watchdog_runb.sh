#!/usr/bin/env bash
set -uo pipefail
D="$HOME/_teammate_repo"; cd "$D" || exit 0
LOG="$D/logs/watchdog_runb.log"; ST="$D/logs/.wd_runb.state"; MAX=8
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_m(){ grep -ql "run-b done" "$D"/logs/run_b_*.out 2>/dev/null \
  || [ -f "$D/docs/figures_final/fig_results_table_b_n200.png" ]; }
off(){ crontab -l 2>/dev/null | grep -v watchdog_runb | crontab - 2>/dev/null; log "off"; }
Q="$(squeue -u "$USER" -n run-b -h -o '%T' 2>/dev/null)"
[ -n "$Q" ] && { log "OK($Q)"; exit 0; }
done_m && { log "SUCCESS"; off; exit 0; }
R=0; [ -f "$ST" ] && R=$(grep -oE '[0-9]+' "$ST"|head -1||echo 0); R=${R:-0}
[ "$R" -ge "$MAX" ] && { log "STOP max"; off; exit 0; }
NID=$(sbatch --parsable "$D/scripts/run_b_campaign_slurm.sh" 2>>"$LOG"); R=$((R+1)); echo "RETRIES=$R">"$ST"
log "RESUBMIT #$R: $NID"
