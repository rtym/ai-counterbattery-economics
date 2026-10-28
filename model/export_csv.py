"""results.json -> outputs/scenarios.csv, robustness.csv, prcc.csv (supplementary tables)."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

METRICS = ["days", "spend_total_musd", "spend_per_day_musd", "loss_B_musd", "loss_R_musd", "cost_exchange",
           "spend_per_red_gun_destroyed_kusd", "destroyed_value_per_day_musd", "burn_per_day_B_musd", "n_star_B",
           "cost_per_kill_A_kusd", "p_kill_B", "p_time_B", "psi_B", "drone_share_B", "fire_ratio"]
with open(OUT / "scenarios.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "label", "posture", "metric", "p05", "median", "p95"])
    for sid, s in R["scenarios"].items():
        for m in METRICS:
            w.writerow([sid, s["label"], s["mode"], m, *s[m][:1], s[m][1], s[m][2]])
        for m in ("p_blue_wins", "p_red_wins", "p_undecided", "ammo_bound_B", "mean_cost_battle_musd",
                  "p_superiority_in_window"):
            if m in s:
                w.writerow([sid, s["label"], s["mode"], m, "", s[m], ""])
with open(OUT / "robustness.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["variant", "scenario", "days_median", "p_blue_holds", "spend_per_red_gun_kusd_median", "ammo_bound_B"])
    for v, row in R["robustness"].items():
        for sid, x in row.items():
            w.writerow([v, sid, x["days"], x["p_blue_wins"], x["spend_per_red_gun_kusd"], x["ammo_bound_B"]])
with open(OUT / "prcc.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "output", "input", "prcc"])
    for sid, s in R["scenarios"].items():
        for out in ("prcc_days", "prcc_cost_per_kill", "prcc_cer"):
            for k, v in s.get(out, {}).items():
                w.writerow([sid, out, k, v])
print("written:", ", ".join(p.name for p in sorted(OUT.glob("*.csv"))))
