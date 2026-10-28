"""Figures for the article on AI-enabled Missile Forces and Artillery and the economics of the fires battle.

All numbers come from research/model (model.py -> results.json, draws_*.npz); module A curves are
recomputed here from the modal parameters in research/model/params.py. Figures are grey-scale with one
accent, 300 dpi: fig*.png for the manuscript, fig*.tif for the separate submission files.
Run from the repository root after model/model.py: python make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
MOD = HERE / "model"
sys.path.insert(0, str(MOD))
import model as M  # noqa: E402
import params as P  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 9})
DARK, MID, LIGHT, PALE, ACC = "#1f1f1f", "#5f5f5f", "#9e9e9e", "#d9d9d9", "#8c2d04"
RES = json.loads((MOD / "results.json").read_text())
SC_ALL = RES["scenarios"]
SC = {k: v for k, v in SC_ALL.items() if k.startswith("S")}          # defence; offence: OFF
OFF = {k: v for k, v in SC_ALL.items() if k.startswith("O")}
SHORT = {"S0": "S0\n2022", "S1": "S1\n2025", "S2": "S2\n2033", "S3": "S3\n2036",
         "S4": "S4\n2036", "S5": "S5\n2036"}


def plain_log(ax, axis="y", ticks=None):
    """Log axis with plain-number tick labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.tif", dpi=(300, 300), compression="tiff_lzw")
    plt.close(fig)


def mode(era, key):
    return P.ERA[key][era][0][1]


def modal_side(era):
    return {k: np.array([v[era][0][1]], float) for k, v in P.ERA.items()}


# --- Figure 1: model structure ---------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(6.7, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=DARK, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold")
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.2, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=DARK, lw=0.8))

    box(1, 30, 28, 20, "Module C: trajectories",
        "7 AI functions (Table 1)\nadoption $\\varphi(t)$, logistic\nadversary erosion $\\varepsilon$\nWright's law for unit costs\n"
        "2025 $\\rightarrow$ 2033 / 2036", PALE)
    box(36, 30, 28, 20, "Module A: engagement",
        "$P(n)=e^{-(T+t_f)/\\tau}\\,\\psi(n)$\nshared aim bias $\\sigma_{TLE}$, $\\sigma_d$, $R_e$\nrounds every $\\Delta$, decoys\n"
        "$n^*=\\arg\\min\\,C(n)$\ncost per neutralised target", "#ececec")
    box(71, 30, 28, 20, "Module B: duel",
        "defence 45 vs 90, offence 90 vs 45\ndetection, decoys, ammunition,\nstrike drones, repair, replacement\n"
        "daily steps, 4,000 draws\nculmination at 50 %", PALE)
    box(13, 3, 34, 19, "Field data (2022, 2025)",
        "RUSI field studies: kill-chain times,\nrounds per day, UAV attrition,\nFPV failure rate, decoys and dug-in guns;\n"
        "prices: industry reports; FPV cost", "white")
    box(53, 3, 34, 19, "Outputs",
        "duration to culmination, outcome\nBlue expenditure by category\nspend per Red gun destroyed\n"
        "cost-exchange ratio; PRCC ranks", "white")
    arrow(29.5, 40, 35.5, 40)
    arrow(64.5, 40, 70.5, 40)
    arrow(30, 22.5, 45, 29.5)
    arrow(30, 22.5, 80, 29.5)
    arrow(85, 29.5, 72, 22.5)
    save(fig, "fig1")


# --- Figure 2: kill chain economics (module A) ------------------------------------------------------------
def fig2():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.7))
    T = np.linspace(0.1, 40, 300)
    for tau, ls in ((10, ":"), (30, "--"), (120, "-")):
        a1.plot(T, np.exp(-T / tau), color=DARK, ls=ls, lw=1, label=f"dwell $\\tau$ = {tau} min")
    for lo, hi, lab, x, y in ((20, 30, "HQ-routed\n(2022)", 25, 0.60), (3, 5, "organic\nUAV", 6.0, 0.03),
                              (0.5, 2, "AI", 1.25, 0.03)):
        a1.axvspan(lo, hi, color=PALE, lw=0)
        a1.text(x, y, lab, ha="left" if lab == "organic\nUAV" else "center", va="bottom", fontsize=6.8, color=MID,
                bbox=dict(fc="white", ec="none", pad=0.4) if y > 0.1 else None)
    a1.set_xlabel("detect-to-effect latency $T$, min")
    a1.set_ylabel("$e^{-T/\\tau}$: target still in place")
    a1.set_ylim(0, 1.02)
    a1.legend(frameon=False, fontsize=7, loc="upper right")
    a1.set_title("(a) latency against target dwell", loc="left")

    R_e = np.array([P.COMMON["R_e"][0][1]])
    for era, tgt, price, ls, lab in (("2022", "2022", P.C_ROUND["2022"][0][1], ":", "2022"),
                                     ("2025", "2025", P.C_ROUND["2025"][0][1], "--", "2025"),
                                     ("ai", "ai", SC["S3"]["price_round_usd"][1], "-", "2036 AI")):
        sh, tg = modal_side(era), modal_side(tgt)
        e = M.engagement(sh, tg, np.array([price]), R_e, 1.0, t_flight=P.COMMON["t_flight"][0][1],
                         interval=1 / (P.COMMON["r_fire"][0][1] * P.GUNS_PER_MISSION["B"][0][1]))
        c = e["cost_curve"][0] / 1e3
        a2.plot(M.N_GRID, c, color=DARK if era == "ai" else MID, ls=ls, lw=1.1, label=lab)
        k = int(np.argmin(c))
        a2.plot(M.N_GRID[k], c[k], "o", color=ACC, ms=3.5)
        a2.annotate(f"$n^*$ = {M.N_GRID[k]}", (M.N_GRID[k], c[k]), xytext=(3, 5), textcoords="offset points", fontsize=7)
    a2.set_yscale("log")
    plain_log(a2, ticks=[100, 200, 500, 1000, 2000, 5000])
    a2.set_ylim(120, 8000)
    a2.set_xlabel("rounds per fire mission $n$")
    a2.set_ylabel("cost per neutralised gun, thousand USD")
    a2.set_xlim(0, 60)
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    a2.set_title("(b) cost per neutralised gun (modal values)", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig2")


# --- Figure 3: trajectories (module C) ---------------------------------------------------------------------
def fig3():
    rng = np.random.default_rng(7)
    n = 4000
    years = np.arange(2025, 2036.01, 0.25)
    tmid = M.lhs_triangular(P.ADOPT["t_mid"][0], n, rng)
    k = M.lhs_triangular(P.ADOPT["k"][0], n, rng)
    ph = np.array([M.phi(y, tmid, k) for y in years])
    eps = M.lhs_triangular(P.EROSION["contested"][0], n, rng)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.6))
    for arr, col, lab in ((ph, DARK, "uncontested"), (ph * (1 - eps), MID, "contested EW ($\\varepsilon$ = 0.3-0.6)")):
        lo, me, hi = np.quantile(arr, [0.05, 0.5, 0.95], axis=1)
        a1.fill_between(years, lo, hi, color=col, alpha=0.18, lw=0)
        a1.plot(years, me, color=col, lw=1.1, label=lab)
    for y in (2033, 2036):
        a1.axvline(y, color=LIGHT, lw=0.7, ls="--")
        a1.text(y, 0.02, str(y), fontsize=7, ha="center", color=MID, bbox=dict(fc="white", ec="none", pad=0.5))
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("year")
    a1.set_ylabel("share of the AI gain realised, $\\varphi(t)$")
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    a1.set_title("(a) adoption of the AI kill web", loc="left")
    for item, col, ls, lab in (("round", DARK, "-", "155-mm round"),
                               ("fuzed", ACC, "-", "round with fuze, adopted share"),
                               ("fpv", MID, "--", "strike drone (incl. AI premium)"),
                               ("sortie", LIGHT, "-.", "ISR sortie")):
        vals = []
        for iy, y in enumerate(years):
            r = np.random.default_rng(11)
            pr = M.prices("ai", y, n, r)
            pr["fuzed"] = pr["c_round"] + M.fuze_price(y, n, np.random.default_rng(12)) * ph[iy]
            base = {"round": M.lhs_triangular(P.C_ROUND["2025"][0], n, np.random.default_rng(11)),
                    "fpv": M.lhs_triangular(P.C_FPV["2025"][0], n, np.random.default_rng(11))}
            v = {"round": pr["c_round"] / np.median(base["round"]), "fuzed": pr["fuzed"] / np.median(base["round"]),
                 "fpv": pr["c_fpv"] / np.median(base["fpv"]),
                 "sortie": pr["sortie_mult"]}[item]
            vals.append(np.quantile(v, [0.05, 0.5, 0.95]))
        vals = np.array(vals)
        a2.fill_between(years, vals[:, 0], vals[:, 2], color=col, alpha=0.15, lw=0)
        a2.plot(years, vals[:, 1], color=col, ls=ls, lw=1.1, label=lab)
    a2.axhline(1, color=LIGHT, lw=0.6)
    a2.set_xlabel("year")
    a2.set_ylabel("unit cost, index (2025 median = 1)")
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    a2.set_title("(b) unit costs, Wright's law", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig3")


# --- Figure 4: Monte Carlo distributions -------------------------------------------------------------------
def fig4():
    ids = list(SC)
    d = {i: np.load(MOD / f"draws_{i}.npz") for i in ids}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.9))
    pos = np.arange(len(ids))
    style = dict(widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=ACC, lw=1.3),
                 whiskerprops=dict(color=DARK, lw=0.7), capprops=dict(color=DARK, lw=0.7))
    bp = a1.boxplot([d[i]["days"] for i in ids], positions=pos, whis=(5, 95), **style)
    for b in bp["boxes"]:
        b.set(facecolor=PALE, edgecolor=DARK, lw=0.7)
    a1.set_yscale("log")
    plain_log(a1, ticks=[1, 2, 5, 10, 20, 50, 100, 180])
    a1.set_ylim(1, 400)
    a1.set_ylabel("days to culmination of one side")
    a1.set_xticks(pos, [SHORT[i] for i in ids], fontsize=6.8)
    a1.set_title("(a) duration of the fires battle", loc="left")
    for x, i in zip(pos, ids):
        a1.text(x, 230, f"{SC[i]['p_blue_wins']:.0%}", ha="center", fontsize=6.8, color=ACC)
    a1.text(len(ids) - 0.6, 320, "share of draws Blue holds", fontsize=6.5, color=ACC, ha="right")
    per = [d[i]["spend"] / np.maximum(d[i]["killsR"], 1) / 1e6 for i in ids]
    bp = a2.boxplot(per, positions=pos, whis=(5, 95), **style)
    for b in bp["boxes"]:
        b.set(facecolor=PALE, edgecolor=DARK, lw=0.7)
    a2.set_yscale("log")
    plain_log(a2, ticks=[0.1, 0.2, 0.5, 1, 2, 5, 10])
    a2.axhspan(P.COMMON["c_gun_R"][0][0] / 1e6, P.COMMON["c_gun_R"][0][2] / 1e6, color=ACC, alpha=0.08, lw=0)
    a2.text(len(ids) - 0.5, P.COMMON["c_gun_R"][0][2] / 1e6 * 1.05, "value of a Red gun", fontsize=6.8,
            ha="right", va="bottom", color=ACC)
    a2.set_ylabel("Blue spend per Red gun destroyed, million USD")
    a2.set_xticks(pos, [SHORT[i] for i in ids], fontsize=6.8)
    a2.set_title("(b) efficiency of the reconnaissance-fire system", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig4")


# --- Figure 5: structure of costs and the intensity paradox -----------------------------------------------
def fig5():
    ids = list(SC)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.9))
    cats = [("rounds", DARK, "rounds"), ("fpv", MID, "strike drones"), ("isr", LIGHT, "ISR sorties"),
            ("c2", PALE, "AI / C2 running cost")]
    pos = np.arange(len(ids))
    bottom = np.zeros(len(ids))
    for k, col, lab in cats:
        v = np.array([np.mean(np.load(MOD / f"draws_{i}.npz")[f"spend_{k}"]) / 1e6 for i in ids])
        a1.bar(pos, v, 0.6, bottom=bottom, color=col, edgecolor=DARK, lw=0.4, label=lab)
        bottom += v
    loss = np.array([np.mean(np.load(MOD / f"draws_{i}.npz")["lossB"]) / 1e6 for i in ids])
    a1.bar(pos, loss, 0.6, bottom=bottom, color="white", edgecolor=ACC, lw=0.8, hatch="////", label="own guns lost")
    a1.set_xticks(pos, [SHORT[i] for i in ids], fontsize=6.8)
    a1.set_ylabel("Blue cost per fires battle, million USD (mean)")
    a1.legend(frameon=False, fontsize=6.8, loc="upper right")
    a1.set_title("(a) what a fires battle costs Blue", loc="left")
    for i, s in SC.items():
        x = SC[i]["spend_per_red_gun_destroyed_kusd"][1] / 1e3
        y = SC[i]["destroyed_value_per_day_musd"][1]
        lo, hi = SC[i]["destroyed_value_per_day_musd"][0], SC[i]["destroyed_value_per_day_musd"][2]
        xl, xh = SC[i]["spend_per_red_gun_destroyed_kusd"][0] / 1e3, SC[i]["spend_per_red_gun_destroyed_kusd"][2] / 1e3
        a2.errorbar(x, y, yerr=[[y - lo], [hi - y]], xerr=[[x - xl], [xh - x]], fmt="o", ms=3.5,
                    color=ACC if s["blue"] == "ai" else DARK, ecolor=LIGHT, elinewidth=0.7, capsize=1.5)
        off = {"S2": (6, -11), "S3": (-2, 8), "S4": (-16, -12), "S5": (5, -9), "S0": (5, -10), "S1": (5, 4)}[i]
        a2.annotate(i, (x, y), xytext=off, textcoords="offset points", fontsize=7)
    a2.set_xscale("log")
    a2.set_yscale("log")
    plain_log(a2, "x", [0.1, 0.2, 0.5, 1, 2, 5, 10])
    plain_log(a2, "y", [2, 5, 10, 20, 50])
    a2.set_xlabel("Blue spend per Red gun destroyed, million USD")
    a2.set_ylabel("materiel destroyed per day, both sides, M USD")
    a2.set_title("(b) cheaper kills, costlier days", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig5")


# --- Figure 6: sensitivity (PRCC) ---------------------------------------------------------------------------
NICE = {"B_T": "Blue latency $T$", "B_s_tle": "Blue location error", "B_s_disp": "Blue round dispersion",
        "B_p_real": "Blue decoy discrimination", "B_sorties": "Blue ISR sorties", "B_p_find": "Blue find rate",
        "B_fpv": "Blue drone cueing share", "B_p_reach": "Blue drone reach rate", "B_c_c2": "Blue AI/C2 cost",
        "B_w": "Blue AI adoption $\\varphi$", "R_T": "Red latency", "R_tau": "Red dwell time",
        "R_conceal": "Red concealment (higher = easier to find)", "R_protect": "Red protection (higher = weaker)",
        "B_tau": "Blue dwell time", "B_conceal": "Blue concealment (higher = easier)", "c_R_e": "lethal radius $R_e$",
        "c_k_fire": "exposure from firing", "c_m_max": "missions per gun", "c_r_rep": "replacement rate",
        "price_round": "price of a round", "price_fpv": "price of a strike drone", "ammo_B": "Blue rounds per gun",
        "ammo_R": "Red rounds per gun", "c_t_flight": "flight time", "c_r_fire": "rate of fire",
        "gpm_B": "Blue guns per mission", "gpm_R": "Red guns per mission"}


def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.2), sharey=False)
    for ax, sid, title in ((axes[0], "S1", "(a) 2025 drone-enabled"), (axes[1], "S3", "(b) 2036 symmetric AI")):
        pr = SC[sid]["prcc_cost_per_kill"]
        top = sorted(pr.items(), key=lambda kv: -abs(kv[1]))[:9][::-1]
        y = np.arange(len(top))
        ax.barh(y, [v for _, v in top], color=[DARK if v > 0 else LIGHT for _, v in top], edgecolor=DARK, lw=0.4)
        ax.set_yticks(y, [NICE[k] for k, _ in top], fontsize=7)
        ax.axvline(0, color=DARK, lw=0.6)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("PRCC with spend per Red gun destroyed")
        ax.set_title(title, loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6")


# --- Figure 7 (Figure 6 in the article): offence ------------------------------------------------------------
OSHORT = {"O1": "O1\n2025", "O2": "O2\n2036 sym.", "O3": "O3\n2036 att. AI", "O4": "O4\ncontested"}


def fig7():
    ids = list(OFF)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 2.8))
    pos = np.arange(len(ids))
    sup = [100 * OFF[i]["p_superiority_in_window"] for i in ids]
    a1.bar(pos, sup, 0.6, color=[DARK if OFF[i]["blue"] == "ai" and OFF[i]["red"] != "ai" else LIGHT for i in ids],
           edgecolor=DARK, lw=0.5)
    for x, v in zip(pos, sup):
        a1.text(x, v + 2, f"{v:.0f} %", ha="center", fontsize=7)
    a1.axhline(50, color=ACC, lw=0.7, ls="--")
    a1.set_ylim(0, 110)
    a1.set_xticks(pos, [OSHORT[i] for i in ids], fontsize=6.8)
    a1.set_ylabel("fire superiority within 30 days, % of draws")
    a1.set_title("(a) does the attacker win the fire fight?", loc="left")
    att = [np.mean(np.load(MOD / f"draws_{i}.npz")["lossB"]) / 1e6 for i in ids]
    dfd = [np.mean(np.load(MOD / f"draws_{i}.npz")["lossR"]) / 1e6 for i in ids]
    w = 0.38
    a2.bar(pos - w / 2, att, w, color="white", edgecolor=ACC, hatch="////", lw=0.8, label="attacker's guns lost")
    a2.bar(pos + w / 2, dfd, w, color=MID, edgecolor=DARK, lw=0.5, label="defender's guns lost")
    a2.set_xticks(pos, [OSHORT[i] for i in ids], fontsize=6.8)
    a2.set_ylabel("value of guns lost, million USD (mean)")
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    a2.set_title("(b) and at what price", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig7")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print("ok", f.__name__)
