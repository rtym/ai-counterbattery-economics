"""Kill-chain economics of Missile Forces and Artillery in a notional conflict 7-10 years ahead.

Three coupled modules (parameters and their provenance: params.py):

A. Engagement. For a shooter with detect-to-effect latency T, target-location error s_tle (common to all
   rounds of a mission), ballistic dispersion s_disp and a Carleton damage function with effective radius
   R_e, the chance that a mission of n rounds neutralises a real target that dwells exp(tau) minutes is
       P(n) = exp(-T/tau) * psi(n),   psi(n) = E_b[1 - (1 - h(b))^n],
       h(b) = R_e^2 / (R_e^2 + 2 s_disp^2) * exp(-b^2 / (R_e^2 + 2 s_disp^2)),
   where the aim bias b is circular normal with s_tle per axis. With u = b^2 / (2 s_tle^2) ~ Exp(1),
   psi is computed by Gauss-Laguerre quadrature. Cost per neutralised target for a mission of n rounds:
       C(n) = (c_sortie / p_find + n * c_round) / (p_real * P(n)),
   and the shooter fires the n* that minimises C(n). Because the bias is shared by every round of the
   mission, extra rounds have sharply diminishing returns when s_tle is large.

B. Fires duel. Two reconnaissance-fire systems (Blue, Red) in a corps sector, daily steps up to 180 days,
   vectorised over Monte Carlo draws. Each day each side detects enemy guns (hazard grows with the
   enemy's firing activity), engages up to its gun and ammunition limits (decoys absorb a share of
   missions), and adds an FPV layer; hits destroy or damage guns, damaged guns return after repair.
   A side culminates when fewer than theta of its fire units are operational. Outputs: duration,
   outcome, Blue expenditure by category, value of Red guns destroyed.

C. Trajectories 2025 -> 2036. Logistic adoption of the AI kill web phi(t) moves each performance
   parameter from its 2025 value toward the mature-AI value (geometric interpolation for scales, linear
   for probabilities), reduced by adversary erosion epsilon. Unit costs follow Wright's law; AI-era rounds carry
   a course-correcting fuze on the adopted share phi.

Run: python model.py  -> results.json, draws_*.npz next to this file.
"""
import json
from pathlib import Path

import numpy as np

import params as P

HERE = Path(__file__).resolve().parent
N_DRAWS = 4000
SEED = 20261025
LAG_X, LAG_W = np.polynomial.laguerre.laggauss(40)
N_GRID = np.arange(1, 61)
LOG_PARAMS = {"T", "tau", "s_tle", "s_disp", "sorties", "c_sortie", "c_c2", "conceal"}


# --- sampling --------------------------------------------------------------------------------------------
def lhs_triangular(spec, n, rng):
    """Latin-hypercube sample of a triangular (low, mode, high) distribution."""
    lo, mo, hi = spec
    u = (rng.permutation(n) + rng.random(n)) / n
    if hi == lo:
        return np.full(n, float(lo))
    c = (mo - lo) / (hi - lo)
    return np.where(u < c, lo + np.sqrt(u * (hi - lo) * (mo - lo)), hi - np.sqrt((1 - u) * (hi - lo) * (hi - mo)))


def sample_era(era, n, rng):
    return {k: lhs_triangular(v[era][0], n, rng) for k, v in P.ERA.items()}


def blend(a, b, w, log):
    """Move from era value a toward AI value b by weight w in [0, 1]."""
    if log:
        return np.exp(np.log(np.maximum(a, 1e-9)) * (1 - w) + np.log(np.maximum(b, 1e-9)) * w)
    return a * (1 - w) + b * w


def phi(year, t_mid, k):
    return 1.0 / (1.0 + np.exp(-k * (year - t_mid)))


def side_params(kind, year, n, rng, eps_kind="none"):
    """Parameter draws for one side. kind: '2022' | '2025' | 'ai' (blend toward AI by the year)."""
    if kind in ("2022", "2025"):
        s = sample_era(kind, n, rng)
        s["w"] = np.zeros(n)
    else:
        base, ai = sample_era("2025", n, rng), sample_era("ai", n, rng)
        w = phi(year, lhs_triangular(P.ADOPT["t_mid"][0], n, rng), lhs_triangular(P.ADOPT["k"][0], n, rng))
        adopt = w
        w = w * (1 - lhs_triangular(P.EROSION[eps_kind][0], n, rng))
        s = {k: blend(base[k], ai[k], w, k in LOG_PARAMS) for k in base}
        s["w"], s["adopt"] = w, adopt
    return s


def prices(kind, year, n, rng):
    """Unit prices (USD) for rounds, FPV and the ISR sortie multiplier for an era or a forecast year."""
    if kind == "2022":
        return {"c_round": lhs_triangular(P.C_ROUND["2022"][0], n, rng),
                "c_fpv": lhs_triangular(P.C_FPV["2025"][0], n, rng), "sortie_mult": np.ones(n)}
    r25 = lhs_triangular(P.C_ROUND["2025"][0], n, rng)
    f25 = lhs_triangular(P.C_FPV["2025"][0], n, rng)
    if kind == "2025":
        return {"c_round": r25, "c_fpv": f25, "sortie_mult": np.ones(n)}
    frac = (year - 2025) / 11.0                                    # share of the 2025-2036 doublings realised
    L = P.LEARN

    def wright(item):
        lr = lhs_triangular(L[item]["lr"][0], n, rng)
        d = lhs_triangular(L[item]["doublings"][0], n, rng) * frac
        return (1 - lr) ** d

    relief = lhs_triangular(L["round"]["scarcity_relief"][0], n, rng)
    c_round = r25 * (1 - (1 - relief) * min(1.0, (year - 2025) / 8)) * wright("round")
    prem = lhs_triangular(L["fpv"]["ai_premium"][0], n, rng)
    c_fpv = f25 * prem * wright("fpv")
    return {"c_round": c_round, "c_fpv": c_fpv, "sortie_mult": wright("sortie")}


def fuze_price(year, n, rng):
    """Course-correcting fuze, USD per round, in a forecast year (Wright's law over the 2025-2036 doublings)."""
    F = P.FUZE
    lr = lhs_triangular(F["lr"][0], n, rng)
    d = lhs_triangular(F["doublings"][0], n, rng) * (year - 2025) / 11.0
    return lhs_triangular(F["price"][0], n, rng) * (1 - lr) ** d


def common(n, rng):
    return {k: lhs_triangular(v[0], n, rng) for k, v in P.COMMON.items()}


def ammo(kind, side, n, rng):
    era = kind if kind in ("2022", "2025") else "ai"
    return lhs_triangular(P.AMMO[(era, side)][0], n, rng)


# --- module A ----------------------------------------------------------------------------------------------
def psi(n_rounds, s_tle, s_disp, R_e, delta_tau=0.0):
    """P(the mission neutralises a real target that was still in place when the first round landed).

    Rounds land every Delta minutes; the target leaves after an exponential time with mean tau, so round j still
    finds it with probability exp(-(j-1) Delta / tau). With the aim bias shared by the mission,
        P(n | b) = h(b) * (1 - rho^n) / (1 - rho),   rho = (1 - h(b)) exp(-Delta / tau),
    averaged over the bias by Gauss-Laguerre quadrature (delta_tau = Delta / tau; 0 gives 1 - (1 - h)^n).
    Arrays broadcast: s_tle, s_disp, R_e, delta_tau shape (N,), n_rounds shape (K,) -> (N, K)."""
    s_tle, s_disp, R_e, dt = (np.broadcast_to(np.asarray(x, float), np.shape(s_tle))[..., None, None]
                              for x in (s_tle, s_disp, R_e, delta_tau))
    denom = R_e ** 2 + 2 * s_disp ** 2
    h = (R_e ** 2 / denom) * np.exp(-(2 * s_tle ** 2 * LAG_X[None, None, :]) / denom)   # (N,1,Q)
    rho = (1 - h) * np.exp(-dt)
    nn = np.asarray(n_rounds, float)[None, :, None]
    return np.sum(LAG_W[None, None, :] * h * (1 - rho ** nn) / (1 - rho), axis=-1)


def engagement(sh, tg, cr, R_e, c_sortie_mult=1.0, t_flight=0.0, interval=0.0):
    """Cost per neutralised target over the n grid, optimal n*, P(n*). Shapes (N,).
    sh - shooter draws, tg - target-side draws (dwell tau, protection, concealment); t_flight - minutes from
    firing to impact; interval - minutes between successive rounds of one mission (1 / (rate x guns))."""
    tau_target = tg["tau"]
    ps = psi(N_GRID, sh["s_tle"], sh["s_disp"], R_e * tg["protect"], interval / tau_target)   # (N, K)
    p_time = np.exp(-(sh["T"] + t_flight) / tau_target)[:, None]
    p_eng = sh["p_real"][:, None] * p_time * ps
    c_det = (sh["c_sortie"] * c_sortie_mult / (sh["p_find"] * tg["conceal"]))[:, None]
    cost = (c_det + N_GRID[None, :] * cr[:, None]) / np.maximum(p_eng, 1e-12)
    i = np.argmin(cost, axis=1)
    rows = np.arange(len(i))
    return {"n_star": N_GRID[i], "cost_kill": cost[rows, i], "p_kill": p_eng[rows, i] / sh["p_real"],
            "p_time": p_time[:, 0], "psi_star": ps[rows, i], "cost_curve": cost, "p_curve": p_eng}


def gun_outcomes(rng, det, art, p_kill, f_destroy, drone, p_reach, d_destroy, d_damage):
    """One mutually exclusive outcome for each detected gun: destroyed, damaged or untouched.

    Of det detected guns, art are engaged by artillery and drone are attacked by strike drones; both are random
    subsets of the detected guns chosen independently, so a gun is engaged by artillery with probability
    alpha = art / det and attacked by a drone with probability delta = drone / det, possibly both. Artillery
    neutralises an engaged gun with p_kill and destroys it with share f_destroy (otherwise damages it); a drone reaches
    it with p_reach and then destroys it with d_destroy or damages it with d_damage. Per gun:
        P(destroyed) = 1 - (1 - alpha p_kill f_destroy)(1 - delta p_reach d_destroy),
        P(untouched) = (1 - alpha p_kill)(1 - delta p_reach (d_destroy + d_damage)),
        P(damaged)   = 1 - P(destroyed) - P(untouched),
    and the det guns are split by a multinomial draw, so destroyed + damaged <= det by construction and a gun hit by
    both means is counted once. Also returns the expected numbers of guns hit by each means (for the drone share)."""
    det = np.asarray(det, int)
    n = np.maximum(det, 1)
    alpha = np.clip(np.asarray(art) / n, 0, 1)
    delta = np.clip(np.asarray(drone) / n, 0, 1)
    a_any, a_kill = alpha * p_kill, alpha * p_kill * f_destroy
    d_any, d_kill = delta * p_reach * (d_destroy + d_damage), delta * p_reach * d_destroy
    p_des = 1 - (1 - a_kill) * (1 - d_kill)
    p_unt = (1 - a_any) * (1 - d_any)
    p_dam = np.clip(1 - p_des - p_unt, 0, 1)
    pv = np.stack([p_des, p_dam, np.clip(1 - p_des - p_dam, 0, 1)], axis=-1)
    pv = pv / pv.sum(axis=-1, keepdims=True)
    out = rng.multinomial(det, pv)
    return out[..., 0], out[..., 1], det * a_any, det * d_any


# --- module B ----------------------------------------------------------------------------------------------
def duel(B, R, cm, pB, pR, aB, aR, rng, days=P.HORIZON_DAYS, gpm=None, g0=None):
    """Daily fires battle. Order of events within a day (both sides act on the state at the start of the day):
      1. detection of each operational enemy gun with probability q (hazard raised by its firing the day before);
      2. missions E = min(D / p_real, m_max G, a G / n*) - decoys take the share 1 - p_real of them;
         missions on real guns min(D, floor(p_real E)), each neutralising with P(n*) (module A);
      3. strike drones attack a share of the detected guns, the decoy share again absorbing sorties;
      4. each detected gun suffers at most one outcome: destroyed (artillery share f_destroy, drones fpv_destroy)
         or damaged; destroyed + damaged <= detected;
      5. losses of both sides are applied simultaneously; guns damaged on earlier days return with probability
         1 / t_repair; destroyed guns are replaced at Poisson(r_rep G0) per day, at most the number destroyed up to
         the previous day;
      6. a side culminates when fewer than theta G0 guns are operational; if both culminate on the same day, the
         side with the larger remaining share holds the field; otherwise the battle stops at the horizon."""
    n = len(aB)
    GB0, GR0 = g0 or (P.G_B0, P.G_R0)
    gpm = gpm or {"B": np.ones(n), "R": np.ones(n)}
    eB = engagement(B, R, pB["c_round"], cm["R_e"], pB["sortie_mult"], cm["t_flight"], 1 / (cm["r_fire"] * gpm["B"]))
    eR = engagement(R, B, pR["c_round"], cm["R_e"], pR["sortie_mult"], cm["t_flight"], 1 / (cm["r_fire"] * gpm["R"]))
    nB, nR = eB["n_star"], eR["n_star"]
    pkB = eB["p_kill"]
    pkR = eR["p_kill"]
    gB, gR = np.full(n, float(GB0)), np.full(n, float(GR0))
    dB, dR = np.zeros(n), np.zeros(n)
    misB, misR = np.zeros(n), np.zeros(n)
    alive = np.ones(n, bool)
    end_day = np.full(n, days, float)
    winner = np.zeros(n, int)                      # +1 Blue, -1 Red, 0 none within the horizon
    spend = {k: np.zeros(n) for k in ("rounds", "fpv", "isr", "c2")}
    lossB_val, lossR_val, killsR = np.zeros(n), np.zeros(n), np.zeros(n)
    ammo_days_B, ammo_days_R, hits_art_B, hits_drone_B = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    rounds_B, rounds_R = np.zeros(n), np.zeros(n)
    for day in range(days):
        a = alive.astype(float)

        def strike(sh, tg, eng_pk, nstar, ammo_pg, g_own, g_tgt, mis_tgt, prices_):
            q = 1 - np.exp(-sh["sorties"] * sh["p_find"] * tg["conceal"] * (1 + cm["k_fire"] * mis_tgt))
            det_real = rng.binomial(g_tgt.astype(int), q)
            cap = np.minimum(cm["m_max"] * g_own, ammo_pg * g_own / nstar)
            engaged = np.minimum(det_real / sh["p_real"], cap)
            real = np.minimum(det_real, np.floor(engaged * sh["p_real"])).astype(int)
            # strike drones attack a share of the detected (cued) guns; decoys absorb their share of sorties as well
            drone_real = rng.binomial(det_real, np.clip(sh["fpv"], 0, 1))
            fpv_sorties = drone_real / sh["p_real"]
            destroyed, damaged, exp_art, exp_drone = gun_outcomes(
                rng, det_real, real, eng_pk, cm["f_destroy"], drone_real, sh["p_reach"], cm["fpv_destroy"],
                cm["fpv_damage"])
            diag = {"ammo_bound": (det_real / sh["p_real"] > cap) & (ammo_pg * g_own / nstar < cm["m_max"] * g_own),
                    "hits_art": exp_art, "hits_drone": exp_drone}
            cost = {"rounds": engaged * nstar * prices_["c_round"], "fpv": fpv_sorties * prices_["c_fpv"],
                    "isr": sh["sorties"] * g_tgt * sh["c_sortie"] * prices_["sortie_mult"], "c2": sh["c_c2"]}
            diag["rounds"] = engaged * nstar
            return destroyed, damaged, engaged / np.maximum(g_own, 1), cost, diag

        desR, dmgR, misB_new, costB, dgB = strike(B, R, pkB, nB, aB, gB, gR, misR, pB)
        desB, dmgB, misR_new, _, dgR = strike(R, B, pkR, nR, aR, gR, gB, misB, pR)
        ammo_days_B += dgB["ammo_bound"] * a
        ammo_days_R += dgR["ammo_bound"] * a
        hits_art_B += dgB["hits_art"] * a
        hits_drone_B += dgB["hits_drone"] * a
        rounds_B += dgB["rounds"] * a
        rounds_R += dgR["rounds"] * a
        for k in spend:
            spend[k] += costB[k] * a
        lossB_val += desB * cm["c_gun_B"] * a
        lossR_val += desR * cm["c_gun_R"] * a
        killsR += desR * a
        repB = rng.binomial(dB.astype(int), 1 / cm["t_repair"])
        repR = rng.binomial(dR.astype(int), 1 / cm["t_repair"])
        # replacement of destroyed guns from reserves and production (cannot exceed what was lost)
        lostB, lostR = GB0 - gB - dB, GR0 - gR - dR
        newB = np.minimum(rng.poisson(cm["r_rep"] * GB0), np.maximum(lostB, 0))
        newR = np.minimum(rng.poisson(cm["r_rep"] * GR0), np.maximum(lostR, 0))
        gB = np.where(alive, gB - desB - dmgB + repB + newB, gB)
        gR = np.where(alive, gR - desR - dmgR + repR + newR, gR)
        dB = np.where(alive, dB + dmgB - repB, dB)
        dR = np.where(alive, dR + dmgR - repR, dR)
        misB, misR = misB_new, misR_new
        cul_B = gB < cm["theta"] * GB0
        cul_R = gR < cm["theta"] * GR0
        done = alive & (cul_B | cul_R)
        # simultaneous culmination: the side with the larger remaining share holds the field
        win_now = np.where(cul_R & ~cul_B, 1, np.where(cul_B & ~cul_R, -1,
                           np.where(gB / GB0 >= gR / GR0, 1, -1)))
        winner = np.where(done, win_now, winner)
        end_day = np.where(done, day + 1, end_day)
        alive &= ~done
        if not alive.any():
            break
    total = sum(spend.values())
    return {"days": end_day, "winner": winner, "spend": spend, "spend_total": total,
            "lossB_val": lossB_val, "lossR_val": lossR_val, "killsR": killsR,
            "ammo_bound_B": ammo_days_B / end_day, "ammo_bound_R": ammo_days_R / end_day,
            "drone_share_B": hits_drone_B / np.maximum(hits_art_B + hits_drone_B, 1),
            "fire_ratio": rounds_R / np.maximum(rounds_B, 1), "rounds_B_day": rounds_B / end_day,
            "rounds_R_day": rounds_R / end_day,
            "n_star_B": nB, "n_star_R": nR, "cost_kill_B": eB["cost_kill"], "p_kill_B": eB["p_kill"],
            "p_time_B": eB["p_time"], "psi_B": eB["psi_star"]}


# --- scenarios ---------------------------------------------------------------------------------------------
SCENARIOS = [
    # id, label, Blue kind, Red kind, year, erosion of Blue's AI gain, Blue posture
    ("S0", "2022 baseline", "2022", "2022", 2022, "none", "defence"),
    ("S1", "2025 drone-enabled", "2025", "2025", 2025, "none", "defence"),
    ("S2", "2033 symmetric AI", "ai", "ai", 2033, "none", "defence"),
    ("S3", "2036 symmetric AI", "ai", "ai", 2036, "none", "defence"),
    ("S4", "2036 Blue-only AI", "ai", "2025", 2036, "none", "defence"),
    ("S5", "2036 Blue AI, contested EW", "ai", "2025", 2036, "contested", "defence"),
    # offence: Blue attacks with 90 fire units against 45 defending ones (params.OFFENCE)
    ("O1", "offence 2025", "2025", "2025", 2025, "none", "offence"),
    ("O2", "offence 2036 symmetric AI", "ai", "ai", 2036, "none", "offence"),
    ("O3", "offence 2036 Blue-only AI", "ai", "2025", 2036, "none", "offence"),
    ("O4", "offence 2036 Blue AI, contested EW", "ai", "2025", 2036, "contested", "offence"),
]


def run_scenario(sid, kb, kr, year, eps, mode="defence", n=N_DRAWS, seed=SEED):
    """Seed: SEED + scenario number (+ 100 for offence), so every scenario is reproducible on its own."""
    rng = np.random.default_rng(seed + int(sid[1]) + (100 if mode == "offence" else 0))
    B = side_params(kb, year, n, rng, eps)
    R = side_params(kr, year, n, rng, "none")
    g0 = None
    if mode == "offence":         # the attacker's batteries move forward: easier to find, less protected
        O = P.OFFENCE
        B["conceal"] = B["conceal"] * lhs_triangular(O["conceal_mult"][0], n, rng)
        B["protect"] = np.minimum(1.0, B["protect"] * lhs_triangular(O["protect_mult"][0], n, rng))
        g0 = (O["G_att"], O["G_def"])
    cm = common(n, rng)
    pB, pR = prices(kb, year, n, rng), prices(kr, year, n, rng)
    # course-correcting fuzes on the adopted share of AI-era rounds; a separate stream keeps the other draws intact
    frng = np.random.default_rng(seed + 1000 + int(sid[1]) + (100 if mode == "offence" else 0))
    for side, pr in ((B, pB), (R, pR)):
        adopt = side.pop("adopt", None)
        if adopt is not None:
            pr["fuze"] = fuze_price(year, n, frng) * adopt
            pr["c_round"] = pr["c_round"] + pr["fuze"]
    aB, aR = ammo(kb, "B", n, rng), ammo(kr, "R", n, rng)
    gpm = {s: lhs_triangular(P.GUNS_PER_MISSION[s][0], n, rng) for s in ("B", "R")}
    out = duel(B, R, cm, pB, pR, aB, aR, rng, gpm=gpm, g0=g0)
    inputs = {f"B_{k}": v for k, v in B.items()} | {f"R_{k}": v for k, v in R.items()} | \
             {f"c_{k}": v for k, v in cm.items()} | {"price_round": pB["c_round"], "price_fpv": pB["c_fpv"], "price_fuze": pB.get("fuze", np.zeros(n)),
                                                    "ammo_B": aB, "ammo_R": aR,
                                                    "gpm_B": gpm["B"], "gpm_R": gpm["R"]}
    return out, inputs


def q(x, qs=(0.05, 0.5, 0.95)):
    return [float(v) for v in np.quantile(x, qs)]


def prcc(X, y):
    """Partial rank correlation of each column of X with y (rank-transformed linear partialling)."""
    def rank(v):
        r = np.empty_like(v, dtype=float)
        r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
        return r
    Xr = np.column_stack([rank(c) for c in X.T])
    yr = rank(y)
    out = []
    for j in range(Xr.shape[1]):
        others = np.column_stack([np.ones(len(yr)), np.delete(Xr, j, axis=1)])
        bx, *_ = np.linalg.lstsq(others, Xr[:, j], rcond=None)
        by, *_ = np.linalg.lstsq(others, yr, rcond=None)
        rx, ry = Xr[:, j] - others @ bx, yr - others @ by
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        out.append(float((rx * ry).sum() / den) if den > 0 else 0.0)
    return out


def summarise(out):
    days, win = out["days"], out["winner"]
    tot = out["spend_total"]
    shares = {k: float(np.median(v / np.maximum(tot, 1))) for k, v in out["spend"].items()}
    cer = out["lossR_val"] / np.maximum(tot + out["lossB_val"], 1)
    per_kill = tot / np.maximum(out["killsR"], 1)
    return {
        "days": q(days), "p_blue_wins": float(np.mean(win == 1)), "p_red_wins": float(np.mean(win == -1)),
        "p_undecided": float(np.mean(win == 0)),
        "spend_total_musd": q(tot / 1e6), "spend_per_day_musd": q(tot / days / 1e6),
        "loss_B_musd": q(out["lossB_val"] / 1e6), "loss_R_musd": q(out["lossR_val"] / 1e6),
        "cost_exchange": q(cer), "spend_per_red_gun_destroyed_kusd": q(per_kill / 1e3),
        "shares": shares,
        "n_star_B": q(out["n_star_B"]), "cost_per_kill_A_kusd": q(out["cost_kill_B"] / 1e3),
        "p_kill_B": q(out["p_kill_B"]), "p_time_B": q(out["p_time_B"]), "psi_B": q(out["psi_B"]),
        "ammo_bound_B": float(np.mean(out["ammo_bound_B"])), "ammo_bound_R": float(np.mean(out["ammo_bound_R"])),
        "drone_share_B": q(out["drone_share_B"]), "spend_total_mean_musd": float(np.mean(tot) / 1e6),
        "fire_ratio": q(out["fire_ratio"]), "rounds_B_day": q(out["rounds_B_day"]), "rounds_R_day": q(out["rounds_R_day"]),
        "mean_cost_battle_musd": float(np.mean(tot + out["lossB_val"]) / 1e6),
        "spend_shares_mean": {k: float(np.mean(v) / np.mean(tot)) for k, v in out["spend"].items()},
        "destroyed_value_per_day_musd": q((out["lossB_val"] + out["lossR_val"]) / days / 1e6),
        "burn_per_day_B_musd": q((tot + out["lossB_val"]) / days / 1e6),
    }


SENS_KEYS = ["B_T", "B_s_tle", "B_s_disp", "B_p_real", "B_sorties", "B_p_find", "B_fpv", "B_p_reach",
             "B_c_c2", "B_w", "R_T", "R_tau", "R_conceal", "R_protect", "B_tau", "B_conceal", "c_R_e",
             "c_k_fire", "c_m_max", "c_r_rep", "c_t_flight", "c_r_fire", "gpm_B", "gpm_R", "price_round",
             "price_fpv", "ammo_B", "ammo_R"]


def main():
    res = {"n_draws": N_DRAWS, "seed": SEED, "G_B0": P.G_B0, "G_R0": P.G_R0, "horizon_days": P.HORIZON_DAYS,
           "scenarios": {}}
    for sid, label, kb, kr, year, eps, mode in SCENARIOS:
        out, inp = run_scenario(sid, kb, kr, year, eps, mode)
        s = summarise(out)
        s.update(label=label, blue=kb, red=kr, year=year, erosion=eps, mode=mode)
        if mode == "offence":
            win = P.OFFENCE["window_days"][0]
            s["p_superiority_in_window"] = float(np.mean((out["winner"] == 1) & (out["days"] <= win)))
            s["window_days"] = win
        if kb == "ai":
            s["phi_B"] = q(inp["B_w"])
            s["price_round_usd"] = q(inp["price_round"])
            s["price_fuze_usd"] = q(inp["price_fuze"])
            s["price_fpv_usd"] = q(inp["price_fpv"])
        if sid in ("S1", "S3", "S5", "O3"):
            X = np.column_stack([inp[k] for k in SENS_KEYS])
            s["prcc_days"] = dict(zip(SENS_KEYS, prcc(X, out["days"])))
            s["prcc_cost_per_kill"] = dict(zip(SENS_KEYS, prcc(X, out["spend_total"] / np.maximum(out["killsR"], 1))))
            s["prcc_cer"] = dict(zip(SENS_KEYS, prcc(X, out["lossR_val"] / np.maximum(out["spend_total"] + out["lossB_val"], 1))))
        res["scenarios"][sid] = s
        np.savez_compressed(HERE / f"draws_{sid}.npz", days=out["days"], winner=out["winner"],
                            spend=out["spend_total"], killsR=out["killsR"], lossB=out["lossB_val"],
                            lossR=out["lossR_val"], cost_kill=out["cost_kill_B"], n_star=out["n_star_B"],
                            **{f"spend_{k}": v for k, v in out["spend"].items()})
        print(f"{sid} {label:34} days {s['days'][1]:6.1f} [{s['days'][0]:.0f}-{s['days'][2]:.0f}]  "
              f"Blue wins {s['p_blue_wins']:.2f}  spend {s['spend_total_musd'][1]:8.1f} M$  "
              f"per kill A {s['cost_per_kill_A_kusd'][1]:7.1f} k$  n* {s['n_star_B'][1]:.0f}  "
              f"CER {s['cost_exchange'][1]:.2f}  ammo-bound B {s['ammo_bound_B']:.2f} R {s['ammo_bound_R']:.2f}  "
              f"drone share {s['drone_share_B'][1]:.2f}")
    res["robustness"] = robustness()
    (HERE / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def robustness(n=2000):
    """Do the headline contrasts (S1 2025 vs S3 2036 symmetric AI) survive changes to structural inputs?"""
    variants = [("base", {}), ("force ratio 1:1", {"G_B0": 90}), ("force ratio 1:3", {"G_B0": 30}),
                ("culmination at 30 %", {"theta": 0.7}), ("culmination at 70 %", {"theta": 0.3}),
                ("no replacement", {"r_rep": 0.0}), ("round price x2", {"round_x": 2.0}),
                ("no hider adaptation after 2025", {"freeze_conceal": True})]
    out = {}
    keep = (P.G_B0, dict(P.COMMON), dict(P.ERA["conceal"]), dict(P.ERA["protect"]))
    for name, v in variants:
        P.G_B0 = v.get("G_B0", keep[0])
        P.COMMON = dict(keep[1])
        if "theta" in v:
            P.COMMON["theta"] = ((v["theta"],) * 3, "robustness")
        if "r_rep" in v:
            P.COMMON["r_rep"] = ((0.0, 0.0, 0.0), "robustness")
        P.ERA["conceal"], P.ERA["protect"] = dict(keep[2]), dict(keep[3])
        if v.get("freeze_conceal"):
            P.ERA["conceal"]["ai"], P.ERA["protect"]["ai"] = keep[2]["2025"], keep[3]["2025"]
        row = {}
        for sid, label, kb, kr, year, eps, mode in SCENARIOS:
            if sid not in ("S1", "S3", "S4"):
                continue
            o, inp = run_scenario(sid, kb, kr, year, eps, mode, n=n)
            if "round_x" in v:       # re-price rounds after the fact: spend on rounds scales linearly
                o["spend_total"] = o["spend_total"] + o["spend"]["rounds"] * (v["round_x"] - 1)
            row[sid] = {"days": float(np.median(o["days"])), "p_blue_wins": float(np.mean(o["winner"] == 1)),
                        "spend_per_red_gun_kusd": float(np.median(o["spend_total"] / np.maximum(o["killsR"], 1)) / 1e3),
                        "ammo_bound_B": float(np.mean(o["ammo_bound_B"]))}
        out[name] = row
        print(f"  {name:32}", {k: (round(r['days']), round(r['p_blue_wins'], 2), round(r['spend_per_red_gun_kusd']),
                                   round(r['ammo_bound_B'], 2)) for k, r in row.items()})
    P.G_B0, P.COMMON, P.ERA["conceal"], P.ERA["protect"] = keep[0], keep[1], keep[2], keep[3]
    return out


if __name__ == "__main__":
    main()
