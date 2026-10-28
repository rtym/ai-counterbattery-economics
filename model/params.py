"""Parameters of the RVA kill-chain economics model, with their provenance.

Every uncertain input is a triangular distribution (low, mode, high). Each entry carries a
"src" tag: a key from ../../sources.json when the range is anchored on a published figure, or
"assumption" when it is a modelling judgement (these are the ones the sensitivity analysis
must rank, and the article reports them as assumptions, not findings).

Eras: "2022" - fires routed through headquarters, scarce UAVs (RUSI, Zabrodskyi et al. 2022);
      "2025" - organic UAV kill chains, mass observation, FPV layer (RUSI, Watling & Reynolds 2025);
      "ai"   - mature AI-enabled kill web, the end state that the adoption curve approaches.
Costs are in 2025 US dollars.
"""

# --- kill chain (module A) and fires duel (module B) -------------------------------------------
ERA = {
    # detect-to-effect latency, minutes. 2022: 20-30 min via a fire-control HQ, 3-5 min with organic UAV
    # (zabrodskyi2022). Kropyva cut the time to engage an unplanned target by two-thirds (zabrodskyi2022).
    "T": {"2022": ((10, 20, 30), "zabrodskyi2022"), "2025": ((3, 4, 8), "zabrodskyi2022"),
          "ai": ((0.5, 1.0, 2.0), "assumption")},
    # mean dwell of an enemy fire unit at a detectable position, minutes. 2022: Russian guns rarely
    # displaced unless fired upon (zabrodskyi2022); 2025: dug-in guns fire for protracted periods but move
    # periodically (watling2025); AI era: automated shoot-and-scoot shortens dwell.
    "tau": {"2022": ((60, 120, 240), "zabrodskyi2022"), "2025": ((30, 60, 120), "watling2025"),
            "ai": ((8, 15, 30), "assumption")},
    # target location error, m (1-sigma per axis): map/observer -> UAV with rangefinder -> AI sensor fusion
    "s_tle": {"2022": ((30, 50, 80), "assumption"), "2025": ((10, 18, 30), "assumption"),
              "ai": ((3, 6, 10), "assumption")},
    # ballistic dispersion of a 155/152-mm round, m (1-sigma); 2025 widened by mixed propellant charges
    # (watling2025); AI era assumes course-correcting fuzes and automated corrections.
    "s_disp": {"2022": ((40, 55, 80), "assumption"), "2025": ((40, 60, 85), "watling2025"),
               "ai": ((15, 25, 40), "assumption")},
    # share of engaged objects that are real targets (decoys); decoy positions are standard by 2025 (watling2025)
    "p_real": {"2022": ((0.70, 0.80, 0.90), "assumption"), "2025": ((0.50, 0.65, 0.80), "watling2025"),
               "ai": ((0.80, 0.88, 0.95), "assumption")},
    # ISR sorties per enemy fire unit per day and the chance that a sortie finds it; only about a third of
    # UAV missions succeeded in 2022 (zabrodskyi2022); reconnaissance is "almost exclusively" UAV by 2025
    "sorties": {"2022": ((0.3, 0.6, 1.0), "zabrodskyi2022"), "2025": ((1.0, 2.0, 3.0), "watling2025"),
                "ai": ((2.0, 4.0, 6.0), "assumption")},
    "p_find": {"2022": ((0.15, 0.25, 0.35), "zabrodskyi2022"), "2025": ((0.20, 0.30, 0.40), "assumption"),
               "ai": ((0.30, 0.45, 0.60), "assumption")},
    # share of detected enemy fire units also attacked by strike drones (counter-battery depth 15-25 km needs
    # mid-range strike drones rather than short-range FPVs); share of drones reaching the target: 60-80 % of
    # Ukrainian FPVs fail to reach the target (watling2025)
    "fpv": {"2022": ((0.0, 0.0, 0.0), "zabrodskyi2022"), "2025": ((0.2, 0.4, 0.7), "watling2025"),
            "ai": ((0.5, 0.7, 0.9), "assumption")},
    "p_reach": {"2022": ((0.2, 0.3, 0.4), "watling2025"), "2025": ((0.20, 0.30, 0.40), "watling2025"),
                "ai": ((0.40, 0.55, 0.70), "assumption")},
    # hider side (attributes of the TARGET): concealment multiplies the finder's detection hazard; protection
    # multiplies the effective lethal radius. 2022: Russian guns static and rarely displaced (zabrodskyi2022);
    # 2025: guns >500 m apart, dug in, netted, sheathed, decoy positions (watling2025); AI era: hider adaptation
    # continues (calcara2022: enduring hider-finder competition)
    "conceal": {"2022": ((0.6, 0.8, 1.0), "zabrodskyi2022"), "2025": ((0.2, 0.3, 0.4), "watling2025"),
                "ai": ((0.10, 0.18, 0.30), "calcara2022")},
    "protect": {"2022": ((0.9, 1.0, 1.0), "zabrodskyi2022"), "2025": ((0.6, 0.7, 0.85), "watling2025"),
                "ai": ((0.55, 0.65, 0.8), "assumption")},
    # cost of an ISR sortie, USD: platform price over expected flights (quadcopter ~3, fixed wing ~6 flights,
    # zabrodskyi2022); AI-era autonomy assumed to raise survivability
    "c_sortie": {"2022": ((1000, 2500, 5000), "zabrodskyi2022"), "2025": ((1000, 2500, 5000), "zabrodskyi2022"),
                 "ai": ((700, 1500, 3000), "assumption")},
    # AI / C2 running cost per sector per day (compute, links, software, specialists), USD
    "c_c2": {"2022": ((2000, 5000, 10000), "assumption"), "2025": ((5000, 10000, 20000), "assumption"),
             "ai": ((20000, 50000, 100000), "assumption")},
}

COMMON = {
    # Carleton effective lethal radius of one 155/152-mm HE round against a prepared gun position, m
    "R_e": ((8, 12, 18), "assumption"),
    "m_max": ((4, 6, 10), "assumption"),           # fire missions a gun can execute per day
    "f_destroy": ((0.5, 0.65, 0.8), "assumption"),  # share of artillery kills that destroy (rest damage)
    # FPV hits on guns: mostly repairable damage (watling2025)
    "fpv_destroy": ((0.03, 0.06, 0.10), "watling2025"),
    "fpv_damage": ((0.30, 0.45, 0.60), "watling2025"),
    "t_repair": ((2, 4, 7), "assumption"),          # days to return a damaged gun
    "k_fire": ((0.1, 0.25, 0.5), "assumption"),     # extra detectability per fire mission a gun executes
    "t_set": ((2, 4, 6), "assumption"),             # minutes from occupying a position to first round
    "r_fire": ((2, 3, 4), "assumption"),            # rounds per minute per gun, burst
    "t_flight": ((0.5, 1.0, 1.5), "assumption"),    # minutes from firing to impact at 15-25 km
    "c_gun_B": ((2.5e6, 3.5e6, 5.0e6), "assumption"),
    "c_gun_R": ((2.0e6, 3.0e6, 4.5e6), "cody2026"),  # Russian artillery system ~ USD 4 M (cody2026 anecdote)
    "theta": ((0.5, 0.5, 0.5), "assumption"),       # culmination: half of the fire units out of action
    # replacement of destroyed fire units from reserves and production, share of establishment per day
    "r_rep": ((0.002, 0.005, 0.01), "assumption"),
}

# 155-mm HE round, USD. NATO: EUR 2,000 at the start of the full-scale invasion, EUR 8,000 later; Western
# European production USD 5,000-10,000 (defensenews2024). The AI-era price comes from module C.
C_ROUND = {"2022": ((2000, 2500, 3500), "defensenews2024"), "2025": ((4000, 6000, 9000), "defensenews2024")}
# strike drone able to reach guns 15-25 km deep, USD: above the FPV ceiling of USD 1,000 (cody2026), below
# loitering munitions of the Lancet class; a modelling range
C_FPV = {"2025": ((3000, 10000, 35000), "assumption")}

# rounds available per gun per day. Russia: ~20,000 rounds/day from >1,100 tubes in the Donbas offensive;
# Ukraine: rarely above 6,000 rounds/day (zabrodskyi2022). 2025 Ukrainian brigades short of 152/122-mm
# rounds and charges (watling2025).
AMMO = {
    ("2022", "B"): ((3, 5, 7), "zabrodskyi2022"), ("2022", "R"): ((14, 18, 25), "zabrodskyi2022"),
    ("2025", "B"): ((4, 6, 9), "watling2025"), ("2025", "R"): ((8, 11, 15), "assumption"),
    ("ai", "B"): ((6, 10, 15), "assumption"), ("ai", "R"): ((8, 11, 15), "assumption"),
}

# guns firing one mission: the UAF mostly assign fire missions to guns, Russian missions go to the battery
# (zabrodskyi2022); the battery size of 4-6 guns is an assumption
GUNS_PER_MISSION = {"B": ((1.0, 1.5, 2.0), "zabrodskyi2022"), "R": ((4.0, 5.0, 6.0), "zabrodskyi2022")}

# Sector: a Russian artillery group of 80-90 guns supports a brigade-size grouping (zabrodskyi2022);
# sector force ratios 1:2 to 1:6 (watling2025).
G_R0 = 90
G_B0 = 45

# Offence: Blue masses 90 fire units against 45 defending ones and must win fire superiority within the preparation
# window; the Kursk offensive involved up to a month of preparatory intelligence collection and shaping fires, and all assault
# actions must be enabled by fires (watling2025). Batteries moving forward are easier to find and less protected
# than dug-in defenders (assumption).
OFFENCE = {"G_att": 90, "G_def": 45, "window_days": (30, "watling2025"),
           "conceal_mult": ((1.3, 1.6, 2.0), "assumption"), "protect_mult": ((1.1, 1.25, 1.4), "assumption")}
HORIZON_DAYS = 180

# --- module C: technology trajectories 2025 -> 2036 ---------------------------------------------
# Wright's law, statistically validated across >50 technologies (way2022): unit cost falls by a constant
# fraction (learning rate) per doubling of cumulative production.
LEARN = {
    # learning rate per doubling, doublings of cumulative output 2025 -> 2036
    "fpv":    {"lr": ((0.12, 0.18, 0.25), "way2022"), "doublings": ((3, 4.5, 6), "assumption"),
               # AI terminal-guidance payload adds cost per unit before learning
               "ai_premium": ((1.3, 1.6, 2.0), "assumption")},
    "round":  {"lr": ((0.03, 0.06, 0.10), "assumption"), "doublings": ((1.0, 1.5, 2.5), "scarazzato2024"),
               "scarcity_relief": ((0.65, 0.75, 0.9), "assumption")},   # price premium of 2023-25 unwinds
    "sortie": {"lr": ((0.10, 0.15, 0.22), "way2022"), "doublings": ((2, 3, 4), "assumption")},
}
# logistic adoption of the AI kill web: phi(t) = 1 / (1 + exp(-k (t - t_mid)))
ADOPT = {"t_mid": ((2028.5, 2030.0, 2032.0), "assumption"), "k": ((0.5, 0.8, 1.2), "assumption")}
# adversary adaptation (EW, deception) erodes the AI gain by a share epsilon (calcara2022, hwang2026)
EROSION = {"none": ((0.0, 0.0, 0.0), "assumption"), "contested": ((0.3, 0.45, 0.6), "hwang2026")}
