# AI counter-battery economics

Simulation model behind the article *"The Economics of AI-Enabled Artillery in a Future High-Intensity War:
Modelling the Counter-Battery Fight"* by Roman Tymoshuk and Nataliia Susol (submitted to *Military Science*).

The model estimates what a counter-battery fight between 152/155-mm howitzer groupings costs and how long it lasts,
from the war in Ukraine in 2022 and 2025 to a notional conflict seven to ten years ahead (2033–2036), when
AI-enabled reconnaissance-fire systems are mature. Defence and offence are modelled; every input is a triangular
distribution, and its source is recorded in the code.

## Structure
| Path | Contents |
|---|---|
| `model/params.py` | every input as a triangular distribution (low, mode, high) with its source tag: a reference key of the article, or `assumption` |
| `model/model.py` | module A (one fire mission, optimal number of rounds), module B (daily counter-battery duel up to 180 days), module C (AI adoption and unit costs, 2025→2036); ten scenarios, partial rank correlation (PRCC) and robustness variants |
| `model/test_model.py` | sanity tests of the engagement and duel functions |
| `model/results.json` | summary of all scenarios (medians and 5–95 % intervals), PRCC and robustness |
| `model/outputs/*.csv` | the same numbers as flat tables: `scenarios.csv`, `robustness.csv`, `prcc.csv` |
| `make_figures.py` | script that draws the seven figures of the article into `figures/` |

The order of events within a simulated day is documented in the docstring of `duel()` in `model/model.py`.

## Scenarios
- **S0–S5, defence:** S0 — 2022; S1 — 2025; S2 — 2033; S3 — 2036; S4 and S5 — 2036 variants of the AI
  balance between the sides (see `SCENARIOS` in `model/model.py`).
- **O1–O4, offence:** the attacker masses 90 fire units against 45 defending ones and must win fire superiority
  within a 30-day preparation window.

## Reproduce
Python 3.12+.
```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd model
../.venv/bin/python model.py          # about 10 s; writes results.json and draws_*.npz
../.venv/bin/python export_csv.py     # writes outputs/*.csv
../.venv/bin/python -m pytest -q      # optional, needs pytest
cd ..
.venv/bin/python make_figures.py      # writes figures/fig1–fig7 (.png, .tif); uses Times New Roman if installed
```
Random numbers: `numpy.random.default_rng(20261025 + k)`, where `k` is the scenario number (S0–S5: 0–5;
O1–O4: 101–104). 4,000 Latin-hypercube draws per scenario, 2,000 per robustness cell. With NumPy 2.5.1 the run
reproduces `results.json` exactly.

## Money
All monetary values are in constant 2025 US dollars.

## Limitations
The model is a notional sector-level abstraction calibrated on open sources. The mature-AI values are
assumptions, marked `assumption` in `model/params.py` and ranked by the sensitivity analysis; they are not
forecasts of any real force.

## Citation
If you use the model, please cite the article (see `CITATION.cff`).

## License
MIT (see `LICENSE`).
