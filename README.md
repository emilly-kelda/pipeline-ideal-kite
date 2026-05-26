# pipeline-ideal-kite

> **Python data pipeline that analyses historical winds from 10 kitesurfing destinations and recommends the best kites for each rider — considering weight, skill level, riding style, wind direction, and budget.**

> 🇧🇷 **Pipeline de dados em Python que analisa ventos históricos de 10 destinos de kitesurf e recomenda as melhores pipas para cada rider — considerando peso, habilidade, estilo de surf, direção do vento e orçamento.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-3.0-green)
![Status](https://img.shields.io/badge/status-completo-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The problem

Choosing the right kite quiver is harder than it looks.

The wrong size for your weight in your local wind conditions means you either can't get up — or you get launched dangerously. Most riders rely on gut feeling, YouTube, or a friend's advice. None of that accounts for what the wind actually does at your specific spot, across the entire season.

I worked at a kitesurfing school. I saw this problem up close — students renting the wrong gear, instructors making judgment calls without data, equipment decisions based on reputation rather than numbers.

This project uses data to answer the question properly.

> 🇧🇷 Escolher as pipas certas é mais difícil do que parece.
>
> O tamanho errado para o seu peso nas condições de vento locais significa que você não consegue levantar — ou é arremessado de forma perigosa. A maioria dos riders depende de intuição, YouTube ou conselho de amigos. Nada disso leva em conta o que o vento realmente faz no seu spot específico, ao longo de toda a temporada.
>
> Trabalhei numa escola de kitesurf. Vi esse problema de perto — alunos alugando equipamentos errados, instrutores tomando decisões sem dados, escolhas de equipamento baseadas em reputação em vez de números.
>
> Este projeto usa dados para responder à pergunta de forma adequada.

---

## What it does

```
Raw messy data  →  Clean structured data  →  Scored combinations  →  Optimal recommendation
```

**Three input datasets (synthetically generated with realistic mess):**
- `kite_catalog_raw.csv` — 284 kites across 10 brands with inconsistent formats
- `wind_observations_raw.csv` — 7,427 daily wind readings from 10 global kite spots
- `rider_profiles_raw.csv` — 20 rider profiles submitted via form with no validation

**Output:**
- Three clean CSVs ready for analysis
- A scored combinations table (~4,500 rider × kite pairs)
- Optimal quiver recommendation per rider — 1 kite if coverage ≥ 80%, 2 kites otherwise
- Score breakdown charts saved to `outputs/charts/`
- Full pipeline runs in under 20 seconds

---

## The data problems it solves

15+ classes of real-world data quality issues — all planted intentionally, all fixed:

| Problem | Example | Fix |
|---|---|---|
| Mixed units | `"20 knots"`, `"37 km/h"`, `"8 m/s"`, `"5 beaufort"` | Dispatch table + unit normalization |
| Mixed currencies | `"$2500"`, `"€1380"`, `"R$8500"` | Currency detection + conversion to USD |
| Inconsistent labels | `"adv"`, `"expert"`, `"Advanced"`, `"pro"` | Static map + fuzzy matching fallback |
| Mixed date formats | `"15/03/2024"`, `"Mar 15 2024"`, `"2024-03-15"` | Multi-format parser with try/except |
| Missing values | `None` wind minimums, missing gusts | Size-based estimation or flagged |
| Sensor spikes | `150 knots`, `-5 knots` | Physical constraint validation |
| Near-duplicate rows | Two loggers recording same day | Window-based deduplication |
| Orphaned foreign keys | `LOC99` in rider profiles | Cross-table validation + pipeline skip |
| Mixed weight units | `"79 kg"`, `"154 lbs"`, `"70"` (bare) | Regex detection + domain heuristic |
| Brand name variants | `"DUOTONE"`, `"Duo-tone"`, `"duotone sports"` | Hybrid static map + rapidfuzz |

---

## Scoring system

Each rider × kite combination is scored across 5 dimensions:

| Score | What it measures |
|---|---|
| `coverage_score` | % of wind days at rider's location inside kite's wind range |
| `skill_score` | % of those days inside rider's skill + style comfort window |
| `weight_score` | how well kite size matches rider weight at local average wind |
| `direction_score` | % of days with safe wind direction per location |
| `gust_score` | % of days where gusts stay within safe limits |
| `overall_score` | weighted combination of all five |

**Smart quiver logic:**
- If best single kite `coverage_score ≥ 0.80` → recommend 1 kite
- If `coverage_score < 0.80` → find best 2-kite combination with minimum 2m² size difference
- Kites the rider already owns are excluded from recommendations

---

## Why Pedro Dias needs two kites

18 out of 19 riders get a single-kite recommendation. Pedro is the exception.

Pedro is an advanced big air rider based in **Cape Town, South Africa** — the most wind-variable location in the dataset:

```
Cumbuco, Brazil:         avg 18 kn, std 3  →  CV = 0.17
Cape Town, South Africa: avg 24 kn, std 8  →  CV = 0.33
```

Cape Town's Coefficient of Variation is nearly 2× higher than Cumbuco. No single kite covers both ends. The optimizer correctly identifies this and recommends a 7m² + 9m² quiver with 91% combined coverage.

---

## How to run it

```bash
git clone https://github.com/emilly-kelda/pipeline-ideal-kite.git
cd pipeline-ideal-kite

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
python run_pipeline.py
```

One command. Under 20 seconds. Full output in `outputs/`.

---

## Project structure

```
pipeline-ideal-kite/
├── data/
│   ├── raw/          ← generated messy CSVs
│   └── clean/        ← pipeline output
├── outputs/
│   ├── charts/       ← one recommendation chart per rider
│   └── quiver_recommendations.csv
├── notebooks/
│   └── analysis.ipynb
├── tests/
│   └── test_cleaner.py
├── config.py         ← all constants in one place
├── data_generator.py ← creates synthetic messy data
├── cleaner.py        ← fixes all 15+ data problems
├── transformer.py    ← engineers 5 scores per rider × kite
├── analyzer.py       ← finds optimal quiver + generates charts
└── run_pipeline.py   ← runs everything end to end
```

---

## Key technical decisions

**Why generate synthetic data?**
Full self-containment — anyone can clone and run with zero external dependencies.

**Why hybrid static map + fuzzy matching?**
Static maps handle known synonyms with 100% accuracy. Fuzzy matching catches typos we didn't anticipate. This is Entity Resolution — a core data engineering problem.

**Why five separate scores instead of one?**
A single score hides the reason. Transparent scores make recommendations explainable.

**Why recommend 1 kite when coverage ≥ 80%?**
A recommendation engine that tells you "you only need one kite" is more trustworthy than one that always pushes two purchases.

**Why CV instead of standard deviation for location comparison?**
CV = std/mean gives fair relative comparison across locations with different average winds.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas 3.0 | Data manipulation and cleaning |
| NumPy | Numerical operations |
| rapidfuzz | Fuzzy string matching for entity resolution |
| matplotlib | Charts and visualizations |
| pytest | Unit tests |

---

## About

Built by **Emilly Kelda** — Applied AI undergraduate at PUCPR, former construction supervision intern, real estate agent, and kitesurfing school staff.

Open to data engineering, data analyst, and AI roles.

> 🇧🇷 Desenvolvido por **Emilly Kelda** — graduanda em IA Aplicada na PUCPR, ex-estagiária de supervisão de obras, corretora de imóveis e funcionária de escola de kitesurf.
>
> Aberta a oportunidades em engenharia de dados, análise de dados e IA.

📍 Goiânia, Brazil
🔗 [linkedin.com/in/emillyc](https://linkedin.com/in/emillyc)
📧 keldacori@gmail.com