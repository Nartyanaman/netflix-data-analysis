# Netflix Movies & TV Shows — Data Analysis

A beginner-friendly exploratory data analysis (EDA) project using the public **Netflix Titles** dataset.  
This project uses **Pandas** for cleaning, **Matplotlib** for static charts, and **Plotly** for interactive visualizations.  
**No machine learning.**

---

## Project structure

```
netflix-data-analysis/
├── analysis.py              # Full modular analysis script
├── notebook.ipynb           # Same analysis in Jupyter notebook form
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   └── netflix_titles.csv   # Raw dataset
├── charts/                  # Generated PNG + interactive HTML charts
└── output/
    └── cleaned_netflix.csv  # Cleaned dataset
```

---

## Dataset

- **Source:** [Kaggle — Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) (collected via Flixable)
- **File:** `data/netflix_titles.csv`
- **Columns:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`

---

## Setup (VS Code / Cursor)

1. Open the `netflix-data-analysis` folder in VS Code or Cursor.
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
```

3. Activate it:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to run

### Option A — Python script (recommended first run)

From the project root:

```bash
python analysis.py
```

This will:

1. Load and clean the data
2. Save `output/cleaned_netflix.csv`
3. Create all charts in `charts/`
4. Print a dataset summary and key insights

### Option B — Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

Or open `notebook.ipynb` in VS Code / Cursor and run all cells (top to bottom).

---

## Analyses included

| Question | Visualization |
|---|---|
| Top 10 producing countries | Bar (Matplotlib + Plotly) |
| Top 10 genres | Bar (Matplotlib + Plotly) |
| Movies vs TV Shows | Pie (Matplotlib + Plotly) |
| Releases over time | Line (Matplotlib + Plotly) |
| Most common ratings | Bar (Matplotlib + Plotly) |
| Top 10 directors | Bar (Matplotlib) |
| Top 10 actors / cast | Bar (Matplotlib) |
| Movie duration distribution | Histogram (Matplotlib + Plotly) |

---

## Cleaning steps

- Remove duplicate rows (and duplicate `show_id`)
- Convert `date_added` to datetime
- Fill missing `director`, `cast`, `country`, and `rating` with `"Unknown"`
- Strip whitespace from text columns
- Parse movie durations (e.g. `"90 min"` → `90`) for the histogram

---

## Outputs

After running the analysis you should see:

- **Cleaned data:** `output/cleaned_netflix.csv`
- **Static charts:** `charts/*.png`
- **Interactive charts:** `charts/*_interactive.html` (open in a browser)

---

## Requirements

- Python 3.9+
- pandas, matplotlib, plotly
- jupyter / ipykernel (for the notebook)

See `requirements.txt` for pinned minimum versions.

---

## Attribution

Dataset originally published on Kaggle by Shivam Bansal, based on Flixable Netflix catalog data.
