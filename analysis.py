"""
Netflix Movies & TV Shows — Exploratory Data Analysis
-----------------------------------------------------
Loads, cleans, and visualizes the netflix_titles.csv dataset.
No machine learning — Pandas + Matplotlib + Plotly only.

Run from the project root:
    python analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# Paths (relative to this script so the project runs from any working directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "netflix_titles.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = BASE_DIR / "charts"
CLEANED_CSV = OUTPUT_DIR / "cleaned_netflix.csv"

# Netflix-inspired accent for static charts
ACCENT = "#E50914"
ACCENT_ALT = "#221F1F"


# =============================================================================
# 1. Load & inspect
# =============================================================================

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Netflix titles CSV and print a quick preview."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"File: {path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values (raw):\n{df.isna().sum()}")
    return df


# =============================================================================
# 2. Clean
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop duplicate rows (and duplicate show_id if present)
    - Convert date_added to datetime
    - Fill missing categoricals with 'Unknown'
    - Strip whitespace from string columns
    - Drop rows missing type or title
    """
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    cleaned = df.copy()
    before = len(cleaned)

    # Drop full-row duplicates
    cleaned = cleaned.drop_duplicates()
    if "show_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["show_id"], keep="first")
    print(f"Duplicates removed: {before - len(cleaned)}")

    # Convert date_added (values look like "August 14, 2020")
    cleaned["date_added"] = pd.to_datetime(
        cleaned["date_added"].astype(str).str.strip(),
        errors="coerce",
    )
    print(f"date_added converted to datetime (NaT count: {cleaned['date_added'].isna().sum()})")

    # Fill common missing categoricals
    fill_cols = ["director", "cast", "country", "rating"]
    for col in fill_cols:
        if col in cleaned.columns:
            missing = cleaned[col].isna().sum()
            cleaned[col] = cleaned[col].fillna("Unknown")
            print(f"Filled {missing} missing values in '{col}' with 'Unknown'")

    # Strip whitespace on object columns
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Keep rows with valid type / title
    cleaned = cleaned.dropna(subset=["type", "title"])
    cleaned = cleaned[cleaned["title"].str.lower() != "nan"]

    print(f"Cleaned shape: {cleaned.shape[0]:,} rows x {cleaned.shape[1]} columns")
    return cleaned.reset_index(drop=True)


def save_cleaned(df: pd.DataFrame, path: Path = CLEANED_CSV) -> None:
    """Save the cleaned DataFrame to output/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nCleaned dataset saved to: {path}")


# =============================================================================
# 3. Summary
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    """Display dataset summary statistics."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values after cleaning:\n{df.isna().sum()}")

    type_counts = df["type"].value_counts()
    print(f"\nContent by type:\n{type_counts}")
    for content_type, count in type_counts.items():
        pct = 100 * count / len(df)
        print(f"  {content_type}: {count:,} ({pct:.1f}%)")


# =============================================================================
# 4. Analysis helpers
# =============================================================================

def explode_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Split a comma-separated column into individual values and return value counts.
    Example: 'United States, India' -> separate rows for each country.
    Excludes 'Unknown' from rankings.
    """
    series = (
        df[column]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    series = series[(series != "") & (series.str.lower() != "unknown") & (series.str.lower() != "nan")]
    return series.value_counts()


def top_n(series: pd.Series, n: int = 10) -> pd.Series:
    """Return the top n items from a value-counts Series."""
    return series.head(n)


def type_comparison(df: pd.DataFrame) -> pd.Series:
    """Count Movies vs TV Shows."""
    return df["type"].value_counts()


def releases_over_time(df: pd.DataFrame) -> pd.Series:
    """Count titles by release_year (sorted by year)."""
    return df["release_year"].value_counts().sort_index()


def rating_counts(df: pd.DataFrame) -> pd.Series:
    """Most common content ratings (excluding Unknown)."""
    ratings = df["rating"].value_counts()
    return ratings[ratings.index.str.lower() != "unknown"]


def movie_durations(df: pd.DataFrame) -> pd.Series:
    """
    Extract numeric duration in minutes for Movies only.
    Example: '90 min' -> 90
    """
    movies = df[df["type"] == "Movie"].copy()
    minutes = (
        movies["duration"]
        .astype(str)
        .str.extract(r"(\d+)\s*min", expand=False)
        .astype(float)
    )
    return minutes.dropna()


# =============================================================================
# 5. Matplotlib (static) charts
# =============================================================================

def _ensure_charts_dir() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_top_countries(counts: pd.Series, n: int = 10) -> Path:
    """Horizontal bar chart — Top N producing countries."""
    _ensure_charts_dir()
    data = top_n(counts, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data.index[::-1], data.values[::-1], color=ACCENT)
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Country")
    ax.set_title(f"Top {n} Countries Producing Netflix Content")
    fig.tight_layout()

    out = CHARTS_DIR / "top_countries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_top_genres(counts: pd.Series, n: int = 10) -> Path:
    """Bar chart — Top N genres."""
    _ensure_charts_dir()
    data = top_n(counts, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data.index, data.values, color=ACCENT)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Titles")
    ax.set_title(f"Top {n} Most Common Genres on Netflix")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = CHARTS_DIR / "top_genres.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_movies_vs_tv(counts: pd.Series) -> Path:
    """Pie chart — Movies vs TV Shows."""
    _ensure_charts_dir()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [ACCENT, ACCENT_ALT]
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(counts)],
        textprops={"fontsize": 12},
    )
    ax.set_title("Movies vs TV Shows on Netflix")
    fig.tight_layout()

    out = CHARTS_DIR / "movies_vs_tv.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_releases_over_time(yearly: pd.Series) -> Path:
    """Line chart — Number of releases by release_year."""
    _ensure_charts_dir()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly.index, yearly.values, color=ACCENT, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")
    ax.set_title("Number of Releases Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = CHARTS_DIR / "releases_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_content_ratings(counts: pd.Series, n: int = 10) -> Path:
    """Bar chart — Most common content ratings."""
    _ensure_charts_dir()
    data = top_n(counts, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data.index, data.values, color=ACCENT)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Titles")
    ax.set_title(f"Top {n} Most Common Content Ratings")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = CHARTS_DIR / "content_ratings.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_top_directors(counts: pd.Series, n: int = 10) -> Path:
    """Bar chart — Top N directors."""
    _ensure_charts_dir()
    data = top_n(counts, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data.index[::-1], data.values[::-1], color=ACCENT)
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Director")
    ax.set_title(f"Top {n} Directors on Netflix")
    fig.tight_layout()

    out = CHARTS_DIR / "top_directors.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_top_actors(counts: pd.Series, n: int = 10) -> Path:
    """Bar chart — Top N cast members."""
    _ensure_charts_dir()
    data = top_n(counts, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data.index[::-1], data.values[::-1], color=ACCENT)
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Actor / Cast Member")
    ax.set_title(f"Top {n} Actors / Cast Members on Netflix")
    fig.tight_layout()

    out = CHARTS_DIR / "top_actors.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_movie_duration_hist(minutes: pd.Series) -> Path:
    """Histogram — Distribution of movie durations (minutes)."""
    _ensure_charts_dir()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(minutes, bins=40, color=ACCENT, edgecolor="white", alpha=0.9)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Distribution of Movie Durations")
    ax.axvline(minutes.median(), color=ACCENT_ALT, linestyle="--", linewidth=2, label=f"Median: {minutes.median():.0f} min")
    ax.legend()
    fig.tight_layout()

    out = CHARTS_DIR / "movie_duration_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# =============================================================================
# 6. Plotly (interactive) charts
# =============================================================================

def plotly_top_countries(counts: pd.Series, n: int = 10) -> Path:
    """Interactive horizontal bar — Top N countries."""
    _ensure_charts_dir()
    data = top_n(counts, n).reset_index()
    data.columns = ["country", "count"]

    fig = px.bar(
        data.sort_values("count"),
        x="count",
        y="country",
        orientation="h",
        title=f"Top {n} Countries Producing Netflix Content",
        labels={"count": "Number of Titles", "country": "Country"},
        color_discrete_sequence=[ACCENT],
    )
    out = CHARTS_DIR / "top_countries_interactive.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return out


def plotly_top_genres(counts: pd.Series, n: int = 10) -> Path:
    """Interactive bar — Top N genres."""
    _ensure_charts_dir()
    data = top_n(counts, n).reset_index()
    data.columns = ["genre", "count"]

    fig = px.bar(
        data,
        x="genre",
        y="count",
        title=f"Top {n} Most Common Genres on Netflix",
        labels={"count": "Number of Titles", "genre": "Genre"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(xaxis_tickangle=-45)
    out = CHARTS_DIR / "top_genres_interactive.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return out


def plotly_movies_vs_tv(counts: pd.Series) -> Path:
    """Interactive pie — Movies vs TV Shows."""
    _ensure_charts_dir()
    data = counts.reset_index()
    data.columns = ["type", "count"]

    fig = px.pie(
        data,
        names="type",
        values="count",
        title="Movies vs TV Shows on Netflix",
        color_discrete_sequence=[ACCENT, ACCENT_ALT],
    )
    out = CHARTS_DIR / "movies_vs_tv_interactive.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return out


def plotly_releases_over_time(yearly: pd.Series) -> Path:
    """Interactive line — Releases over time."""
    _ensure_charts_dir()
    data = yearly.reset_index()
    data.columns = ["release_year", "count"]

    fig = px.line(
        data,
        x="release_year",
        y="count",
        markers=True,
        title="Number of Releases Over Time",
        labels={"release_year": "Release Year", "count": "Number of Titles"},
        color_discrete_sequence=[ACCENT],
    )
    out = CHARTS_DIR / "releases_over_time_interactive.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return out


def plotly_content_ratings(counts: pd.Series, n: int = 10) -> Path:
    """Interactive bar — Content ratings."""
    _ensure_charts_dir()
    data = top_n(counts, n).reset_index()
    data.columns = ["rating", "count"]

    fig = px.bar(
        data,
        x="rating",
        y="count",
        title=f"Top {n} Most Common Content Ratings",
        labels={"count": "Number of Titles", "rating": "Rating"},
        color_discrete_sequence=[ACCENT],
    )
    out = CHARTS_DIR / "content_ratings_interactive.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return out


def plotly_movie_duration(minutes: pd.Series) -> Path:
    """Interactive histogram — Movie durations."""
    _ensure_charts_dir()
    data = pd.DataFrame({"duration_minutes": minutes})

    fig = px.histogram(
        data,
        x="duration_minutes",
        nbins=40,
        title="Distribution of Movie Durations",
        labels={"duration_minutes": "Duration (minutes)"},
        color_discrete_sequence=[ACCENT],
    )
    out = CHARTS_DIR / "movie_duration_interactive.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")
    return out


# =============================================================================
# 7. Insights
# =============================================================================

def print_insights(
    df: pd.DataFrame,
    country_counts: pd.Series,
    genre_counts: pd.Series,
    type_counts: pd.Series,
    yearly: pd.Series,
    ratings: pd.Series,
    director_counts: pd.Series,
    actor_counts: pd.Series,
    minutes: pd.Series,
) -> None:
    """Print key takeaways from the analysis."""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    total = len(df)
    movie_pct = 100 * type_counts.get("Movie", 0) / total
    tv_pct = 100 * type_counts.get("TV Show", 0) / total
    peak_year = int(yearly.idxmax())
    peak_count = int(yearly.max())

    print(f"1. Top producing country: {country_counts.index[0]} ({country_counts.iloc[0]:,} titles)")
    print(f"2. Most common genre: {genre_counts.index[0]} ({genre_counts.iloc[0]:,} titles)")
    print(f"3. Content mix: Movies {movie_pct:.1f}% | TV Shows {tv_pct:.1f}%")
    print(f"4. Peak release year: {peak_year} ({peak_count:,} titles)")
    print(f"5. Most common rating: {ratings.index[0]} ({ratings.iloc[0]:,} titles)")
    print(f"6. Top director: {director_counts.index[0]} ({director_counts.iloc[0]:,} titles)")
    print(f"7. Top cast member: {actor_counts.index[0]} ({actor_counts.iloc[0]:,} titles)")
    if len(minutes) > 0:
        print(
            f"8. Movie duration: mean {minutes.mean():.0f} min | "
            f"median {minutes.median():.0f} min | "
            f"range {minutes.min():.0f}–{minutes.max():.0f} min"
        )
    print(f"9. Dataset spans release years {int(df['release_year'].min())}–{int(df['release_year'].max())}")
    print("=" * 60)


# =============================================================================
# 8. Main pipeline
# =============================================================================

def main() -> None:
    """Run the full analysis pipeline."""
    # Load & clean
    raw = load_data()
    df = clean_data(raw)
    save_cleaned(df)
    print_summary(df)

    # Compute analysis series
    print("\n" + "=" * 60)
    print("RUNNING ANALYSES & CREATING CHARTS")
    print("=" * 60)

    country_counts = explode_column(df, "country")
    genre_counts = explode_column(df, "listed_in")
    director_counts = explode_column(df, "director")
    actor_counts = explode_column(df, "cast")
    type_counts = type_comparison(df)
    yearly = releases_over_time(df)
    ratings = rating_counts(df)
    minutes = movie_durations(df)

    # Matplotlib static charts
    plot_top_countries(country_counts)
    plot_top_genres(genre_counts)
    plot_movies_vs_tv(type_counts)
    plot_releases_over_time(yearly)
    plot_content_ratings(ratings)
    plot_top_directors(director_counts)
    plot_top_actors(actor_counts)
    plot_movie_duration_hist(minutes)

    # Plotly interactive charts
    plotly_top_countries(country_counts)
    plotly_top_genres(genre_counts)
    plotly_movies_vs_tv(type_counts)
    plotly_releases_over_time(yearly)
    plotly_content_ratings(ratings)
    plotly_movie_duration(minutes)

    # Insights
    print_insights(
        df,
        country_counts,
        genre_counts,
        type_counts,
        yearly,
        ratings,
        director_counts,
        actor_counts,
        minutes,
    )

    print("\nDone! Open the charts/ folder for PNGs and interactive HTML files.")
    print(f"Cleaned data: {CLEANED_CSV}")


if __name__ == "__main__":
    main()
