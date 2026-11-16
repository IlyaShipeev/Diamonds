"""Общие пути и списки признаков для ноутбука и приложения Streamlit."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "diamonds.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
REPORT_PATH = ROOT / "artifacts" / "benchmark.json"

NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
CATEGORIES = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": list("DEFGHIJ"),
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}
FEATURES = NUMERIC + list(CATEGORIES)

