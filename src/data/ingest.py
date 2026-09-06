"""Load transaction data from CSV/Parquet and optionally persist to SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.paths import DB_PATH, RAW_CSV, RAW_PARQUET

PathLike = Union[str, Path]


def load_transactions(
    path: Optional[PathLike] = None,
    prefer_parquet: bool = True,
) -> pd.DataFrame:
    """Load standardized raw transactions from data/raw/."""
    if path is not None:
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    elif prefer_parquet and RAW_PARQUET.exists():
        df = pd.read_parquet(RAW_PARQUET)
    elif RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
    else:
        raise FileNotFoundError(
            f"No raw dataset found at {RAW_PARQUET} or {RAW_CSV}. "
            "Run: python run_pipeline.py"
        )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def store_to_sqlite(
    df: pd.DataFrame,
    db_path: PathLike = DB_PATH,
    table_name: str = "transactions",
    if_exists: str = "replace",
) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # SQLite-friendly: convert categoricals / bins to string
    to_write = df.copy()
    for col in to_write.select_dtypes(include=["category"]).columns:
        to_write[col] = to_write[col].astype(str)
    with sqlite3.connect(db_path) as conn:
        to_write.to_sql(table_name, conn, if_exists=if_exists, index=False)
    return db_path


def query_sqlite(sql: str, db_path: PathLike = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)
