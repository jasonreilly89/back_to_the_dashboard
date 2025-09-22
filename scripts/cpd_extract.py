#!/usr/bin/env python3
"""Extract CPD context data for dashboard API endpoints."""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_models(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    model_cols = [col for col in df.columns if col.startswith("model_post_")]
    if model_cols:
        names = [col.replace("model_post_", "") for col in model_cols]
        values = df[model_cols].to_numpy(dtype=float, copy=False)
        return names, values

    if "model_post" not in df.columns:
        return [], np.empty((len(df), 0))

    names: list[str] = []
    matrix = np.zeros((len(df), 0))
    rows = []
    for entry in df["model_post"]:
        entry = entry or {}
        if isinstance(entry, dict):
            for key in entry.keys():
                if key not in names:
                    names.append(key)
            rows.append(entry)
        else:
            rows.append({})
    matrix = np.zeros((len(rows), len(names)), dtype=float)
    for row_idx, entry in enumerate(rows):
        for col_idx, name in enumerate(names):
            matrix[row_idx, col_idx] = float(entry.get(name, 0.0))
    return names, matrix


def extract_series(df: pd.DataFrame, downsample: int) -> list[dict[str, Any]]:
    if downsample > 1:
        df = df.iloc[::downsample, :].copy()
    df = df.sort_values("timestamp")
    timestamps = _normalise_timestamp(df["timestamp"])
    log_col = None
    for candidate in ("logpred", "log_pred", "logPred"):
        if candidate in df.columns:
            log_col = candidate
            break

    records: list[dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        row = df.iloc[idx]
        payload = {
            "timestamp": ts.isoformat() if not pd.isna(ts) else None,
            "cp_prob": _safe_float(row.get("cp_prob")),
            "runlen_map": _safe_int(row.get("runlen_map")),
            "runlen_expect": _safe_float(row.get("runlen_expect")),
            "expected_runlen": _safe_float(row.get("expected_runlen")),
        }
        if log_col:
            payload["log_pred"] = _safe_float(row.get(log_col))
        records.append(payload)
    return records


def extract_models(df: pd.DataFrame, downsample: int) -> list[dict[str, Any]]:
    if downsample > 1:
        df = df.iloc[::downsample, :].copy()
    df = df.sort_values("timestamp")
    timestamps = _normalise_timestamp(df["timestamp"])
    names, matrix = _normalise_models(df)

    records: list[dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        payload = {"timestamp": ts.isoformat() if not pd.isna(ts) else None}
        for col_idx, name in enumerate(names):
            payload[name] = float(matrix[idx, col_idx])
        records.append(payload)
    return records


def extract_summary(df: pd.DataFrame) -> dict[str, Any]:
    timestamps = _normalise_timestamp(df["timestamp"])
    count = int(len(df))
    cp_prob = df["cp_prob"].astype(float) if "cp_prob" in df.columns else pd.Series(dtype=float)
    runlen_col = "runlen_expect" if "runlen_expect" in df.columns else "expected_runlen"
    runlen = df[runlen_col].astype(float) if runlen_col in df.columns else pd.Series(dtype=float)

    names, matrix = _normalise_models(df)
    model_means = {name: float(matrix[:, idx].mean()) for idx, name in enumerate(names)} if names else {}

    summary = {
        "count": count,
        "timestamps": {
            "start": timestamps.min().isoformat() if count else None,
            "end": timestamps.max().isoformat() if count else None,
        },
        "cp_prob": {
            "mean": _safe_float(cp_prob.mean()) if count else None,
            "max": _safe_float(cp_prob.max()) if count else None,
            "p95": _safe_float(cp_prob.quantile(0.95)) if count else None,
        },
        "runlen_expect": {
            "mean": _safe_float(runlen.mean()) if count else None,
            "max": _safe_float(runlen.max()) if count else None,
        },
        "models": model_means,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CPD context artefacts")
    parser.add_argument("--file", required=True, help="Path to cpd_ctx.parquet")
    parser.add_argument("--mode", choices=["series", "models", "summary"], required=True)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    if "timestamp" not in df.columns:
        raise SystemExit("cpd_ctx parquet missing timestamp column")

    downsample = max(1, int(args.downsample or 1))

    if args.mode == "series":
        data = extract_series(df, downsample)
    elif args.mode == "models":
        data = extract_models(df, downsample)
    else:
        data = extract_summary(df)

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
