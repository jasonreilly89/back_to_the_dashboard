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
    log_col = next((col for col in ("logpred", "log_pred", "logPred") if col in df.columns), None)

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
    model_means = {name: float(matrix[:, idx].mean()) for idx, name in enumerate(names)} if len(names) > 0 else {}

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


def detect_events(df: pd.DataFrame, threshold: float, min_spacing: int, limit: int | None) -> list[dict[str, Any]]:
    df = df.sort_values("timestamp")
    timestamps = _normalise_timestamp(df["timestamp"])
    names, matrix = _normalise_models(df)

    events: list[dict[str, Any]] = []
    last_event_time = None
    spacing_ms = max(0, int(min_spacing)) * 1000

    for idx, ts in enumerate(timestamps):
        cp_prob = _safe_float(df.iloc[idx].get("cp_prob")) or 0.0
        if cp_prob < threshold:
            continue
        if pd.isna(ts):
            continue
        if last_event_time is not None and ts.timestamp() * 1000 - last_event_time < spacing_ms:
            continue
        last_event_time = ts.timestamp() * 1000
        runlen_expect = _safe_float(df.iloc[idx].get("runlen_expect")) or _safe_float(df.iloc[idx].get("expected_runlen"))
        event = {
            "timestamp": ts.isoformat(),
            "cp_prob": cp_prob,
            "runlen_expect": runlen_expect,
            "runlen_map": _safe_int(df.iloc[idx].get("runlen_map")),
        }
        if len(names) > 0:
            weights = {name: float(matrix[idx, col]) for col, name in enumerate(names)}
            event["models"] = weights
        events.append(event)
        if limit and len(events) >= limit:
            break
    return events


def window_slice(df: pd.DataFrame, center: str, seconds: int) -> pd.DataFrame:
    if not center:
        raise ValueError('center timestamp required')
    ts_center = pd.to_datetime(center, utc=True, errors='coerce')
    if ts_center is pd.NaT:
        raise ValueError('invalid center timestamp')
    timestamps = _normalise_timestamp(df["timestamp"])
    df = df.copy()
    df["__ts"] = timestamps
    window = max(1, int(seconds))
    start = ts_center - pd.Timedelta(seconds=window)
    end = ts_center + pd.Timedelta(seconds=window)
    return df[(df["__ts"] >= start) & (df["__ts"] <= end)].drop(columns=["__ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CPD context artefacts")
    parser.add_argument("--file", required=True, help="Path to cpd_ctx.parquet")
    parser.add_argument("--mode", choices=["series", "models", "summary", "events", "window"], required=True)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-spacing", type=int, default=45, dest="min_spacing")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--center", help="Center timestamp for window mode (ISO)")
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    if "timestamp" not in df.columns:
        raise SystemExit("cpd_ctx parquet missing timestamp column")

    downsample = max(1, int(args.downsample or 1))

    if args.mode == "series":
        data = extract_series(df, downsample)
    elif args.mode == "models":
        data = extract_models(df, downsample)
    elif args.mode == "summary":
        data = extract_summary(df)
    elif args.mode == "events":
        data = detect_events(df, threshold=float(args.threshold or 0), min_spacing=args.min_spacing or 0, limit=args.limit)
    else:
        window_df = window_slice(df, args.center, args.seconds)
        data = {
            "series": extract_series(window_df, downsample),
            "models": extract_models(window_df, downsample),
            "summary": extract_summary(window_df),
        }

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
