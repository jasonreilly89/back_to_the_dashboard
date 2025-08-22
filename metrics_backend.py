import json
import os
import statistics
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from flask import Blueprint, jsonify, render_template, request

DEFAULT_METRICS_PATH = "/home/jason/ml/datasets/models/prepped_setup12_60_setup/metrics.jsonl"
DEFAULT_TITLE = "Walk-Forward PRAUC"

metrics_bp = Blueprint("metrics_bp", __name__)


def get_metrics_path() -> Path:
    return Path(os.environ.get("METRICS_PATH", DEFAULT_METRICS_PATH))


def tail_lines(path: Path, limit: int) -> Iterable[str]:
    """Yield lines from the end of file efficiently."""
    if limit <= 0:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                yield line
        return
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        size = 1024
        data = bytearray()
        while end > 0 and data.count(b"\n") <= limit:
            read_size = min(size, end)
            f.seek(end - read_size)
            chunk = f.read(read_size)
            data[:0] = chunk
            end -= read_size
            if end == 0:
                break
        lines = data.splitlines()
        if len(lines) > limit:
            lines = lines[-limit:]
        for line in lines:
            yield line.decode("utf-8", errors="ignore")


def parse_metrics(limit: int, since_day: Optional[str]) -> List[Dict[str, Any]]:
    path = get_metrics_path()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in tail_lines(path, limit):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        day = obj.get("day")
        if isinstance(day, int):
            day = f"{day:08d}"
        else:
            day = str(day)
        if since_day and day < since_day:
            continue
        row = {
            "day": day,
            "epoch": int(obj.get("epoch", 0)),
            "train_loss": float(obj.get("train_loss", 0.0)),
            "time_s": float(obj.get("time_s", 0.0)),
            "ap_micro": float(obj.get("ap_micro", 0.0)),
            "ap_macro": float(obj.get("ap_macro", 0.0)),
            "prev_micro": float(obj.get("prev_micro", 0.0)),
            "prev_macro": float(obj.get("prev_macro", 0.0)),
            "pos_total": int(obj.get("pos_total", 0)),
            "neg_total": int(obj.get("neg_total", 0)),
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["day"], r["epoch"]))
    return rows


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        day = r["day"]
        if day not in best:
            best[day] = r
        else:
            b = best[day]
            if r["ap_micro"] > b["ap_micro"] or (
                r["ap_micro"] == b["ap_micro"] and r["epoch"] < b["epoch"]
            ):
                best[day] = r
    return [best[d] for d in sorted(best.keys())]


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count_rows": 0,
            "days": 0,
            "ap_micro_mean": 0,
            "ap_micro_median": 0,
            "prev_micro_mean": 0,
        }
    ap_values = [r["ap_micro"] for r in rows]
    prev_values = [r["prev_micro"] for r in rows]
    return {
        "count_rows": len(rows),
        "days": len({r["day"] for r in rows}),
        "ap_micro_mean": statistics.mean(ap_values),
        "ap_micro_median": statistics.median(ap_values),
        "prev_micro_mean": statistics.mean(prev_values) if prev_values else 0,
    }


@metrics_bp.route("/api/metrics")
def api_metrics():
    limit = int(request.args.get("limit", 0))
    since_day = request.args.get("since_day")
    if since_day is not None:
        since_day = str(since_day)
        if since_day.isdigit():
            since_day = f"{int(since_day):08d}"
    aggregate = request.args.get("aggregate", "true").lower() != "false"

    rows = parse_metrics(limit, since_day)
    summary = compute_summary(rows)
    best = aggregate_rows(rows) if aggregate else []

    return jsonify({"summary": summary, "best_per_day": best, "rows": rows})


@metrics_bp.route("/dashboard/metrics")
def dashboard():
    title = os.environ.get("DASHBOARD_TITLE", DEFAULT_TITLE)
    return render_template("metrics.html", title=title)
