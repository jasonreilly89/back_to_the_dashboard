from __future__ import annotations

import re
from pathlib import Path
from flask import Flask, jsonify, render_template

LOG_FILE = Path("train.log")

app = Flask(__name__)

LOG_RE = re.compile(
    r"\[val (\d+)\] ep (\d+) tr:loss=(\d+\.\d+) \| va:APμ=(\d+\.\d+) AP̄=(\d+\.\d+) F1̄=(\d+\.\d+)(.*?)sel=prauc:(\d+\.\d+)"
)


def parse_logs():
    """Parse the log file into structured entries."""
    entries = []
    if not LOG_FILE.exists():
        return entries
    with LOG_FILE.open() as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                date, epoch, loss, ap_mu, ap_bar, f1_bar, extras, prauc = m.groups()
                prevalence = None
                lift_abs = None
                if extras:
                    m_prev = re.search(r"prev=(\d+\.\d+)", extras)
                    if m_prev:
                        prevalence = float(m_prev.group(1))
                    m_lift = re.search(r"lift_abs=(\d+\.\d+)", extras)
                    if m_lift:
                        lift_abs = float(m_lift.group(1))
                entries.append(
                    {
                        "date": date,
                        "epoch": int(epoch),
                        "train_loss": float(loss),
                        "ap_mu": float(ap_mu),
                        "ap_bar": float(ap_bar),
                        "f1_bar": float(f1_bar),
                        "prevalence": prevalence,
                        "lift_abs": lift_abs,
                        "prauc": float(prauc),
                    }
                )
    return entries


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def data():
    return jsonify(parse_logs())


if __name__ == "__main__":
    # Debug server for local development.
    app.run(debug=True, port=5000)
