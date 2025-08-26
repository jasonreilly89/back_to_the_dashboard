# back_to_the_dashboard

A minimal Flask dashboard that parses training logs for trading models and
displays PRAUC and prevalence charts.

## Usage
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Write your training output to `train.log` in this directory.
3. Start the dashboard
   ```bash
   python dashboard.py
   ```
4. Open <http://localhost:5000/> to view updates. The page refreshes every five
   seconds by reading the latest contents of `train.log`.

Example log line expected by the parser:
```
[val 20220105] ep 01 tr:loss=0.6596 | va:APμ=0.6367 AP̄=0.3606 F1̄=0.7351 prev=0.02 sel=prauc:0.6367 (time=0.9s)
```

## Plotting metrics
The `scripts/plot_baseline.py` helper script generates matplotlib graphs
showing AP micro versus its prevalence baseline and the lift over that
baseline.  It reads metrics from the JSONL file used by the dashboard.

```bash
python scripts/plot_baseline.py
```

Set the `METRICS_PATH` environment variable if your metrics file lives at a
non‑default location.
