"""
monitor.py
Basic pipeline monitoring: checks data freshness (is Bronze data recent
enough?) and row-count anomalies (did a source suddenly return far fewer
rows than expected?). Mirrors the JD's "Support the implementation of
basic monitoring and alerting for pipeline and data quality issues."

In a real deployment this would push to Slack/PagerDuty/email; here it
writes a structured JSON health report and exits non-zero on failure so
it can gate an Airflow DAG or CI pipeline.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("monitor")

BASE = Path(__file__).resolve().parents[2]
BRONZE = BASE / "data" / "bronze"
SILVER = BASE / "data" / "silver"
CATALOG = BASE / "catalog"

# Minimum expected row counts -- catches a silently-broken upstream source
EXPECTED_MIN_ROWS = {
    "weather": 300,      # 10 regions * ~37 days
    "ag_exports": 45,    # ~50 US states minus territories
    "geo": 45,
}

FRESHNESS_MAX_AGE_HOURS = 24


def check_freshness() -> list:
    issues = []
    meta_files = sorted(BRONZE.glob("*.meta.json"))
    if not meta_files:
        issues.append({"check": "freshness", "status": "FAIL", "detail": "No ingestion metadata found in bronze layer"})
        return issues

    latest_meta = json.loads(meta_files[-1].read_text())
    ingested_at = datetime.strptime(latest_meta["ingested_at_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - ingested_at).total_seconds() / 3600

    if age_hours > FRESHNESS_MAX_AGE_HOURS:
        issues.append({
            "check": "freshness", "status": "FAIL",
            "detail": f"Latest weather ingestion is {age_hours:.1f}h old (max allowed {FRESHNESS_MAX_AGE_HOURS}h)"
        })
    else:
        logger.info(f"Freshness OK: latest ingestion {age_hours:.1f}h ago")
    return issues


def check_row_counts() -> list:
    issues = []
    for name, min_rows in EXPECTED_MIN_ROWS.items():
        path = SILVER / f"{name}.parquet"
        if not path.exists():
            issues.append({"check": "row_count", "dataset": name, "status": "FAIL", "detail": "Silver file missing"})
            continue
        df = pd.read_parquet(path)
        if len(df) < min_rows:
            issues.append({
                "check": "row_count", "dataset": name, "status": "FAIL",
                "detail": f"{len(df)} rows < expected minimum {min_rows}"
            })
        else:
            logger.info(f"Row count OK for {name}: {len(df)} rows (min {min_rows})")
    return issues


def main():
    all_issues = check_freshness() + check_row_counts()

    CATALOG.mkdir(parents=True, exist_ok=True)
    report_path = CATALOG / "health_report.json"
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if all_issues else "PASS",
        "issues": all_issues,
    }
    report_path.write_text(json.dumps(report, indent=2))

    if all_issues:
        logger.error(f"{len(all_issues)} monitoring issue(s) found -- see {report_path}")
        for issue in all_issues:
            logger.error(f"  - {issue}")
        sys.exit(1)
    else:
        logger.info(f"All monitoring checks passed -- report written to {report_path}")


if __name__ == "__main__":
    main()
