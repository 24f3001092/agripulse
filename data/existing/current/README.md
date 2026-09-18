# data/existing/current — transition snapshot (do not delete)

This directory is the **on-disk snapshot** of the previous U.S.-focused
AgriPulse build, captured when the India migration started. It exists so the
previous implementation remains recoverable during the migration:

| Here | Was originally |
|------|----------------|
| `catalog/regions_us.json` | `catalog/regions.json` (10 U.S. states) |
| `catalog/data_catalog_us.json` | `catalog/data_catalog.json` (U.S. source metadata) |
| `bronze/` | `data/bronze/` (Open-Meteo U.S. weather pulls + U.S. market/geo CSVs) |
| `silver/` | `data/silver/` (validated U.S. parquet tables) |
| `gold/` | `data/gold/` (Delta/parquet U.S. feature + summary tables) |
| `demo/` | `data/demo/` (tracked U.S. deployment fallback) |
| `catalog/` | `catalog/health_report.json`, `model_report.json`, `prediction_run.json` |

None of these files are consumed by the active (India) pipeline. They are kept
entire for recovery — the full previous build is also reachable via `git`
history and the pre-migration branches.

- Active pipeline data tree: `data/india/`
- Country & pipeline configuration: `catalog/pipeline_config.json`, `catalog/india/`

Do not delete this directory; it documents exactly what the previous U.S.
build looked like.