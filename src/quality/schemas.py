"""
schemas.py
Pandera schema definitions enforcing data-quality contracts on Silver-layer
datasets for the active country profile (AgriPulse India). Each schema is the
"data quality gate" a real record must pass before being promoted from Bronze
to Silver. Failures are collected and logged (not silently dropped) so pipeline
operators can see exactly which rows/columns broke the contract.

Contracts:
  * weather_schema          - daily Open-Meteo weather, metric units (deg C, mm, %, km/h)
  * geo_india_schema        - India geography reference: State/UT > District > Mandi/APMC
  * market_commodity_schema - e-NAM / AGMARKNET market rows (INR per quintal) -- only
                              validated when a live loader is configured; never fabricated
"""

import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

weather_schema = DataFrameSchema(
    {
        "region_name": Column(str, nullable=False),
        "date": Column(pa.DateTime, nullable=False),
        "temp_max_c": Column(float, Check.in_range(-40, 55), nullable=False),
        "temp_min_c": Column(float, Check.in_range(-50, 45), nullable=False),
        "precipitation_mm": Column(float, Check.ge(0), nullable=False),
        "humidity_pct": Column(float, Check.in_range(0, 100), nullable=False),
        "windspeed_max_kmh": Column(float, Check.ge(0), nullable=False),
    },
    checks=[
        Check(lambda df: df["temp_max_c"] >= df["temp_min_c"],
              error="temp_max_c must be >= temp_min_c"),
    ],
    strict=False,
    coerce=True,
)

# India geography: a monitored district with its market committee (mandi/APMC)
# used as the weather-API anchor point.
geo_india_schema = DataFrameSchema(
    {
        "region_name": Column(str, nullable=False),
        "state_ut": Column(str, nullable=False),
        "state_code": Column(str, Check.str_length(2, 2), nullable=False),
        "district": Column(str, nullable=False),
        "mandi_apmc": Column(str, nullable=False),
        "latitude": Column(float, Check.in_range(6, 37), nullable=False),
        "longitude": Column(float, Check.in_range(68, 98), nullable=False),
    },
    strict=False,
    coerce=True,
)

# e-NAM / AGMARKNET market-rows contract (INR per quintal). Only enforced when a
# real loader is configured; this schema is the documented target shape.
market_commodity_schema = DataFrameSchema(
    {
        "commodity": Column(str, nullable=False),
        "variety": Column(str, nullable=True),
        "state_ut": Column(str, nullable=False),
        "district": Column(str, nullable=False),
        "mandi": Column(str, nullable=False),
        "date": Column(pa.DateTime, nullable=False),
        "market_price_inr_per_quintal": Column(float, Check.ge(0), nullable=False),
        "arrivals_quintal": Column(float, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_or_report(df, schema: DataFrameSchema, dataset_name: str) -> tuple:
    """
    Validates a dataframe against a schema. Returns (clean_df, error_report).
    Uses lazy validation to collect ALL failures at once (not fail-fast),
    which is what you want for a monitoring/alerting dashboard.
    """
    try:
        validated = schema.validate(df, lazy=True)
        return validated, None
    except pa.errors.SchemaErrors as e:
        error_report = {
            "dataset": dataset_name,
            "failure_count": len(e.failure_cases),
            "failures": e.failure_cases.to_dict(orient="records"),
        }
        return None, error_report