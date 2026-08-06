"""
schemas.py
Pandera schema definitions enforcing data-quality contracts on Silver-layer
datasets. Each schema is the "data quality gate" a real record must pass
before being promoted from Bronze to Silver. Failures are collected and
logged (not silently dropped) so pipeline operators can see exactly which
rows/columns broke the contract -- mirrors the "monitoring and alerting for
pipeline and data quality issues" requirement.
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

ag_exports_schema = DataFrameSchema(
    {
        "state_code": Column(str, Check.str_length(2, 2), nullable=False),
        "state_name": Column(str, nullable=False),
        "total_exports_musd": Column(float, Check.ge(0), nullable=False),
        "corn_musd": Column(float, Check.ge(0), nullable=False),
        "wheat_musd": Column(float, Check.ge(0), nullable=False),
        "cotton_musd": Column(float, Check.ge(0), nullable=False),
        "dairy_musd": Column(float, Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
)

geo_schema = DataFrameSchema(
    {
        "state_name": Column(str, nullable=False),
        "capital": Column(str, nullable=False),
        "latitude": Column(float, Check.in_range(-90, 90), nullable=False),
        "longitude": Column(float, Check.in_range(-180, 180), nullable=False),
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
