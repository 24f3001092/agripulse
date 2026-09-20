"""
india_schemas.py

Pandera contracts for the India agriculture pipeline (Bronze -> Silver gate).

Validates the canonical DE&S/MoAFW crop Area-Production-Yield frame covering
the requested dimensions: State, District, Crop, Season, Year, Area (Hectare),
Production (Tonnes), Yield (Tonnes/Hectare).

Rules:
  * Only columns actually present in the source are used (the normalized frame
    is built from observed source columns -- see india_agriculture.normalize).
  * Missing VALUES are allowed only where the source reports them (production
    and yield may be null); area and every key dimension are required.
  * Numeric fields must be non-negative.
  * Duplicate handling is season-aware: the same state/district/crop/year is a
    legitimate record per season (e.g. Kharif + Rabi + Total), so the dedup key
    includes season; exact-row duplicates are removed before validation.

mandi_price_schema
  Contract for AGMARKNET daily APMC price/arrival records (Bronze -> Silver
  gate for the mandi market-intelligence pipeline). Only fields actually
  available in the source are retained -- AGMARKNET does not report a traded
  quantity, so no traded_quantity column exists in the contract.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

india_agriculture_schema = DataFrameSchema(
    {
        "id": Column(int, nullable=False),
        "year": Column(str, nullable=False),
        "year_start": Column(int, Check.ge(1900), nullable=True),
        "state": Column(str, nullable=False),
        "state_code": Column("Int64", Check.ge(0), nullable=True),
        "district": Column(str, nullable=False),
        "district_code": Column("Int64", Check.ge(0), nullable=True),
        "crop": Column(str, nullable=False),
        "crop_type": Column(str, nullable=True),
        "season": Column(str, nullable=False),
        "area": Column(float, Check.ge(0), nullable=False),
        "production": Column(float, Check.ge(0), nullable=True),
        "yield": Column(float, Check.ge(0), nullable=True),
    },
    checks=[
        Check(
            lambda df: df["production"].notna() | df["yield"].notna() | df["area"].notna(),
            error="a row must carry area, production, or yield",
        ),
        Check(
            lambda df: df["season"].str.strip().str.len() > 0,
            error="season must be non-empty when present",
        ),
    ],
    strict=False,  # tolerate source passthrough columns, only validate present ones
    coerce=True,
)


def validate_agriculture(df):
    """
    Lazy-validate a normalized agriculture frame against the contract.

    Returns (clean_df, error_report): `error_report` is None when the frame
    passes, else a dict with failure details (matching schemas.validate_or_report).
    """
    try:
        validated = india_agriculture_schema.validate(df, lazy=True)
        return validated, None
    except pa.errors.SchemaErrors as e:
        report = {
            "dataset": "india_agriculture",
            "failure_count": len(e.failure_cases),
            "failure_cases": e.failure_cases.to_dict(orient="records"),
        }
        return df, report


# --------------------------------------------------------------------------- #
# Mandi market intelligence (AGMARKNET APMC price/arrival records)
# --------------------------------------------------------------------------- #
mandi_price_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "state": Column(str, nullable=False),
        "state_code": Column(str, nullable=True),
        "district": Column(str, nullable=False),
        "district_code": Column(str, nullable=True),
        "apmc": Column(str, nullable=False),
        "market_center_code": Column(str, nullable=True),
        "commodity": Column(str, nullable=False),
        "commodity_id": Column(str, nullable=True),
        "variety": Column(str, nullable=False),
        "grade": Column(str, nullable=True),
        "min_price": Column(float, Check.ge(0), nullable=True),
        "modal_price": Column(float, Check.ge(0), nullable=True),
        "max_price": Column(float, Check.ge(0), nullable=True),
        "price_unit": Column(str, nullable=True),
        "arrival_quantity": Column(float, Check.ge(0), nullable=True),
        "arrival_units": Column(str, nullable=True),
    },
    checks=[
        Check(
            lambda df: df["commodity"].astype("string").str.strip().ne(""),
            error="commodity must be a non-empty string",
        ),
        Check(
            lambda df: df["apmc"].astype("string").str.strip().ne(""),
            error="apmc must be a non-empty string",
        ),
        Check(
            lambda df: (df["min_price"].isna() | df["modal_price"].isna() |
                        (df["min_price"] <= df["modal_price"])),
            error="min_price must be <= modal_price when both present",
        ),
        Check(
            lambda df: (df["modal_price"].isna() | df["max_price"].isna() |
                        (df["modal_price"] <= df["max_price"])),
            error="modal_price must be <= max_price when both present",
        ),
    ],
    strict=False,
    coerce=True,
)


def validate_mandi(df):
    """Lazy-validate a normalized mandi frame against the market contract."""
    try:
        validated = mandi_price_schema.validate(df, lazy=True)
        return validated, None
    except pa.errors.SchemaErrors as e:
        report = {
            "dataset": "india_mandi",
            "failure_count": len(e.failure_cases),
            "failure_cases": e.failure_cases.to_dict(orient="records"),
        }
        return df, report