"""Smoke checks for the synthetic SAR range fixture.

These tests ensure the canonical Parquet asset can be loaded with DuckDB and
that basic probabilistic invariants hold. They protect the Dockerised test
environment by failing fast when required dependencies or fixtures are
missing inside the container.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sar_range_final_synthetic.parquet"


def test_fixture_exists() -> None:
    """Verify the synthetic Parquet fixture is available to the test runner."""
    assert FIXTURE_PATH.exists(), "Synthetic fixture sar_range_final_synthetic.parquet not found"


def test_probability_mass_is_preserved() -> None:
    """The classifier probabilities should add up to ~1 for every row."""
    con = duckdb.connect(database=":memory:")
    try:
        totals = con.execute(
            """
            SELECT prob_range_below + prob_range_in + prob_range_above AS total
            FROM read_parquet(?)
            """,
            [str(FIXTURE_PATH)],
        ).fetchall()
    finally:
        con.close()

    assert totals, "Fixture Parquet file yielded no records"
    for (value,) in totals:
        assert abs(value - 1.0) < 1e-6
