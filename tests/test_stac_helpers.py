from __future__ import annotations

import json
from pathlib import Path

from hf_wind_resource.io import (
    ResolvedStacAsset,
    load_catalog_index,
    resolve_ann_asset,
    resolve_catalog_asset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_KEY = "sar_range_final_pivots_joined"
EXPECTED_COLLECTION = Path("catalogs") / "sar_range_final_pivots_joined" / "collection.json"
EXPECTED_ASSET = Path("catalogs") / "sar_range_final_pivots_joined" / "assets" / "data.parquet"


def test_resolve_ann_asset_returns_local_path() -> None:
    """The helper should resolve the ANN dataset to the GeoParquet asset."""

    resolved = resolve_ann_asset(root=REPO_ROOT)
    assert isinstance(resolved, ResolvedStacAsset)
    assert resolved.item_id == "data"
    assert resolved.asset_key == "data"
    assert resolved.require_local_path() == (REPO_ROOT / EXPECTED_ASSET).resolve()


def test_load_catalog_index_without_file_uses_default(tmp_path: Path) -> None:
    """Missing index files fall back to the baked-in dataset mapping."""

    index = load_catalog_index(tmp_path / "missing.json")
    assert DEFAULT_DATASET_KEY in index
    entry = index[DEFAULT_DATASET_KEY]
    assert entry.collection == EXPECTED_COLLECTION
    assert entry.default_item == "data"
    assert entry.default_asset == "data"


def test_resolve_catalog_asset_uses_custom_dataset(tmp_path: Path) -> None:
    """Custom dataset keys defined in a config file should be honoured."""

    config_path = tmp_path / "stac_catalogs.json"
    payload = {
        "custom_dataset": {
            "collection": EXPECTED_COLLECTION.as_posix(),
            "default_item": "data",
            "default_asset": "data",
        }
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_catalog_asset(
        "custom_dataset",
        config_path=config_path,
        root=REPO_ROOT,
    )
    assert resolved.require_local_path() == (REPO_ROOT / EXPECTED_ASSET).resolve()
