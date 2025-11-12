"""Lightweight helpers to resolve local STAC assets.

The wind-resource toolkit treats STAC collections as the single source of
truth for locating GeoParquet assets. This module provides a minimal reader
that works offline, discovers items declared in ``collection.json`` and
returns paths to their assets. The helpers are intentionally simple: they
avoid any third-party dependency so they can run inside restricted
environments (e.g. Docker containers without network access).

The default configuration is stored in :mod:`config/stac_catalogs.json` and
can be overridden per deployment to point at alternate dataset versions or
mirror locations. When the configuration file is missing, the helper falls
back to the repository snapshot archived under
``use_case/catalogs/sar_range_final_pivots_joined/collection.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple
from urllib.parse import urlparse

__all__ = [
    "ResolvedStacAsset",
    "StacCatalogEntry",
    "load_catalog_index",
    "resolve_catalog_asset",
    "resolve_ann_asset",
]

_DEFAULT_INDEX_PATH = Path("config") / "stac_catalogs.json"
_DEFAULT_DATASET_KEY = "sar_range_final_pivots_joined"
_DEFAULT_COLLECTION_PATH = Path("use_case") / "catalogs" / "sar_range_final_pivots_joined" / "collection.json"
_DEFAULT_ITEM_ID = "data"
_DEFAULT_ASSET_KEY = "data"


@dataclass(frozen=True)
class StacCatalogEntry:
    """Declarative entry describing how to locate a dataset within the index."""

    collection: Path
    default_item: str
    default_asset: str


@dataclass(frozen=True)
class ResolvedStacAsset:
    """Fully resolved STAC asset pointing to a local or remote resource."""

    dataset: str
    collection_path: Path
    item_path: Path
    item_id: str
    asset_key: str
    href: str
    path: Path | None

    def require_local_path(self) -> Path:
        """Return the asset as a local path raising on remote references."""

        if self.path is None:
            raise ValueError(
                f"Asset {self.asset_key!r} from item {self.item_id!r} resolves to a non-local href: {self.href}"
            )
        return self.path


def load_catalog_index(path: str | Path | None = None) -> Mapping[str, StacCatalogEntry]:
    """Load the STAC catalog index from JSON returning a mapping by dataset key."""

    target_path = Path(path) if path is not None else _DEFAULT_INDEX_PATH
    entries: Dict[str, StacCatalogEntry] = {}

    if target_path.exists():
        try:
            payload = json.loads(target_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in STAC catalog index: {target_path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("STAC catalog index must be a JSON object mapping dataset keys to entries.")

        for dataset_key, raw_entry in payload.items():
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"Catalog entry for {dataset_key!r} must be a JSON object.")

            collection = raw_entry.get("collection") or raw_entry.get("collection_path")
            if collection is None:
                raise ValueError(f"Catalog entry for {dataset_key!r} is missing the 'collection' attribute.")

            default_item = str(
                raw_entry.get("default_item", raw_entry.get("item", _DEFAULT_ITEM_ID))
            )
            default_asset = str(
                raw_entry.get("default_asset", raw_entry.get("asset", _DEFAULT_ASSET_KEY))
            )

            entries[str(dataset_key)] = StacCatalogEntry(
                collection=Path(collection),
                default_item=default_item,
                default_asset=default_asset,
            )
    else:
        entries[_DEFAULT_DATASET_KEY] = StacCatalogEntry(
            collection=_DEFAULT_COLLECTION_PATH,
            default_item=_DEFAULT_ITEM_ID,
            default_asset=_DEFAULT_ASSET_KEY,
        )
        return entries

    if _DEFAULT_DATASET_KEY not in entries:
        entries[_DEFAULT_DATASET_KEY] = StacCatalogEntry(
            collection=_DEFAULT_COLLECTION_PATH,
            default_item=_DEFAULT_ITEM_ID,
            default_asset=_DEFAULT_ASSET_KEY,
        )

    return entries


def resolve_catalog_asset(
    dataset: str,
    *,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    item_id: str | None = None,
    asset_key: str | None = None,
) -> ResolvedStacAsset:
    """Resolve *dataset* using the catalog index and return the matching asset."""

    entries = load_catalog_index(config_path)
    try:
        entry = entries[dataset]
    except KeyError as exc:
        raise KeyError(f"Dataset {dataset!r} not defined in STAC catalog index {config_path or _DEFAULT_INDEX_PATH}") from exc

    root_path = Path(root) if root is not None else Path.cwd()
    collection = entry.collection
    if not collection.is_absolute():
        collection = (root_path / collection).resolve()

    desired_item = item_id or entry.default_item
    desired_asset = asset_key or entry.default_asset

    collection_payload = _load_json(collection)
    item_payload, item_path = _resolve_item(collection_payload, collection.parent, desired_item)
    asset_href = _extract_asset_href(item_payload, desired_asset)
    asset_path = _href_to_path(asset_href, item_path.parent)

    return ResolvedStacAsset(
        dataset=dataset,
        collection_path=collection,
        item_path=item_path,
        item_id=desired_item,
        asset_key=desired_asset,
        href=asset_href,
        path=asset_path,
    )


def resolve_ann_asset(
    *,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    item_id: str | None = None,
    asset_key: str | None = None,
) -> ResolvedStacAsset:
    """Shortcut for resolving the SAR range-aware ANN dataset asset."""

    return resolve_catalog_asset(
        _DEFAULT_DATASET_KEY,
        config_path=config_path,
        root=root,
        item_id=item_id,
        asset_key=asset_key,
    )


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"STAC file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot decode STAC JSON payload: {path}") from exc


def _resolve_item(
    collection_payload: Mapping[str, object],
    base_dir: Path,
    item_id: str,
) -> Tuple[Mapping[str, object], Path]:
    links = collection_payload.get("links", [])
    if isinstance(links, list):
        for raw_link in links:
            if not isinstance(raw_link, Mapping):
                continue
            if raw_link.get("rel") not in {"item", "child"}:
                continue
            href = raw_link.get("href")
            if not isinstance(href, str):
                continue
            candidate = (base_dir / href).resolve()
            if not candidate.exists():
                continue
            payload = _load_json(candidate)
            if payload.get("id") == item_id:
                return payload, candidate

    fallback = base_dir / "items" / f"{item_id}.json"
    if fallback.exists():
        payload = _load_json(fallback)
        if payload.get("id") == item_id:
            return payload, fallback.resolve()

    raise KeyError(f"Item {item_id!r} not found within STAC collection at {base_dir}")


def _extract_asset_href(item_payload: Mapping[str, object], asset_key: str) -> str:
    assets = item_payload.get("assets")
    if not isinstance(assets, Mapping):
        raise KeyError(f"Item {item_payload.get('id')!r} does not declare any assets.")

    asset = assets.get(asset_key)
    if not isinstance(asset, Mapping):
        raise KeyError(f"Asset {asset_key!r} not present in item {item_payload.get('id')!r}.")

    href = asset.get("href")
    if not isinstance(href, str):
        raise ValueError(f"Asset {asset_key!r} is missing a valid 'href'.")
    return href


def _href_to_path(href: str, base_dir: Path) -> Path | None:
    parsed = urlparse(href)
    if parsed.scheme and parsed.scheme not in {"file"}:
        return None

    if parsed.scheme == "file":
        candidate = Path(parsed.path)
        return candidate.resolve()

    candidate = Path(href)
    if candidate.is_absolute():
        return candidate.resolve()

    return (base_dir / candidate).resolve()
