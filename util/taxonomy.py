"""Reads files/taxonomy.yaml, which says which classes belong together."""
from pathlib import Path

import yaml

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "files" / "taxonomy.yaml"


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    """The two groupings, each empty when the file does not define it."""
    data = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
    data = data or {}
    return {
        "smoothing_groups": data.get("smoothing_groups") or [],
        "report_groups": data.get("report_groups") or {},
    }


def species_to_group(report_groups: dict[str, list[str]]) -> dict[str, str]:
    """Inverts report_groups into a class name -> group name mapping."""
    return {name: group for group, names in report_groups.items() for name in names}