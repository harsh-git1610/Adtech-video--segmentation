#!/usr/bin/env python3
"""
validate_schemas.py
-------------------
CLI utility that loads every JSON file in data/ and validates it against
the matching JSON Schema in schemas/.

Usage:
    python scripts/validate_schemas.py

Exit codes:
    0  all validations passed
    1  one or more validation failures (details printed to stderr)

Relies on: jsonschema >= 4.18 (draft 2020-12 support)
"""

import json
import sys
from pathlib import Path

try:
    from jsonschema.validators import validator_for
except ImportError:
    sys.exit(
        "ERROR: 'jsonschema' package not found. "
        "Install it with:  pip install 'jsonschema[format-nongpl]'"
    )

# ── Resolve paths relative to the repo root (one level above scripts/) ────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"
DATA_DIR = REPO_ROOT / "data"

# Map: data filename -> (schema filename, top-level structure type)
# "array" means the data file is a JSON array and each element is validated
# individually against the schema. "object" means the file is a single object.
VALIDATION_TARGETS = {
    "brand_catalogue.json": ("brand.schema.json", "array"),
    # Future additions as pipeline stages are built:
    # "scenes.json":           ("scene.schema.json",          "array"),
    # "break_candidates.json": ("break_candidate.schema.json", "array"),
}


def load_json(path: Path):
    """Load and parse a JSON file, exiting on decode errors."""
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: Failed to parse JSON at {path}: {exc}")


def build_validator(schema: dict):
    """Return a validator instance appropriate for the schema's declared draft."""
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_item(item, validator, label: str) -> list:
    """Validate a single item against a validator. Returns list of error messages."""
    errors = []
    for err in validator.iter_errors(item):
        path = " -> ".join(str(p) for p in err.absolute_path) or "(root)"
        errors.append(f"  [{label}] {path}: {err.message}")
    return errors


def main() -> int:
    all_passed = True

    for data_filename, (schema_filename, structure) in VALIDATION_TARGETS.items():
        data_path = DATA_DIR / data_filename
        schema_path = SCHEMAS_DIR / schema_filename

        if not data_path.exists():
            print(f"SKIP  {data_filename} - file not found at {data_path}", file=sys.stderr)
            continue

        if not schema_path.exists():
            print(
                f"ERROR {data_filename} - schema not found at {schema_path}",
                file=sys.stderr,
            )
            all_passed = False
            continue

        schema = load_json(schema_path)
        data = load_json(data_path)
        validator = build_validator(schema)

        if structure == "array":
            if not isinstance(data, list):
                print(
                    f"ERROR {data_filename} - expected a JSON array at top level",
                    file=sys.stderr,
                )
                all_passed = False
                continue

            file_errors = []
            for idx, item in enumerate(data):
                item_id = item.get("brand_id") or item.get("scene_id") or f"index_{idx}"
                errs = validate_item(item, validator, label=item_id)
                file_errors.extend(errs)

            if file_errors:
                print(
                    f"FAIL  {data_filename} ({len(file_errors)} error(s)):",
                    file=sys.stderr,
                )
                for msg in file_errors:
                    print(msg, file=sys.stderr)
                all_passed = False
            else:
                print(
                    f"PASS  {data_filename} - {len(data)} item(s) validated "
                    f"against {schema_filename}"
                )
        else:
            errs = validate_item(data, validator, label=data_filename)
            if errs:
                print(f"FAIL  {data_filename}:", file=sys.stderr)
                for msg in errs:
                    print(msg, file=sys.stderr)
                all_passed = False
            else:
                print(f"PASS  {data_filename} - validated against {schema_filename}")

    if not all_passed:
        print(
            "\nValidation FAILED - fix errors above before proceeding.",
            file=sys.stderr,
        )
        return 1

    print("\nAll validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
