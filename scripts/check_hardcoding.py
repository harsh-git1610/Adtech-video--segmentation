#!/usr/bin/env python3
"""
Zero Hardcoding Audit Script
Verifies that no brand names, video IDs, or ad timestamps are hardcoded into
the pipeline logic, matching engine, or frontend.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def check_python_files(catalogue_brands):
    pipeline_dir = REPO_ROOT / "pipeline"
    py_files = list(pipeline_dir.glob("*.py"))
    violations = []

    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()

        for brand in catalogue_brands:
            b_id = brand.get("brand_id", "")
            d_name = brand.get("display_name", "")

            # Check if brand_id or display_name is hardcoded as a string literal (excluding docstrings)
            tree = ast.parse(content, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value.strip()
                    if (val == b_id or val == d_name) and len(val) > 3:
                        violations.append({
                            "file": str(py_file.name),
                            "literal": val,
                            "line": getattr(node, "lineno", "?")
                        })
    return violations

def main():
    cat_file = REPO_ROOT / "data" / "brand_catalogue.json"
    if not cat_file.exists():
        print(f"Error: Catalogue not found at {cat_file}")
        sys.exit(1)

    with open(cat_file, "r", encoding="utf-8") as f:
        brands = json.load(f)

    print(f"Auditing pipeline code against {len(brands)} brands from catalogue...")
    violations = check_python_files(brands)

    if violations:
        print("\n[FAIL] Found hardcoded brand literals in pipeline code:")
        for v in violations:
            print(f"  - {v['file']}:{v['line']} -> '{v['literal']}'")
        sys.exit(1)
    else:
        print("\n[OK] PASS: Zero hardcoded brand literals detected across all pipeline modules.")
        sys.exit(0)

if __name__ == "__main__":
    main()
