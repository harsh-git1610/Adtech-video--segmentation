#!/usr/bin/env bash
# check_no_hardcoding.sh — CI gate for Phase 4 brand-name literals.
#
# Scans pipeline/*.py for any string literals that match brand_ids or display_names
# from data/brand_catalogue.json. Exits with code 1 if any hardcoded brands found.
#
# Usage: bash scripts/check_no_hardcoding.sh [catalogue_path]

set -euo pipefail

CATALOGUE="${1:-data/brand_catalogue.json}"
PIPELINE_DIR="pipeline"
EXIT_CODE=0

if [ ! -f "$CATALOGUE" ]; then
    echo "ERROR: Catalogue not found at $CATALOGUE"
    exit 1
fi

# Extract brand_id and display_name values from JSON using python (cross-platform)
BRAND_TERMS=$(python -c "
import json, sys
with open('$CATALOGUE') as f:
    brands = json.load(f)
terms = set()
for b in brands:
    terms.add(b['brand_id'])
    # Also add display_name without spaces (as a search term)
    terms.add(b.get('display_name', '').replace(' ', ''))
print('\n'.join(t.lower() for t in terms if t))
")

echo "=== Checking $PIPELINE_DIR for hardcoded brand identifiers ==="
echo "Source: $CATALOGUE"
echo ""

while IFS= read -r term; do
    [ -z "$term" ] && continue
    matches=$(grep -r -i --include="*.py" -l "$term" "$PIPELINE_DIR" 2>/dev/null || true)
    if [ -n "$matches" ]; then
        echo "FAIL: Hardcoded brand term '$term' found in:"
        echo "$matches" | sed 's/^/  /'
        EXIT_CODE=1
    fi
done <<< "$BRAND_TERMS"

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "PASS: No hardcoded brand names found in $PIPELINE_DIR"
else
    echo ""
    echo "FAIL: Hardcoded brand references violate 'brands are data' contract."
    echo "Move all brand-specific logic to $CATALOGUE"
fi

exit $EXIT_CODE
