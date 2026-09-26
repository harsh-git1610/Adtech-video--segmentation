import os
import json
from pathlib import Path

PIPELINE_ROOT = Path(r"F:\projects\adtech-video-pipeline").resolve()
schema_file = PIPELINE_ROOT / "schemas" / "break_candidate.schema.json"

if schema_file.exists():
    with open(schema_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Schema Title:", data.get("title"))
    print("Required fields:", data.get("required"))
    print("Properties:", list(data.get("properties", {}).keys()))
    with open(r"f:\projects\hoichoi\break_candidate_schema_copy.json", "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2)
else:
    print("Schema file not found:", schema_file)
