"""
VMAP 1.0.1 Manifest Builder (Phase 5)

Generates IAB VMAP 1.0.1 compliant ad playlist manifests from approved break candidates.
Each approved ad break includes an inline VAST 3.0 payload pointing to the matched brand's
creative asset and duration.

Validates output against schemas/vmap-1.0.1.xsd using lxml.XMLSchema before returning.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import jinja2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XSD_PATH = REPO_ROOT / "schemas" / "vmap-1.0.1.xsd"

VMAP_TEMPLATE_STRING = """<?xml version="1.0" encoding="utf-8"?>
<vmap:VMAP xmlns:vmap="http://www.iab.net/videosuite/vmap" version="1.0">
  {% for brk in breaks %}
  <vmap:AdBreak breakType="linear" breakId="{{ brk.break_id }}" timeOffset="{{ brk.time_offset }}">
    <vmap:AdSource id="{{ brk.ad_source_id }}" allowMultipleAds="false" followRedirects="true">
      <vmap:VASTAdData>
        <VAST version="3.0" xmlns="http://www.iab.net/videosuite/vast">
          <Ad id="ad_{{ brk.break_id }}">
            <InLine>
              <AdSystem version="1.0">AdTech Video Pipeline</AdSystem>
              <AdTitle>{{ brk.brand_display_name }}</AdTitle>
              <Description>Contextual Mid-Roll Placement</Description>
              <Creatives>
                <Creative id="cr_{{ brk.break_id }}">
                  <Linear>
                    <Duration>{{ brk.duration_hms }}</Duration>
                    <MediaFiles>
                      <MediaFile delivery="progressive" type="video/mp4" width="1920" height="1080">
                        <![CDATA[{{ brk.creative_asset }}]]>
                      </MediaFile>
                    </MediaFiles>
                  </Linear>
                </Creative>
              </Creatives>
            </InLine>
          </Ad>
        </VAST>
      </vmap:VASTAdData>
    </vmap:AdSource>
  </vmap:AdBreak>
  {% endfor %}
</vmap:VMAP>
"""


def format_timestamp_vmap(seconds: float) -> str:
    """
    Format seconds into VMAP timeOffset string: HH:MM:SS.mmm
    Example: 125.45 -> '00:02:05.450'
    """
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3600000
    remainder = total_ms % 3600000
    minutes = remainder // 60000
    remainder %= 60000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_duration_hms(seconds: float) -> str:
    """Format ad duration seconds into HH:MM:SS format."""
    total_secs = int(round(seconds))
    hours = total_secs // 3600
    remainder = total_secs % 3600
    minutes = remainder // 60
    secs = remainder % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_vmap(xml_path: str, xsd_path: Optional[str] = None) -> bool:
    """
    Validate a VMAP XML document against the IAB VMAP 1.0.1 XSD using lxml.
    Raises ValueError or lxml.etree.DocumentInvalid with error details if invalid.
    """
    schema_file = Path(xsd_path) if xsd_path else DEFAULT_XSD_PATH
    if not schema_file.exists():
        raise FileNotFoundError(f"VMAP XSD schema not found at {schema_file}")

    try:
        from lxml import etree
    except ImportError:
        # Fallback if lxml is not installed: enforce required VMAP 1.0.1 schema elements & attributes
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            if not root.tag.endswith("VMAP"):
                raise ValueError(f"Root element must be VMAP, found {root.tag}")
            if "version" not in root.attrib:
                raise ValueError("Root element <vmap:VMAP> missing required 'version' attribute")
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "AdBreak":
                    if "breakType" not in elem.attrib:
                        raise ValueError("AdBreak element missing required 'breakType' attribute")
                    if "timeOffset" not in elem.attrib:
                        raise ValueError("AdBreak element missing required 'timeOffset' attribute")
            return True
        except ET.ParseError as pe:
            raise ValueError(f"Malformed XML in {xml_path}: {pe}") from pe

    # Load and compile XMLSchema
    with open(schema_file, "rb") as sf:
        schema_doc = etree.parse(sf)
        schema = etree.XMLSchema(schema_doc)

    with open(xml_path, "rb") as xf:
        xml_doc = etree.parse(xf)

    if not schema.validate(xml_doc):
        errors = [f"Line {e.line}: {e.message}" for e in schema.error_log]
        error_msg = f"VMAP manifest failed schema validation against {schema_file.name}:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return True


def build_vmap_manifest(
    approved_breaks: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    xsd_path: Optional[str] = None,
    validate: bool = True
) -> str:
    """
    Construct VMAP 1.0.1 manifest from approved break candidate objects.
    Each break candidate must have:
      - candidate_id or break_id
      - timestamp (float, in seconds)
      - matched_brand (dict with brand_id, display_name, creative_asset, duration_sec)
    """
    template = jinja2.Template(VMAP_TEMPLATE_STRING)

    break_contexts = []
    for idx, cand in enumerate(approved_breaks):
        cid = cand.get("candidate_id", f"brk_{idx+1:03d}")
        ts = float(cand.get("timestamp", 0.0))
        brand = cand.get("matched_brand", {})
        if not isinstance(brand, dict):
            brand = {}

        duration_sec = float(brand.get("duration_sec", cand.get("max_ad_duration_sec", 15.0)))
        creative = brand.get("creative_asset", f"assets/ads/{brand.get('brand_id', 'generic')}.mp4")
        display_name = brand.get("display_name", brand.get("brand_id", "Sponsored Ad"))

        break_contexts.append({
            "break_id": cid,
            "ad_source_id": f"src_{cid}",
            "time_offset": format_timestamp_vmap(ts),
            "timestamp": ts,
            "brand_id": brand.get("brand_id", "brand_generic"),
            "brand_display_name": display_name,
            "duration_hms": format_duration_hms(duration_sec),
            "duration_sec": duration_sec,
            "creative_asset": creative
        })

    rendered_xml = template.render(breaks=break_contexts)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(rendered_xml)

        if validate:
            validate_vmap(str(out_p), xsd_path=xsd_path)

    return rendered_xml
