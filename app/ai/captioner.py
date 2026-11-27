"""Describe detected items with a vision model, cheaply and without guessing.

For each item the pipeline reports, take the clearest moment (nobody near
it), crop the item from the full-resolution frame and send the crop plus a
small context frame in ONE call. The answer must match a JSON schema whose
values are the contract's own enums; anything else is dropped to null.

Crops are saved even without a key: the chat shows them so people can see
what the camera saw, and the photo search compares colours against them.
Uses the shared LLMClient, so the daily call limit and cache apply.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .prompts import load as load_prompt

ITEM_TYPES = ["jacket", "coat", "hoodie", "sweater", "backpack", "bag", "bottle", "umbrella", "book", "phone", "other"]
COLOURS = ["black", "white", "gray", "red", "blue", "navy blue", "green", "yellow", "orange", "purple", "pink",
           "brown", "beige"]
FEATURES = ["hood", "zipper", "buttons", "pattern", "logo", "stripes", "pocket", "collar", "quilted", "rain_shell"]


def description_schema() -> dict:
    nullable = lambda values: {"type": ["string", "null"], "enum": values + [None]}  # noqa: E731
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["is_item", "is_worn", "item_type", "primary_color", "secondary_color",
                     "visible_features", "location_hint", "confidence"],
        "properties": {
            "is_item": {"type": "boolean"},
            "is_worn": {"type": "boolean"},
            "item_type": nullable(ITEM_TYPES),
            "primary_color": nullable(COLOURS),
            "secondary_color": nullable(COLOURS),
            "visible_features": {"type": "array", "items": {"type": "string", "enum": FEATURES}},
            "location_hint": {"type": ["string", "null"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
    }


def clean(desc: dict | None) -> dict | None:
    """Enforce the schema in code too: never trust the model's output shape."""
    if not isinstance(desc, dict):
        return None
    pick = lambda v, allowed: v if v in allowed else None  # noqa: E731
    features = [f for f in desc.get("visible_features") or [] if f in FEATURES]
    hint = desc.get("location_hint")
    hint = hint.strip()[:80] if isinstance(hint, str) and hint.strip() else None
    return {
        "is_item": bool(desc.get("is_item")),
        "is_worn": bool(desc.get("is_worn")),
        "item_type": pick(desc.get("item_type"), ITEM_TYPES),
        "primary_color": pick(desc.get("primary_color"), COLOURS),
        "secondary_color": pick(desc.get("secondary_color"), COLOURS),
        "visible_features": list(dict.fromkeys(features)),
        "location_hint": hint,
        "confidence": pick(desc.get("confidence"), ["low", "medium", "high"]) or "low",
    }


def phrase(desc: dict) -> str:
    """'gray hoodie with a hood and a zipper'"""
    colour = " and ".join(c for c in (desc.get("primary_color"), desc.get("secondary_color")) if c)
    item = desc.get("item_type") or "item"
    feats = [f.replace("_", " ") for f in desc.get("visible_features") or []]
    text = f"{colour} {item}".strip()
    if feats:
        text += " with " + ", ".join(f"a {f}" if f not in ("buttons", "stripes") else f for f in feats)
    return text


# ------------------------------------------------------------------ frames

def grab_frame(media: str | Path, offset_s: float, max_width: int = 1280) -> np.ndarray | None:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, offset_s):.2f}", "-i", str(media), "-frames:v", "1",
         "-vf", f"scale='min({max_width},iw)':-2", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    return frame


def crop_item(frame: np.ndarray, bbox, work_size) -> np.ndarray:
    """Crop the item (bbox in detector coordinates) with 40% padding."""
    fh, fw = frame.shape[:2]
    sx, sy = fw / max(work_size[0], 1), fh / max(work_size[1], 1)
    x, y, w, h = bbox
    cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
    half = max(w * sx, h * sy, 48) * 0.7
    x0, x1 = int(max(0, cx - half)), int(min(fw, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(fh, cy + half))
    return frame[y0:y1, x0:x1]


def jpeg(image: np.ndarray, max_side: int, quality: int = 85) -> bytes:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""


@dataclass
class ItemView:
    observation_id: str
    offset_s: float
    bbox: tuple
    work_size: tuple


def max_captions() -> int:
    try:
        return max(0, int(os.environ.get("CAPTIONS_PER_VIDEO", 12)))
    except ValueError:
        return 12


def describe_items(llm, media: str | Path, views: list[ItemView], crops_dir: Path, log=None) -> dict[str, dict]:
    """Save a crop per item; with a key, describe up to CAPTIONS_PER_VIDEO of them."""
    log = log or (lambda line: None)
    crops_dir.mkdir(parents=True, exist_ok=True)
    version, system = load_prompt("captioner")
    out: dict[str, dict] = {}
    budget = max_captions()
    for view in views:
        frame = None
        # A seek can fail on some files (for example at the very end of an
        # MPEG stream): step back a little rather than lose the picture.
        for offset in (view.offset_s, view.offset_s - 0.5, view.offset_s - 2.0, view.offset_s / 2):
            frame = grab_frame(media, offset)
            if frame is not None:
                break
        if frame is None:
            log(f"could not read a frame for {view.observation_id}; no picture saved")
            continue
        crop = crop_item(frame, view.bbox, view.work_size)
        if crop.size == 0:
            continue
        crop_jpg = jpeg(crop, 384)
        (crops_dir / f"{view.observation_id}.jpg").write_bytes(crop_jpg)
        if not llm.enabled or budget <= 0:
            continue
        budget -= 1
        context_jpg = jpeg(frame, 512, quality=75)
        desc = clean(llm.vision_json(
            system,
            "Image 1: close crop of the item. Image 2: the wider camera view.",
            [crop_jpg, context_jpg],
            description_schema(),
            cache_key=f"caption:{version}",
        ))
        if desc:
            desc["prompt_version"] = version
            out[view.observation_id] = desc
            log(f"AI description: {phrase(desc)}" + ("" if desc["is_item"] else " (may not be an item)"))
    return out


def apply(observation: dict, desc: dict, model: str) -> dict:
    """Merge a description into an observation, staying inside the contract."""
    attrs = observation["attributes"]
    if desc["is_item"]:
        attrs["item_type"] = desc["item_type"]
        if desc["primary_color"]:
            attrs["primary_color"] = desc["primary_color"]
        attrs["secondary_color"] = desc["secondary_color"]
        attrs["visible_features"] = desc["visible_features"]
    uncertainty = list(observation.get("uncertainty") or [])
    if not desc["is_item"] or desc["is_worn"] or desc["confidence"] == "low":
        if "ambiguous_attributes" not in uncertainty:
            uncertainty.append("ambiguous_attributes")
    elif desc["item_type"] and desc["item_type"] != "other" and "ambiguous_attributes" in uncertainty:
        uncertainty.remove("ambiguous_attributes")
    observation["uncertainty"] = uncertainty
    if desc["location_hint"] and attrs.get("location_label"):
        attrs["location_label"] = f"{desc['location_hint']}, {attrs['location_label']}"[:200]
    note = (f" AI description ({desc['confidence']} confidence): {phrase(desc)}."
            if desc["is_item"] else
            " The AI description says this may not be an item (for example part of a person or furniture).")
    observation["caption_text"] = (observation.get("caption_text") or "") + note
    observation["caption_version"] = f"openai_vision:{model}:captioner-{desc.get('prompt_version', '?')}"
    return observation
