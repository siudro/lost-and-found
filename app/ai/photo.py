"""Photo queries: "Can you find this?" with a picture of the lost item.

With an OpenAI key, a vision model describes the photo into the same
structured fields the captioner uses (item type, colours, features). Without
one, only the main colour of the centre of the photo is used, and the
answer says so. The photo is processed in memory and never stored.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.detector import name_colour
from .captioner import clean, description_schema, jpeg, phrase
from .prompts import load as load_prompt

MAX_PHOTO_BYTES = 8 * 1024 * 1024


class PhotoError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    if not data:
        raise PhotoError("the photo is empty")
    if len(data) > MAX_PHOTO_BYTES:
        raise PhotoError("the photo is larger than 8 MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise PhotoError("that file is not a photo this server can read (use JPEG or PNG)")
    return image


def main_colour(image: np.ndarray) -> str:
    """Colour of the middle of the photo, where the item usually is."""
    h, w = image.shape[:2]
    centre = image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    b, g, r = (float(v) for v in np.median(centre.reshape(-1, 3), axis=0))
    return name_colour(r, g, b)


def describe_photo(llm, image: np.ndarray) -> dict:
    """{'source': 'ai'|'colour', 'description': {...}, 'phrase': str}"""
    if llm.enabled:
        version, system = load_prompt("photo_query")
        desc = clean(llm.vision_json(system, "The visitor's photo of their lost item.", [jpeg(image, 512)],
                                     description_schema(), cache_key=f"photo:{version}"))
        if desc and desc["is_item"]:
            return {"source": "ai", "description": desc, "phrase": phrase(desc)}
    colour = main_colour(image)
    desc = {"is_item": True, "is_worn": False, "item_type": None, "primary_color": colour,
            "secondary_color": None, "visible_features": [], "location_hint": None, "confidence": "low"}
    return {"source": "colour", "description": desc, "phrase": f"{colour} item"}


def search_text(message: str, desc: dict) -> str:
    """Turn the photo description into words the query parser understands."""
    words = [desc.get("primary_color") or "", desc.get("secondary_color") or ""]
    item = desc.get("item_type")
    if item and item != "other":
        words.append(item)
    words += [f.replace("_", " ") for f in desc.get("visible_features") or []]
    base = message.strip() or "Have you seen this?"
    return f"{base} ({' '.join(w for w in words if w)})"
