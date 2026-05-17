import os
import re
import json
import time
import random
import mimetypes
import hashlib
from pathlib import Path
from typing import Union, Optional
from io import BytesIO
from pydantic import BaseModel, field_validator, ConfigDict, Field, ValidationError
from google import genai
from PIL import Image
import imagehash


CACHE_DIR = Path("cache")
from google.genai.types import Part


MAX_FILE_SIZE_BYTES = 1500 * 1024  # 1500KB
DEFAULT_TIMEOUT_MS = 300000  # 5 minutes
DEFAULT_TEMPERATURE = 0.1  # Low for deterministic structured output
DEFAULT_SEED = 42  # For reproducibility
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 60.0
IMAGE_TOKENS_ESTIMATE = 258
MIN_CONFIDENCE_THRESHOLD = 0.6  # 60% - for retry
HUMAN_APPROVAL_THRESHOLD = 0.5  # 50% - require human approval below this
ENABLE_HUMAN_APPROVAL = True  # Toggle for human-in-the-loop
MAX_INPUT_TOKENS = 1_000_000

FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash-8k"]


# ============ Guardrails ============

def validate_non_empty_input(image_path: str) -> Path:
    if not image_path:
        raise ValueError("Image path cannot be empty")
    path = Path(image_path)
    if not isinstance(image_path, (str, Path)):
        raise ValueError("Image path must be string or Path")
    return path


def validate_api_key(api_key: str) -> str:
    if not api_key:
        raise ValueError("API key cannot be empty")
    if len(api_key) < 20:
        raise ValueError("API key appears invalid")
    return api_key


def validate_image_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError("Image file is empty")


def compress_image(image_bytes: bytes, quality: int = 85, max_dimension: int = 1920) -> bytes:
    """Compress image to reduce cost while maintaining quality."""
    img = Image.open(BytesIO(image_bytes))
    
    # Resize if too large
    if max(img.width, img.height) > max_dimension:
        ratio = max_dimension / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Compress to JPEG
    output = BytesIO()
    img = img.convert("RGB")  # Ensure RGB for JPEG
    img.save(output, format="JPEG", quality=quality, optimize=True)
    
    return output.getvalue()


def get_image_hash(image_bytes: bytes) -> str:
    """Generate perceptual hash (pHash) for visually similar image detection."""
    img = Image.open(BytesIO(image_bytes))
    # pHash: 8x8 hash = 64 bits, good for detecting visual similarity
    return str(imagehash.phash(img))


def get_from_cache(image_hash: str) -> Optional[dict]:
    """Check cache for visually similar image using hamming distance."""
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Compare with existing cached images
    max_hamming_distance = 10  # Allow up to 10 bits difference (~85% similar)
    
    for cache_file in CACHE_DIR.glob("*.json"):
        cached_hash = cache_file.stem
        try:
            distance = imagehash.hex_to_hash(image_hash) - imagehash.hex_to_hash(cached_hash)
            if distance <= max_hamming_distance:
                print(f"📋 Cache hit: {cached_hash[:16]}... (hamming dist: {distance})")
                return json.loads(cache_file.read_text())
        except Exception:
            continue
    
    return None


def save_to_cache(image_hash: str, result: dict) -> None:
    """Save result to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{image_hash}.json"
    cache_file.write_text(json.dumps(result, indent=2))
    print(f"💾 Cached: {image_hash[:16]}...")


def request_human_approval(validated: ScreenOutput) -> bool:
    """Request human approval for low-confidence results."""
    if not ENABLE_HUMAN_APPROVAL:
        return True
    
    # Check average confidence
    confidences = [e.metadata.confidence for e in validated.elements if e.metadata.confidence is not None]
    if not confidences:
        return True
    
    avg_confidence = sum(confidences) / len(confidences)
    
    if avg_confidence < HUMAN_APPROVAL_THRESHOLD:
        low_conf_count = sum(1 for c in confidences if c < HUMAN_APPROVAL_THRESHOLD)
        
        print(f"\n⚠️ LOW CONFIDENCE DETECTED")
        print(f"   Average: {avg_confidence*100:.1f}%")
        print(f"   Low confidence elements: {low_conf_count}/{len(confidences)}")
        print(f"\nOptions:")
        print(f"   [a] - Accept and continue")
        print(f"   [r] - Reject and return None")
        print(f"   [v] - View full output")
        
        while True:
            choice = input("   Enter choice (a/r/v): ").strip().lower()
            
            if choice == 'a':
                print(f"✅ Human approved - continuing")
                return True
            elif choice == 'r':
                print(f"❌ Human rejected")
                return False
            elif choice == 'v':
                print(f"\n--- OUTPUT ---")
                for i, el in enumerate(validated.elements[:5]):
                    conf = el.metadata.confidence or 0
                    print(f"  {i+1}. {el.identity.type} (conf: {conf*100:.0f}%)")
                if len(validated.elements) > 5:
                    print(f"  ... and {len(validated.elements)-5} more")
                print(f"--- END ---\n")
            else:
                print("   Invalid choice. Try a, r, or v")
    
    return True


def estimate_token_count(text: str, num_images: int = 1) -> int:
    text_tokens = len(text) // 4
    image_tokens = num_images * IMAGE_TOKENS_ESTIMATE
    return text_tokens + image_tokens


def validate_token_limit(text: str, num_images: int = 1) -> None:
    estimated = estimate_token_count(text, num_images)
    if estimated > MAX_INPUT_TOKENS:
        raise ValueError(f"Tokens ({estimated:,}) exceed limit ({MAX_INPUT_TOKENS:,})")


def mask_pii_in_text(text: str) -> str:
    if not text:
        return text
    
    # Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Phone (various formats)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', '[PHONE]', text)
    
    # Credit card
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', text)
    
    # SSN
    text = re.sub(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[SSN]', text)
    
    # IP address
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)
    
    # Names (capitalized words - heuristic)
    text = re.sub(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', '[NAME]', text)
    
    return text


def mask_pii_in_output(llm_output: dict) -> dict:
    """Mask PII in all text fields of the output."""
    for element in llm_output.get("elements", []):
        content = element.get("content", {})
        metadata = element.get("metadata", {})
        
        text_fields = [
            content.get("label"),
            content.get("placeholder"),
            content.get("value"),
            content.get("text_content"),
            content.get("tooltip"),
            content.get("description"),
            metadata.get("name"),
            metadata.get("href"),
        ]
        
        for field in text_fields:
            if field and isinstance(field, str):
                masked = mask_pii_in_text(field)
                if field == content.get("label"):
                    content["label"] = masked
                elif field == content.get("placeholder"):
                    content["placeholder"] = masked
                elif field == content.get("value"):
                    content["value"] = masked
                elif field == content.get("text_content"):
                    content["text_content"] = masked
                elif field == content.get("tooltip"):
                    content["tooltip"] = masked
                elif field == content.get("description"):
                    content["description"] = masked
                elif field == metadata.get("name"):
                    metadata["name"] = masked
                elif field == metadata.get("href"):
                    metadata["href"] = masked
    
    return llm_output


# ============ Input Validation ============

class ImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    image: Union[str, Path, bytes]
    max_size_bytes: int = MAX_FILE_SIZE_BYTES
    
    @field_validator('image')
    @classmethod
    def validate_image_format(cls, v: Union[str, Path, bytes], info) -> Union[str, Path, bytes]:
        max_size = info.data.get('max_size_bytes', MAX_FILE_SIZE_BYTES)
        
        if isinstance(v, (str, Path)):
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Image file does not exist: {path}")
            
            ext = path.suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                raise ValueError(f"Invalid format '{ext}'. Must be .jpg, .jpeg, or .png")
            
            size = path.stat().st_size
            if size > max_size:
                raise ValueError(f"Image too large: {size/1024:.1f}KB exceeds {max_size/1024:.0f}KB")
            
            magic = path.read_bytes()[:8]
            if not (magic[:8] == b'\x89PNG\r\n\x1a\n' or magic[:3] == b'\xff\xd8\xff'):
                raise ValueError("Image must be JPEG or PNG")
        
        elif isinstance(v, bytes):
            if len(v) > max_size:
                raise ValueError(f"Image too large: {len(v)/1024:.1f}KB")
            
            if not (v[:8] == b'\x89PNG\r\n\x1a\n' or v[:3] == b'\xff\xd8\xff'):
                raise ValueError("Image must be JPEG or PNG")
        
        return v


# ============ Output Schema ============

class Identity(BaseModel):
    type: str
    role: Optional[str] = None
    semantic_type: Optional[str] = None


class Position(BaseModel):
    coordinate_type: Optional[str] = "pixel"
    x: float
    y: float
    width: float
    height: float
    center_x: Optional[float] = None
    center_y: Optional[float] = None


class Content(BaseModel):
    label: Optional[str] = None
    placeholder: Optional[str] = None
    value: Optional[str] = None
    alt_text: Optional[str] = None
    tooltip: Optional[str] = None
    description: Optional[str] = None
    text_content: Optional[str] = None
    icon_name: Optional[str] = None


class State(BaseModel):
    enabled: Optional[bool] = True
    visible: Optional[bool] = True
    focused: Optional[bool] = False
    checked: Optional[bool] = False
    selected: Optional[bool] = False
    hovered: Optional[bool] = False
    expanded: Optional[bool] = False
    loading: Optional[bool] = False
    required: Optional[bool] = False
    readonly: Optional[bool] = False
    invalid: Optional[bool] = False


class Hierarchy(BaseModel):
    parent_id: Optional[str] = None
    parent_type: Optional[str] = None
    siblings: Optional[list[str]] = []
    children: Optional[list[str]] = []
    depth: Optional[int] = 0
    path: Optional[str] = None


class Styling(BaseModel):
    color: Optional[str] = None
    background_color: Optional[str] = None
    size_class: Optional[str] = None
    prominence: Optional[str] = None


class Accessibility(BaseModel):
    aria_label: Optional[str] = None
    aria_role: Optional[str] = None
    tab_index: Optional[int] = None


class ElementMetadata(BaseModel):
    id: str
    name: Optional[str] = None
    href: Optional[str] = None
    src: Optional[str] = None
    confidence: Optional[float] = None


class ScreenElement(BaseModel):
    identity: Identity
    position: Position
    content: Optional[Content] = None
    state: Optional[State] = None
    hierarchy: Optional[Hierarchy] = None
    styling: Optional[Styling] = None
    accessibility: Optional[Accessibility] = None
    metadata: ElementMetadata


class ScreenOutput(BaseModel):
    elements: list[ScreenElement]
    metadata: Optional[dict] = None


# ============ LLM Integration ============

SYSTEM_PROMPT = """You are an expert UI element detection system.

Output JSON with these fields. Include confidence score (0-1) for each element.

For enum fields: if uncertain, use "unknown" and set confidence < 0.75.
If confident >= 75%, you may guess.

Return ONLY valid JSON."""


def _classify_error(err: str) -> tuple[str, str]:
    err_lower = err.lower()
    if "429" in err or "rate limit" in err_lower:
        return "rate_limit", "Rate limited"
    if "404" in err or "not found" in err_lower:
        return "model_not_found", "Model not found"
    if any(x in err for x in ["500", "502", "503", "INTERNAL", "UNAVAILABLE"]):
        return "server_error", "Server error"
    if "401" in err or "invalid" in err_lower:
        return "auth_error", "Auth error"
    if "403" in err:
        return "permission_error", "Permission denied"
    return "other", "Unknown"


def process_screenshot(
    image_path: str,
    api_key: str,
    output_path: str = "output.json",
    max_retries: int = MAX_RETRIES,
    primary_model: str = "gemini-2.5-flash"
) -> Optional[ScreenOutput]:
    
    # Guardrails
    path = validate_non_empty_input(image_path)
    validate_api_key(api_key)
    validate_image_file(path)
    
    try:
        ImageInput(image=str(path))
    except ValidationError as ve:
        if "too large" in str(ve):
            print("❌ Image too large. Re-upload under 1.5MB")
            return None
        raise
    
    image_bytes = path.read_bytes()
    
    # Compress image to reduce cost
    original_size = len(image_bytes)
    image_bytes = compress_image(image_bytes)
    compressed_size = len(image_bytes)
    print(f"📦 Image: {original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB (saved {100-((compressed_size/original_size)*100):.0f}%)")
    
    # Check cache before calling LLM
    image_hash = get_image_hash(image_bytes)
    cached_result = get_from_cache(image_hash)
    if cached_result:
        print(f"✅ Returning cached result")
        with open(output_path, "w") as f:
            json.dump(cached_result, f, indent=2)
        return ScreenOutput.model_validate(cached_result)
    
    validate_token_limit(SYSTEM_PROMPT, 1)
    
    client = genai.Client(api_key=api_key, http_options={"timeout": DEFAULT_TIMEOUT_MS})
    
    models = [primary_model] + FALLBACK_MODELS
    circuit_broken = False
    
    for model in models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[SYSTEM_PROMPT, Part.from_bytes(data=image_bytes, mime_type="image/png")],
                    config={
                        "temperature": DEFAULT_TEMPERATURE,
                        "seed": DEFAULT_SEED
                    }
                )
                
                if not response.text or not response.text.strip():
                    raise ValueError("Empty response")
                
                llm_output = json.loads(response.text)
                
                # PII Masking
                llm_output = mask_pii_in_output(llm_output)
                
                # Validation with circuit breaker
                for vattempt in range(max_retries):
                    try:
                        validated = ScreenOutput.model_validate(llm_output)
                        
                        low_conf = [e for e in validated.elements 
                                   if e.metadata.confidence and e.metadata.confidence < MIN_CONFIDENCE_THRESHOLD]
                        
                        if low_conf and vattempt < max_retries - 1:
                            response = client.models.generate_content(
                                model=model,
                                contents=["Improve detection, retry"],
                                config={
                                    "temperature": DEFAULT_TEMPERATURE,
                                    "seed": DEFAULT_SEED
                                }
                            )
                            llm_output = json.loads(response.text)
                            llm_output = mask_pii_in_output(llm_output)
                            continue
                        
                        circuit_broken = True
                        break
                    except ValidationError:
                        if vattempt < max_retries - 1:
                            response = client.models.generate_content(
                                model=model,
                                contents=["Fix JSON format errors"],
                                config={
                                    "temperature": DEFAULT_TEMPERATURE,
                                    "seed": DEFAULT_SEED
                                }
                            )
                            llm_output = json.loads(response.text)
                            llm_output = mask_pii_in_output(llm_output)
                        else:
                            return None
                
                if circuit_broken:
                    break
                    
            except Exception as e:
                err_type, _ = _classify_error(str(e))
                
                if err_type == "rate_limit" and attempt < max_retries - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"Rate limited, retry in {delay:.1f}s...")
                    time.sleep(delay + random.uniform(0, delay * 0.3))
                    continue
                
                if err_type == "server_error" and attempt < max_retries - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    time.sleep(delay + random.uniform(0, delay * 0.3))
                    continue
                
                if err_type == "model_not_found":
                    break
                
                if err_type in ["auth_error", "permission_error"]:
                    print(f"❌ {err_type}: {e}")
                    return None
                
                break
        
        if circuit_broken:
            break
    
    if circuit_broken:
        # Human approval loop for low confidence
        if not request_human_approval(validated):
            return None
        
        with open(output_path, "w") as f:
            json.dump(llm_output, f, indent=2)
        save_to_cache(image_hash, llm_output)
        print(f"✅ Saved to {output_path}")
        return validated
    
    print("❌ Failed after all retries")
    return None


if __name__ == "__main__":
    result = process_screenshot(
        "/Users/nandana/build/screen_understanding_agent/image.png",
        "AIzaSyCg_fHYi6R-jwUOcp7neXuynuG__LUC9b0"
    )
    print(f"✅ Done: {len(result.elements) if result else 0} elements")