"""InternVL model integration for VQA Agent (Brain + Tools).

Uses dynamic high-resolution pre-processing and handles multi-image contexts.
Shared singleton model instance to conserve VRAM.
"""

from __future__ import annotations

import json
import logging
import re
import math
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from agent.brain import BrainResponse
from agent.tools import Tool

LOGGER = logging.getLogger(__name__)

# --- InternVL Vision Preprocessing Utils ---

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int = 448) -> T.Compose:
    """Build preprocessing transform for InternVL."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file: str | Path, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Load and preprocess image dynamically for InternVL."""
    image = Image.open(str(image_file)).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# --- Singleton Model Instance ---

_INTERNVL_MODEL = None
_INTERNVL_TOKENIZER = None

def get_internvl_model_and_tokenizer(model_name: str):
    """Lazy load InternVL model and tokenizer. Shares instance across Brain and Tools."""
    global _INTERNVL_MODEL, _INTERNVL_TOKENIZER
    if _INTERNVL_MODEL is not None and _INTERNVL_TOKENIZER is not None:
        return _INTERNVL_MODEL, _INTERNVL_TOKENIZER

    LOGGER.info("Loading InternVL model: %s...", model_name)
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Please install transformers and torchvision to use InternVL")

    # Use auto device map, but can be customized later if needed
    try:
        _INTERNVL_MODEL = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
    except Exception as exc:
        LOGGER.warning("Failed to load InternVL with flash_attn (%s). Falling back to eager mode.", exc)
        _INTERNVL_MODEL = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

    # Patch for transformers >= 4.45 compatibility
    if not hasattr(_INTERNVL_MODEL, "all_tied_weights_keys"):
        _INTERNVL_MODEL.all_tied_weights_keys = {}
        
    _INTERNVL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    LOGGER.info("InternVL loaded successfully.")
    
    return _INTERNVL_MODEL, _INTERNVL_TOKENIZER

# --- Brain & Tools ---

# We reuse the same system prompt structure but optimized for InternVL's multi-image capability.
SYSTEM_PROMPT = """You are a VQA (Video Question Answering) agent.
Your task is to answer questions about video frames accurately.

You have access to these tools:
1. caption(image_path): Describe the image in detail (objects, actions, colors, people).
2. ocr(image_path): Extract all visible written text, words, numbers, and signs from the image.

Rules:
- Read the question carefully. 
- If the question asks for names, exact text, poetry, or numbers -> YOU MUST use the `ocr` tool.
- If the question asks for visual descriptions, colors, or actions -> YOU MUST use the `caption` tool.
- Input format for tools: {"image_path": "file.jpg"}
- Answer the user directly after receiving the tool's observation. Keep it short and precise.
- Answer IN THE SAME LANGUAGE as the question (usually Vietnamese).

Examples of Tool Calling:
{"thought": "The user asks for the name on the sign. I need to read the text.", "action": "ocr", "action_input": {"image_path": "file.jpg"}}

Example of Answering:
{"thought": "The OCR result says 'Nguyen Trung Truc'. That is the answer.", "answer": "Nguyễn Trung Trực", "finished": true}
"""

class InternVLBrain:
    """Brain implementation using InternVL 2.5 local model."""

    def __init__(self, model_name: str = "OpenGVLab/InternVL2_5-2B") -> None:
        self.model_name = model_name
        self._history = None # Maintain history for multi-turn if needed
        # Trigger lazy load if possible, though tools might trigger it first
        
    def reset(self) -> None:
        """Reset conversation history."""
        self._history = None

    def reason(
        self,
        question: str,
        shot_info: str,
        frame_count: int,
        tool_results: list[dict] | None = None,
    ) -> BrainResponse:
        model, tokenizer = get_internvl_model_and_tokenizer(self.model_name)
        
        context = (
            f"Question: {question}\n"
            f"Shot info: {shot_info}\n"
            f"Frames available: {frame_count}\n"
        )
        if tool_results:
            history_text = "\n".join(
                f"Tool {r.get('tool', '?')} returned: {r.get('result', '')}" for r in tool_results
            )
            context += f"\nPrevious observations:\n{history_text}"

        prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nPlease generate a valid JSON object to act or answer."
        generation_config = dict(max_new_tokens=512, do_sample=True, temperature=0.3)
        
        try:
            # We don't pass an image to the brain, it only reasons on text. 
            # The tools will be the ones that look at the images.
            response, _ = model.chat(
                tokenizer, 
                pixel_values=None, 
                question=prompt, 
                generation_config=generation_config,
                history=None, # ReAct is practically stateless on the LLM side per step (we provide context)
                return_history=True
            )
            return self._parse_response(response.strip())
        except Exception as exc:
            LOGGER.exception("InternVLBrain reason failed")
            return BrainResponse(answer=f"Reasoning error: {exc}", finished=True)

    def _parse_response(self, text: str) -> BrainResponse:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            LOGGER.warning("No JSON in InternVL brain response: %s", text[:200])
            return BrainResponse(answer=text.strip(), finished=True)
        try:
            data = json.loads(json_match.group())
            if data.get("finished") or data.get("answer"):
                return BrainResponse(
                    thought=data.get("thought", ""),
                    answer=data.get("answer", ""),
                    finished=True,
                )
            return BrainResponse(
                thought=data.get("thought", ""),
                action=data.get("action", ""),
                action_input=data.get("action_input", {}),
                finished=False,
            )
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse InternVL brain JSON: %s", exc)
            return BrainResponse(answer=text.strip(), finished=True)


class InternVLCaptionTool(Tool):
    """Describe an image using InternVL."""

    name = "caption"
    description = (
        "Describe an image in detail. Input: image_path. "
        "Output: a detailed description of what is visible."
    )

    def __init__(self, model_name: str = "OpenGVLab/InternVL2_5-2B") -> None:
        self.model_name = model_name

    def run(self, image_path: str = "", prompt: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"

        if not prompt:
            prompt = (
                "Describe this image in detail in Vietnamese. "
                "Focus on: objects, colors, text, people, actions, positions. "
                "Be specific about colors (xanh dương, đỏ, trắng, vàng, v.v.) and text."
            )
            
        prompt = f"<image>\n{prompt}"
        
        try:
            model, tokenizer = get_internvl_model_and_tokenizer(self.model_name)
            # max_num=6 is a good balance for captioning (not too heavy)
            pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(model.device)
            generation_config = dict(max_new_tokens=512, do_sample=True, temperature=0.3)
            
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return response.strip()
        except Exception as exc:
            LOGGER.exception("InternVLCaptionTool failed")
            return f"[Caption error: {exc}]"


class InternVLOCRTool(Tool):
    """Read text from an image using InternVL."""

    name = "ocr"
    description = (
        "Read and extract text visible in an image. Input: image_path. "
        "Output: all text found in the image."
    )

    def __init__(self, model_name: str = "OpenGVLab/InternVL2_5-2B") -> None:
        self.model_name = model_name

    def run(self, image_path: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"

        prompt = "<image>\nExtract all visible text from this image. Return only the text found. If no text is found, reply 'No text found'."
        
        try:
            model, tokenizer = get_internvl_model_and_tokenizer(self.model_name)
            # max_num=12 for OCR to get maximum resolution for reading small text
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(model.device)
            generation_config = dict(max_new_tokens=512, do_sample=False) # Greedy search for OCR
            
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return response.strip()
        except Exception as exc:
            LOGGER.exception("InternVLOCRTool failed")
            return f"[OCR error: {exc}]"


def internvl_default_tools(model_name: str = "OpenGVLab/InternVL2_5-2B") -> dict[str, Tool]:
    """Return dict of InternVL-based tools."""
    return {
        "caption": InternVLCaptionTool(model_name=model_name),
        "ocr": InternVLOCRTool(model_name=model_name),
        "detect": __import__("agent.tools", fromlist=["DetectTool"]).DetectTool(),
        "asr": __import__("agent.tools", fromlist=["ASRTool"]).ASRTool(),
    }
