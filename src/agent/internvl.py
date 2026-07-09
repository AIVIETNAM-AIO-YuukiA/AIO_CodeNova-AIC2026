"""InternVL model integration for VQA Agent (Brain + Tools).

Uses the HF-native InternVL3-2B-hf model (InternVLForConditionalGeneration),
which is upstreamed to transformers. No trust_remote_code, no monkey-patching,
no DynamicCache hacks — just standard transformers API.

Shared singleton model + processor instance to conserve VRAM.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image

from agent.brain import BrainResponse
from agent.tools import Tool

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton Model Loader
# ---------------------------------------------------------------------------

_INTERNVL_MODEL = None
_INTERNVL_PROCESSOR = None


def get_internvl_model_and_processor(model_name: str):
    """Lazy-load InternVL model and processor. Singleton — loads only once."""
    global _INTERNVL_MODEL, _INTERNVL_PROCESSOR
    if _INTERNVL_MODEL is not None and _INTERNVL_PROCESSOR is not None:
        return _INTERNVL_MODEL, _INTERNVL_PROCESSOR

    try:
        from transformers import AutoProcessor, InternVLForConditionalGeneration, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "InternVLForConditionalGeneration not found. "
            "Upgrade: pip install transformers>=5.0 bitsandbytes accelerate"
        ) from exc

    LOGGER.info("Loading InternVL model: %s...", model_name)
    _INTERNVL_PROCESSOR = AutoProcessor.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 supports float16 better than bfloat16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    try:
        _INTERNVL_MODEL = InternVLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
    except Exception as exc:
        LOGGER.warning("Flash Attention unavailable (%s). Using eager attention.", exc)
        _INTERNVL_MODEL = InternVLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()

    LOGGER.info("InternVL loaded successfully.")
    return _INTERNVL_MODEL, _INTERNVL_PROCESSOR


# ---------------------------------------------------------------------------
# Inference Helpers
# ---------------------------------------------------------------------------

def _run_text_inference(
    model,
    processor,
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.3,
) -> str:
    """Run text-only inference (no image)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    return processor.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _run_image_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.3,
) -> str:
    """Run image + text inference."""
    # Resize image to prevent massive patch generation (and OOM)
    # InternVL splits images into 448x448 tiles. Restricting max dimension to 448
    # ensures it generates EXACTLY 1 patch, dropping memory by > 75%.
    max_dim = 448
    if max(image.width, image.height) > max_dim:
        ratio = max_dim / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    return processor.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------

class InternVLBrain:
    """ReAct brain powered by InternVL3-2B-hf (text-only reasoning step)."""

    def __init__(self, model_name: str = "OpenGVLab/InternVL3-2B-hf") -> None:
        self.model_name = model_name

    def reset(self) -> None:
        pass  # Stateless per-step; context is re-built each call.

    def reason(
        self,
        question: str,
        shot_info: str,
        frame_count: int,
        tool_results: list[dict] | None = None,
    ) -> BrainResponse:
        model, processor = get_internvl_model_and_processor(self.model_name)

        context = f"Question: {question}\nShot info: {shot_info}\nFrames available: {frame_count}\n"
        if tool_results:
            obs = "\n".join(
                f"Tool {r.get('tool', '?')} returned: {r.get('result', '')}"
                for r in tool_results
            )
            context += f"\nPrevious observations:\n{obs}"

        prompt = (
            f"{SYSTEM_PROMPT}\n\n{context}\n\n"
            "Please generate a valid JSON object to act or answer."
        )

        try:
            response = _run_text_inference(model, processor, prompt)
            return self._parse_response(response)
        except Exception as exc:
            LOGGER.exception("InternVLBrain reason failed")
            return BrainResponse(answer=f"Reasoning error: {exc}", finished=True)

    def _parse_response(self, text: str) -> BrainResponse:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            LOGGER.warning("No JSON in InternVL response: %s", text[:200])
            return BrainResponse(answer=text.strip(), finished=True)
        try:
            data = json.loads(match.group())
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
            LOGGER.warning("Failed to parse InternVL JSON: %s", exc)
            return BrainResponse(answer=text.strip(), finished=True)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class InternVLCaptionTool(Tool):
    """Describe an image in detail using InternVL."""

    name = "caption"
    description = "Describe an image in detail. Input: image_path. Output: detailed description."

    def __init__(self, model_name: str = "OpenGVLab/InternVL3-2B-hf") -> None:
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

        try:
            model, processor = get_internvl_model_and_processor(self.model_name)
            image = Image.open(image_path).convert("RGB")
            return _run_image_inference(model, processor, image, prompt)
        except Exception as exc:
            LOGGER.exception("InternVLCaptionTool failed")
            return f"[Caption error: {exc}]"


class InternVLOCRTool(Tool):
    """Extract visible text from an image using InternVL."""

    name = "ocr"
    description = "Read all visible text from an image. Input: image_path. Output: extracted text."

    def __init__(self, model_name: str = "OpenGVLab/InternVL3-2B-hf") -> None:
        self.model_name = model_name

    def run(self, image_path: str = "") -> str:
        if not image_path:
            return "Error: image_path is required."
        if not Path(image_path).is_file():
            return f"Error: file not found: {image_path}"

        prompt = (
            "Extract all visible text from this image. "
            "Return only the text. If no text is found, reply 'No text found'."
        )

        try:
            model, processor = get_internvl_model_and_processor(self.model_name)
            image = Image.open(image_path).convert("RGB")
            return _run_image_inference(
                model, processor, image, prompt, do_sample=False
            )
        except Exception as exc:
            LOGGER.exception("InternVLOCRTool failed")
            return f"[OCR error: {exc}]"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def internvl_default_tools(model_name: str = "OpenGVLab/InternVL3-2B-hf") -> dict[str, Tool]:
    """Return the default InternVL tool set."""
    return {
        "caption": InternVLCaptionTool(model_name=model_name),
        "ocr": InternVLOCRTool(model_name=model_name),
        "detect": __import__("agent.tools", fromlist=["DetectTool"]).DetectTool(),
        "asr": __import__("agent.tools", fromlist=["ASRTool"]).ASRTool(),
    }
