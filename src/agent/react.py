"""ReAct Agent: Thought → Action → Tool → Observation → Answer."""

from __future__ import annotations

from pathlib import Path
import logging

from retrieval.temporal_search import ShotInput
from agent.brain import VlmBrain
from agent.tools import Tool, default_tools

LOGGER = logging.getLogger(__name__)

MAX_STEPS = 5


class Agent:
    """ReAct Agent for VQA answer generation.

    Args:
        brain: VlmBrain instance (Gemini API).
        tools: Dict of tool_name → Tool instance.
        max_steps: Maximum ReAct iterations before forced answer.
    """

    def __init__(
        self,
        brain: VlmBrain | None = None,
        tools: dict[str, Tool] | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.brain = brain or VlmBrain()
        self.tools = tools or default_tools()
        self.max_steps = max_steps

    def answer(
        self,
        shot: ShotInput,
        question: str,
    ) -> str:
        """Run ReAct loop to answer a VQA question.

        Args:
            shot: Validated shot input (frames with context).
            question: The VQA question.

        Returns:
            Final answer text.
        """
        self.brain.reset()

        shot_info = (
            (
                f"video={shot.video_name or shot.video_id}, "
                f"frames={shot.frame_count}, "
                f"time=[{shot.start_timestamp:.1f}s - {shot.end_timestamp:.1f}s]"
            )
            if shot.start_timestamp
            else (f"video={shot.video_name or shot.video_id}, " f"frames={shot.frame_count}")
        )

        LOGGER.info("Agent starting: question=%s shot_info=%s", question, shot_info)

        tool_results: list[dict] = []
        first_frame_path = shot.frame_paths[len(shot.frame_paths) // 2] if shot.frame_paths else ""

        for step in range(1, self.max_steps + 1):
            LOGGER.info("Agent step %d/%d", step, self.max_steps)

            response = self.brain.reason(
                question=question,
                shot_info=shot_info,
                frame_count=shot.frame_count,
                tool_results=tool_results,
            )

            LOGGER.debug(
                "Step %d brain: thought=%s action=%s answer=%s",
                step,
                response.thought[:100] if response.thought else "",
                response.action,
                response.answer[:100] if response.answer else "",
            )

            # Nếu brain đã có đáp án
            if response.finished:
                answer = (response.answer or "").strip()
                if answer:
                    LOGGER.info("Agent finished with answer: %s", answer[:200])
                    return answer
                # Fallback: answer rỗng → dùng thought hoặc tool result cuối
                fallback = response.thought.strip() if response.thought else ""
                if not fallback and tool_results:
                    fallback = tool_results[-1].get("result", "")
                LOGGER.info("Agent finished with empty answer, fallback: %s", fallback[:200])
                return fallback or "No answer generated."

            # Thực thi tool
            if response.action and response.action in self.tools:
                tool = self.tools[response.action]
                raw = response.action_input
                tool_input = dict(raw) if isinstance(raw, dict) else {}

                # Luôn gắn image_path mặc định nếu tool cần ảnh và chưa có
                if response.action in ("caption", "ocr", "detect"):
                    if "image_path" not in tool_input or not tool_input["image_path"]:
                        tool_input["image_path"] = first_frame_path
                    # Truyền câu hỏi làm prompt cho caption
                    if response.action == "caption":
                        tool_input["prompt"] = question

                LOGGER.info("Executing tool=%s input=%s", response.action, tool_input)
                try:
                    result = tool.run(**tool_input)
                except Exception as exc:
                    result = f"[Tool error: {exc}]"
                    LOGGER.exception("Tool %s failed", response.action)

                tool_results.append({"tool": response.action, "result": result})
                LOGGER.debug("Tool result: %s", result[:200])
            else:
                LOGGER.warning("Unknown tool: %s", response.action)
                tool_results.append({"tool": response.action, "result": "[Unknown tool]"})

        # Hết step mà chưa có đáp án → force answer
        LOGGER.warning("Agent reached max steps without final answer")
        final_context = "\n".join(f"{r['tool']}: {r['result'][:300]}" for r in tool_results)
        fallback = (
            f"Based on the analysis of the video shot:\n"
            f"{final_context}\n\n"
            f"Question: {question}"
        )
        return fallback


def _load_dotenv() -> None:
    """Load .env file if GEMINI_API_KEY is not already set."""
    import os

    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                if key.strip() == "GEMINI_API_KEY":
                    os.environ["GEMINI_API_KEY"] = val.strip()
                    break


def create_agent() -> Agent:
    """Create an Agent.

    Priority:
      1. Gemini API key available → VlmBrain + cloud tools
      2. Ollama running → LocalBrain + local tools
      3. Otherwise → VlmBrain (will error gracefully at runtime)
    """
    import os

    _load_dotenv()
    if os.environ.get("GEMINI_API_KEY"):
        LOGGER.info("GEMINI_API_KEY found — using cloud brain and tools")
        brain = VlmBrain()
        tools = default_tools()
        return Agent(brain=brain, tools=tools, max_steps=MAX_STEPS)

    try:
        from agent.local import is_ollama_running, LocalBrain, local_default_tools

        if is_ollama_running():
            LOGGER.info("Ollama detected — using local brain and tools")
            brain = LocalBrain()
            tools = local_default_tools()
            return Agent(brain=brain, tools=tools, max_steps=MAX_STEPS)
    except Exception as exc:
        LOGGER.debug("Ollama detection failed: %s", exc)

    brain = VlmBrain()
    tools = default_tools()
    return Agent(brain=brain, tools=tools, max_steps=MAX_STEPS)
