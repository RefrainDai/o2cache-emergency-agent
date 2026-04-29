import argparse
import base64
import json
import os
import re
import io
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from tool_agents.system_prompt import SYSTEM_PROMPT
from tool_agents.tool_registry import get_tool, registry


DEFAULT_MODEL_DIR = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"


def _print_json_block(title: str, payload: Dict[str, Any]) -> None:
    def _sanitize(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {item_key: _sanitize(item_val, item_key) for item_key, item_val in value.items()}
        if isinstance(value, list):
            return [_sanitize(item, key) for item in value]
        if isinstance(value, str) and key.endswith("_base64"):
            return f"<omitted base64, length={len(value)}>"
        if key == "llm_handoff" and isinstance(value, str):
            return f"<omitted handoff text, length={len(value)}>"
        if isinstance(value, str) and len(value) > 12000:
            return f"<omitted long string, length={len(value)}>"
        return value

    print(f"\n[{title}]")
    print(json.dumps(_sanitize(payload), ensure_ascii=False, indent=2))


def _sanitize_for_display(payload: Any) -> Any:
    if isinstance(payload, dict):
        compact: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str) and key.endswith("_base64"):
                compact[key] = f"<omitted base64, length={len(value)}>"
            elif key == "llm_handoff" and isinstance(value, str):
                compact[key] = f"<omitted handoff text, length={len(value)}>"
            else:
                compact[key] = _sanitize_for_display(value)
        return compact
    if isinstance(payload, list):
        return [_sanitize_for_display(item) for item in payload]
    if isinstance(payload, str) and len(payload) > 20000:
        return f"<omitted very long string, length={len(payload)}>"
    return payload


def _compact_tool_history_for_prompt(tool_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_history: List[Dict[str, Any]] = []
    for item in tool_history:
        compact_item: Dict[str, Any] = {
            "step": item.get("step"),
            "tool_name": item.get("tool_name"),
            "tool_args": _sanitize_for_display(item.get("tool_args", {})),
        }
        result = item.get("result", {})
        compact_item["result"] = _sanitize_for_display(result)
        compact_history.append(compact_item)
    return compact_history


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Cannot parse JSON from model output: {text}")
    return json.loads(match.group(0))


def _build_final_answer_prompt(user_query: str, handoff_payload: str) -> str:
    return (
        f"系统角色: {SYSTEM_PROMPT}\n"
        f"用户请求: {user_query}\n"
        f"工具回传数据: {handoff_payload}"
    )


def _planning_tool_input(
    user_query: str,
    available_tools: List[str],
    allow_tools: bool,
    tool_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "user_query": user_query,
        "available_tools": available_tools,
        "allow_tools": allow_tools,
        "tool_history": _compact_tool_history_for_prompt(tool_history),
    }


def _normalize_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize planner output when tool JSON is embedded as response text."""
    action = decision.get("action")
    if action != "respond":
        return decision

    response = decision.get("response")
    if not isinstance(response, str):
        return decision

    text = response.strip()
    if not text.startswith("{"):
        return decision

    try:
        nested = _extract_json_object(text)
    except Exception:
        return decision

    if isinstance(nested, dict) and nested.get("action") in {"call_tool", "respond"}:
        return nested
    return decision


def _maybe_extract_embedded_tool_call(response: str) -> Dict[str, Any]:
    """Best-effort extraction for truncated/non-JSON planner outputs."""
    text = response or ""
    if "call_tool" not in text:
        return {}

    if "tool_ending" in text:
        return {"action": "call_tool", "tool_name": "tool_ending", "tool_args": {}}
    if "report_making" in text:
        return {"action": "call_tool", "tool_name": "report_making", "tool_args": {}}
    if "image_analysis" in text:
        return {"action": "call_tool", "tool_name": "image_analysis", "tool_args": {}}
    if "rescuenet_segmentation" in text:
        return {"action": "call_tool", "tool_name": "rescuenet_segmentation", "tool_args": {}}
    return {}


def _decode_base64_image(image_base64: str) -> Image.Image:
    raw = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _user_requests_report(user_query: str) -> bool:
    text = (user_query or "").lower()
    keywords = [
        "报告",
        "生成报告",
        "出报告",
        "汇报",
        "总结",
        "写报告",
        "report",
        "summary",
    ]
    return any(keyword in text for keyword in keywords)


class QwenToolAgent:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str = "auto") -> None:
        self.model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Qwen3-VL model. Please confirm transformers version supports "
                "Qwen3VLForConditionalGeneration and model files are complete."
            ) from exc

        if self.device != "cuda":
            self.model.to(self.device)

    def _generate_with_images(self, prompt: str, images: List[Image.Image], max_new_tokens: int = 768) -> str:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": f"系统角色: {SYSTEM_PROMPT}\n{prompt}"})
        messages = [{"role": "user", "content": content}]

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs["input_ids"], outputs)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer(chat_text, return_tensors="pt")
        else:
            model_inputs = self.tokenizer(prompt, return_tensors="pt")

        input_ids = model_inputs["input_ids"].to(self.model.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _run_rescuenet_segmentation(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        image_paths = tool_args.get("image_paths")
        if not image_paths and isinstance(tool_args.get("image_path"), str):
            image_paths = [tool_args["image_path"]]
        if not isinstance(image_paths, list) or not image_paths:
            raise ValueError("tool_args.image_paths must be a non-empty list")

        workspace_root = tool_args.get("workspace_root")

        tool = get_tool("rescuenet_segmentation", workspace_root=workspace_root)
        results = tool.infer_batch(image_paths=image_paths)

        return {
            "tool_name": "rescuenet_segmentation",
            "num_images": len(results),
            "image_paths": [item.get("image_path") for item in results],
            "items": results,
        }

    def _run_image_analysis(
        self,
        tool_args: Dict[str, Any],
        user_query: str,
        history: List[Dict[str, Any]],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        analysis_request = get_tool(
            "image_analysis",
            user_query=user_query,
            tool_history=history,
            target_step=tool_args.get("target_step"),
            image_paths=tool_args.get("image_paths"),
        )

        used_segmentation_tool = bool(analysis_request.get("used_segmentation_tool", False))
        analyses: List[Dict[str, Any]] = []
        for item in analysis_request.get("items", []):
            if used_segmentation_tool:
                original_image = _decode_base64_image(item["original_image_base64"])
                overlay_image = _decode_base64_image(item["overlay_image_base64"])
                images_for_model = [original_image, overlay_image]
                image_prompt = (
                    f"{analysis_request.get('analysis_prompt', '')}\n\n"
                    f"用户任务: {user_query}\n"
                    f"图像路径: {item.get('image_path')}\n"
                    f"是否使用分割工具: {used_segmentation_tool}\n"
                    f"类别统计: {json.dumps(item.get('class_statistics', []), ensure_ascii=False)}\n"
                    "请重点分析原图中可见的灾害现象，以及叠加图反映的地物类别分布、受损区域、道路通行、水体、植被和建筑损伤状态。"
                    "输出应详细、准确、审慎，并明确说明哪些判断来自分割色块，哪些判断来自原始视觉线索。"
                )
            else:
                image_path = item.get("image_path")
                if not isinstance(image_path, str) or not image_path:
                    raise ValueError("image_analysis item.image_path is required when segmentation is not used")
                original_image = Image.open(image_path).convert("RGB")
                images_for_model = [original_image]
                image_prompt = (
                    f"{analysis_request.get('analysis_prompt', '')}\n\n"
                    f"用户任务: {user_query}\n"
                    f"图像路径: {image_path}\n"
                    f"是否使用分割工具: {used_segmentation_tool}\n"
                    "请仅基于原图进行分析，不要假设存在分割图。"
                )

            analysis_text = self._generate_with_images(
                image_prompt,
                images_for_model,
                max_new_tokens=max_new_tokens,
            )
            analyses.append(
                {
                    "image_path": item.get("image_path"),
                    "analysis": analysis_text,
                    "class_statistics": item.get("class_statistics", [] if used_segmentation_tool else None),
                }
            )

        return {
            "tool_name": "image_analysis",
            "based_on_step": analysis_request.get("target_step"),
            "used_segmentation_tool": used_segmentation_tool,
            "analyses": analyses,
        }

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_query: str,
        history: List[Dict[str, Any]],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        if tool_name == "rescuenet_segmentation":
            return self._run_rescuenet_segmentation(tool_args)
        if tool_name == "image_analysis":
            return self._run_image_analysis(tool_args, user_query, history, max_new_tokens)
        if tool_name == "report_making":
            return get_tool(
                "report_making",
                user_query=user_query,
                tool_history=history,
                extra_notes=tool_args.get("extra_notes", ""),
            )
        if tool_name == "tool_ending":
            return get_tool(
                "tool_ending",
                user_query=user_query,
                tool_history=history,
                report=tool_args.get("report", ""),
            )
        raise ValueError(f"Unsupported tool in dispatcher: {tool_name}")

    @staticmethod
    def _maybe_redirect_tool_call(tool_name: str, tool_args: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        segmentation_steps = [item for item in history if item.get("tool_name") == "rescuenet_segmentation"]
        image_analysis_steps = [item for item in history if item.get("tool_name") == "image_analysis"]

        if tool_name == "rescuenet_segmentation" and segmentation_steps:
            return {
                "tool_name": "image_analysis",
                "tool_args": {"target_step": segmentation_steps[-1].get("step")},
            }
        if tool_name == "report_making" and segmentation_steps and not image_analysis_steps:
            return {
                "tool_name": "image_analysis",
                "tool_args": {"target_step": segmentation_steps[-1].get("step")},
            }
        return {"tool_name": tool_name, "tool_args": tool_args}

    def run(
        self,
        user_query: str,
        allow_tools: bool = True,
        max_new_tokens: int = 512,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        print("\n[AGENT INPUT]")
        print(user_query)

        available_tools = registry.list_tools()
        tool_history: List[Dict[str, Any]] = []

        for step in range(max_steps):
            planning_input = _planning_tool_input(user_query, available_tools, allow_tools, tool_history)
            if step == 0:
                _print_json_block(
                    "TOOL INPUT",
                    {
                        "step": 1,
                        "tool_name": "thinking_and_plan",
                        "tool_args": planning_input,
                    },
                )
            plan = get_tool(
                "thinking_and_plan",
                **planning_input,
            )
            if step == 0:
                _print_json_block(
                    "TOOL OUTPUT",
                    {
                        "step": 1,
                        "tool_name": "thinking_and_plan",
                        "tool_result": plan,
                    },
                )

            planner_prompt = (
                f"步骤: {step + 1}/{max_steps}\n"
                f"{plan['planner_prompt']}"
            )
            planner_text = self._generate(planner_prompt, max_new_tokens=max_new_tokens)

            try:
                decision = _extract_json_object(planner_text)
            except Exception:
                decision = {
                    "action": "respond",
                    "response": planner_text,
                }
            decision = _normalize_decision(decision)

            action = decision.get("action", "respond")

            if action == "respond" or not allow_tools:
                if allow_tools and action == "respond":
                    response_text = decision.get("response", "")
                    if isinstance(response_text, str):
                        embedded_call = _maybe_extract_embedded_tool_call(response_text)
                        if embedded_call:
                            decision = embedded_call
                            action = "call_tool"

            if action == "respond" or not allow_tools:
                answer = decision.get("response") or self._generate(user_query, max_new_tokens=max_new_tokens)
                return {
                    "decision": decision,
                    "tool_history": tool_history,
                    "answer": answer,
                }

            if action != "call_tool":
                return {
                    "decision": decision,
                    "tool_history": tool_history,
                    "answer": "模型未返回可执行动作，流程结束。",
                }

            tool_name = decision.get("tool_name", "")
            tool_args = decision.get("tool_args", {})
            redirected = self._maybe_redirect_tool_call(tool_name, tool_args, tool_history)
            tool_name = redirected["tool_name"]
            tool_args = redirected["tool_args"]

            if tool_name == "report_making" and not _user_requests_report(user_query):
                tool_name = "tool_ending"
                tool_args = {}

            if tool_name in {"thinking_and_plan", "tthinking_and_plan"}:
                tool_result = plan
            else:
                try:
                    _print_json_block(
                        "TOOL INPUT",
                        {
                            "step": step + 1,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                        },
                    )
                    tool_result = self._dispatch_tool(tool_name, tool_args, user_query, tool_history, max_new_tokens)
                    _print_json_block(
                        "TOOL OUTPUT",
                        {
                            "step": step + 1,
                            "tool_name": tool_name,
                            "tool_result": tool_result,
                        },
                    )
                except Exception as exc:
                    _print_json_block(
                        "TOOL OUTPUT",
                        {
                            "step": step + 1,
                            "tool_name": tool_name,
                            "error": str(exc),
                        },
                    )
                    return {
                        "decision": decision,
                        "tool_history": tool_history,
                        "tool_result": {
                            "error": str(exc),
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                        },
                        "answer": f"工具调用失败: {exc}",
                    }

            history_item = {
                "step": step + 1,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": tool_result,
            }
            tool_history.append(history_item)

            if tool_name == "tool_ending":
                handoff = tool_result.get("llm_handoff", json.dumps(tool_history, ensure_ascii=False))
                final_prompt = _build_final_answer_prompt(user_query, handoff)
                final_answer = self._generate(final_prompt, max_new_tokens=max_new_tokens)
                return {
                    "decision": decision,
                    "tool_history": tool_history,
                    "answer": final_answer,
                }

        # Fallback when max steps reached: auto analysis (if needed) + ending.
        if any(item.get("tool_name") == "rescuenet_segmentation" for item in tool_history) and not any(
            item.get("tool_name") == "image_analysis" for item in tool_history
        ):
            analysis = self._run_image_analysis({}, user_query, tool_history, max_new_tokens)
            tool_history.append(
                {
                    "step": len(tool_history) + 1,
                    "tool_name": "image_analysis",
                    "tool_args": {},
                    "result": analysis,
                }
            )

        report_text = ""
        if _user_requests_report(user_query):
            report = get_tool("report_making", user_query=user_query, tool_history=tool_history)
            tool_history.append(
                {
                    "step": len(tool_history) + 1,
                    "tool_name": "report_making",
                    "tool_args": {},
                    "result": report,
                }
            )
            report_text = report.get("report", "")

        ending = get_tool(
            "tool_ending",
            user_query=user_query,
            tool_history=tool_history,
            report=report_text,
        )
        tool_history.append(
            {
                "step": len(tool_history) + 1,
                "tool_name": "tool_ending",
                "tool_args": {},
                "result": ending,
            }
        )

        final_prompt = _build_final_answer_prompt(user_query, ending.get("llm_handoff", ""))
        final_answer = self._generate(final_prompt, max_new_tokens=max_new_tokens)
        return {
            "decision": {"action": "auto_end"},
            "tool_history": tool_history,
            "answer": final_answer,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen local agent with multi-step tool calling")
    parser.add_argument("--query", type=str, required=True, help="User query text")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Local Qwen model directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Run device")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum generated tokens")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool-calling steps")
    parser.add_argument("--allow-tools", action="store_true", help="Allow the model to call tools")
    parser.add_argument("--no-tools", action="store_true", help="Force disable tool calling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    allow_tools = args.allow_tools
    if args.no_tools:
        allow_tools = False

    agent = QwenToolAgent(model_dir=args.model_dir, device=args.device)
    result = agent.run(
        user_query=args.query,
        allow_tools=allow_tools,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.max_steps,
    )

    print(json.dumps(_sanitize_for_display(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
