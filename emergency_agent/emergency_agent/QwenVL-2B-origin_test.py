import argparse
import difflib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate QwenVL 2B origin model on valid set.")
	parser.add_argument(
		"--dataset-file",
		type=str,
		default="/root/autodl-tmp/labels/labels_Qwen/valid/annotations.json",
	)
	parser.add_argument(
		"--labels-root",
		type=str,
		default="/root/autodl-tmp/labels/labels_Qwen",
	)
	parser.add_argument(
		"--label-to-class-file",
		type=str,
		default="/root/autodl-tmp/labels/RescueNet/label_to_class.json",
	)
	parser.add_argument(
		"--model-dir",
		type=str,
		default="/root/autodl-tmp/models/Qwen3-VL-2B-Instruct",
	)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument(
		"--attn-implementation",
		type=str,
		default="flash_attention_2",
		choices=["flash_attention_2", "sdpa", "eager"],
	)
	parser.add_argument("--max-samples", type=int, default=0)
	parser.add_argument(
		"--image-size",
		type=int,
		default=512,
		help="Resize input image to fixed square size (image_size x image_size).",
	)
	parser.add_argument("--max-new-tokens", type=int, default=32)
	parser.add_argument(
		"--infer-batch-size",
		type=int,
		default=2,
		help="Batch size for one generate() call.",
	)
	parser.add_argument(
		"--io-workers",
		type=int,
		default=min(32, (os.cpu_count() or 8) * 2),
		help="Number of threads for image file preloading.",
	)
	parser.add_argument(
		"--torch-threads",
		type=int,
		default=0,
		help="CPU thread count for torch; <=0 keeps default.",
	)
	parser.add_argument(
		"--output-json",
		type=str,
		default="/root/autodl-tmp/Outputs/qwen_valid_eval_2b_origin.json",
	)
	return parser.parse_args()


def resolve_device(device_arg: str) -> str:
	if device_arg == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return device_arg


def safe_read_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def configure_runtime(device: str, torch_threads: int) -> None:
	if torch_threads and torch_threads > 0:
		torch.set_num_threads(torch_threads)
		if hasattr(torch, "set_num_interop_threads"):
			torch.set_num_interop_threads(max(1, min(8, torch_threads // 2)))

	if device == "cuda":
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	if hasattr(torch, "set_float32_matmul_precision"):
		torch.set_float32_matmul_precision("high")


def normalize_text(text: str) -> str:
	clean = re.sub(r"[^a-z0-9\s/+-]", " ", str(text).lower())
	return " ".join(clean.strip().split())


def parse_answer_variants(answer: str) -> List[str]:
	text = str(answer).strip()
	if not text:
		return []
	parts = re.split(r"\s*(?:\||/|\bor\b|或)\s*", text, flags=re.IGNORECASE)
	variants = []
	for p in parts:
		norm = normalize_text(p)
		if norm:
			variants.append(norm)
	return list(dict.fromkeys(variants))


def load_label_vocab(label_to_class_file: str) -> List[str]:
	obj = safe_read_json(label_to_class_file)
	if not isinstance(obj, dict):
		raise ValueError(f"label_to_class must be a dict: {label_to_class_file}")
	labels = [str(v).strip() for v in obj.values() if str(v).strip()]
	if not labels:
		raise ValueError(f"No labels found in: {label_to_class_file}")
	# Preserve order while deduplicating.
	return list(dict.fromkeys(labels))


def normalize_pred_text(text: str) -> str:
	norm = normalize_text(text)
	norm = re.sub(r"^(answer|final answer)\s*", "", norm).strip()
	norm = re.sub(r"^(absolute|absolutely|definitely|certainly)\s+", "", norm).strip()
	norm = norm.replace("absolute_", "")
	norm = re.sub(r"^\((.*?)\)$", r"\1", norm).strip()
	return norm


def map_prediction_to_allowed(raw_pred: str, allowed_labels: List[str]) -> str:
	if not allowed_labels:
		return raw_pred.strip()

	allowed_map = {normalize_text(x): x for x in allowed_labels}
	raw_norm = normalize_text(raw_pred)
	clean_norm = normalize_pred_text(raw_pred)

	if clean_norm in allowed_map:
		return allowed_map[clean_norm]
	if raw_norm in allowed_map:
		return allowed_map[raw_norm]

	# Common boolean salvage for outputs like "absolute yes".
	if "yes" in clean_norm and "yes" in allowed_map:
		return allowed_map["yes"]
	if "no" in clean_norm and "no" in allowed_map:
		return allowed_map["no"]

	for norm_key, label in allowed_map.items():
		if clean_norm.startswith(norm_key) or norm_key.startswith(clean_norm):
			return label

	close = difflib.get_close_matches(clean_norm, list(allowed_map.keys()), n=1, cutoff=0.55)
	if close:
		return allowed_map[close[0]]

	# Fallback keeps behavior deterministic in-range.
	return allowed_labels[0]


def is_correct(pred: str, gold: str) -> bool:
	pred_norm = normalize_text(pred)
	gold_norm = normalize_text(gold)
	if pred_norm == gold_norm:
		return True
	variants = parse_answer_variants(gold)
	if variants and pred_norm in variants:
		return True
	if gold_norm and pred_norm.startswith(gold_norm):
		return True
	return False


def extract_user_image_path(conversations: List[dict]) -> Optional[str]:
	for turn in conversations:
		if turn.get("role") != "user":
			continue
		for item in turn.get("content", []):
			if item.get("type") == "image" and item.get("image"):
				return item["image"]
	return None


def extract_user_question(conversations: List[dict], fallback: str) -> str:
	for turn in conversations:
		if turn.get("role") != "user":
			continue
		texts = [c.get("text", "") for c in turn.get("content", []) if c.get("type") == "text"]
		if texts:
			return " ".join(t for t in texts if t).strip() or fallback
	return fallback


def extract_assistant_answer(conversations: List[dict]) -> str:
	for turn in conversations:
		if turn.get("role") != "assistant":
			continue
		texts = [c.get("text", "") for c in turn.get("content", []) if c.get("type") == "text"]
		answer = " ".join(t for t in texts if t).strip()
		if answer:
			return answer
	return ""


def load_eval_samples(dataset_file: str, max_samples: int = 0) -> List[dict]:
	rows = safe_read_json(dataset_file)
	if not isinstance(rows, list):
		raise ValueError(f"Dataset file must contain a list: {dataset_file}")

	samples = []
	for row in rows:
		conversations = row.get("conversations", [])
		image_path = extract_user_image_path(conversations)
		question = extract_user_question(conversations, str(row.get("subtype", "")).strip())
		answer = extract_assistant_answer(conversations)
		dtype = str(row.get("dtype", "unknown")).strip() or "unknown"
		subtype = str(row.get("subtype", "unknown")).strip() or "unknown"

		if not image_path or not os.path.isfile(image_path):
			continue
		if not question or not answer:
			continue

		samples.append(
			{
				"id": str(row.get("id", "")),
				"image_path": image_path,
				"question": question,
				"gold_answer": answer,
				"dtype": dtype,
				"subtype": subtype,
			}
		)

	if max_samples > 0:
		samples = samples[:max_samples]
	if not samples:
		raise ValueError("No valid evaluation samples were found.")
	return samples


def collect_answer_spaces(labels_root: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
	split_files = [
		os.path.join(labels_root, "train", "annotations.json"),
		os.path.join(labels_root, "valid", "annotations.json"),
		os.path.join(labels_root, "test", "annotations.json"),
	]
	dtype_to_answers = defaultdict(set)
	subtype_to_answers = defaultdict(set)
	for path in split_files:
		if not os.path.isfile(path):
			continue
		rows = safe_read_json(path)
		if not isinstance(rows, list):
			continue
		for row in rows:
			answer = extract_assistant_answer(row.get("conversations", []))
			if not answer:
				continue
			dtype = str(row.get("dtype", "unknown")).strip() or "unknown"
			subtype = str(row.get("subtype", "unknown")).strip() or "unknown"
			dtype_to_answers[dtype].add(answer)
			subtype_to_answers[subtype].add(answer)

	return ({k: sorted(v) for k, v in dtype_to_answers.items()}, {k: sorted(v) for k, v in subtype_to_answers.items()})


def load_qwen_model(model_dir: str, device: str, attn_implementation: str):
	dtype = torch.float16 if device == "cuda" else torch.float32
	kwargs = {
		"dtype": dtype,
		"trust_remote_code": True,
		"device_map": "auto" if device == "cuda" else None,
		"attn_implementation": attn_implementation,
	}
	try:
		model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs)
	except Exception:
		if attn_implementation != "eager":
			kwargs["attn_implementation"] = "eager"
			model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs)
		else:
			raise
	if device != "cuda":
		model.to(device)
	model.eval()
	return model


class BaseQwenPredictor:
	def __init__(self, model_dir: str, device: str, attn_implementation: str):
		if not os.path.isdir(model_dir):
			raise FileNotFoundError(f"Model directory not found: {model_dir}")
		self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
		self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
		self.model = load_qwen_model(model_dir, device, attn_implementation)

	@staticmethod
	def build_constrained_prompt(question: str, dtype_name: str, subtype_name: str, answer_space: List[str]) -> str:
		if not answer_space:
			return (
				f"Question Type(dType): {dtype_name}\n"
				f"Question SubType: {subtype_name}\n"
				f"Question: {question}\n"
				"Please answer briefly with only the final answer text."
			)
		return (
			f"Question Type(dType): {dtype_name}\n"
			f"Question SubType: {subtype_name}\n"
			f"Question: {question}\n"
			f"Candidate Answer Range: {' ; '.join(answer_space)}\n"
			"Choose exactly one answer from the candidate range. Return only the final answer text."
		)

	@torch.no_grad()
	def predict_batch(
		self,
		image_objs: List[Image.Image],
		questions: List[str],
		dtype_names: List[str],
		subtype_names: List[str],
		answer_spaces: List[List[str]],
		max_new_tokens: int,
	) -> List[str]:
		input_id_list = []
		attn_mask_list = []
		pixel_values_list = []
		image_grid_list = []

		for image_obj, question, dtype_name, subtype_name, answer_space in zip(
			image_objs,
			questions,
			dtype_names,
			subtype_names,
			answer_spaces,
		):
			prompt = self.build_constrained_prompt(question, dtype_name, subtype_name, answer_space)
			messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": prompt}]}]
			features = self.processor.apply_chat_template(
				messages,
				tokenize=True,
				add_generation_prompt=True,
				return_dict=True,
				return_tensors="pt",
			)
			input_id_list.append(features["input_ids"][0])
			attn_mask_list.append(features["attention_mask"][0])
			pixel_values_list.append(features["pixel_values"])
			image_grid_list.append(features["image_grid_thw"])

		padded = self.processor.tokenizer.pad(
			{"input_ids": input_id_list, "attention_mask": attn_mask_list},
			padding=True,
			return_tensors="pt",
		)

		model_inputs = {
			"input_ids": padded["input_ids"],
			"attention_mask": padded["attention_mask"],
			"pixel_values": torch.cat(pixel_values_list, dim=0),
			"image_grid_thw": torch.cat(image_grid_list, dim=0),
		}
		model_inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

		outputs = self.model.generate(
			**model_inputs,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			eos_token_id=self.tokenizer.eos_token_id,
			pad_token_id=self.tokenizer.eos_token_id,
		)

		input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
		generated_ids_trimmed = [out_ids[int(in_len):] for in_len, out_ids in zip(input_lengths, outputs)]
		decoded = self.processor.batch_decode(
			generated_ids_trimmed,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=False,
		)
		return [text.strip() for text in decoded]


def evaluate(
	predictor: BaseQwenPredictor,
	samples: List[dict],
	dtype_space: Dict[str, List[str]],
	subtype_space: Dict[str, List[str]],
	global_labels: List[str],
	image_size: int,
	max_new_tokens: int,
	io_workers: int,
	infer_batch_size: int,
) -> Dict[str, object]:
	per_dtype: Dict[str, Dict[str, object]] = {}
	predictions: List[dict] = []

	def load_item(sample: dict):
		with Image.open(sample["image_path"]) as img:
			return sample, img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)

	total_samples = len(samples)
	batch_size = max(1, infer_batch_size)
	for start in range(0, total_samples, batch_size):
		chunk = samples[start : start + batch_size]
		with ThreadPoolExecutor(max_workers=max(1, io_workers)) as executor:
			loaded = list(executor.map(load_item, chunk))

		chunk_samples = [x[0] for x in loaded]
		chunk_images = [x[1] for x in loaded]
		chunk_dtype_names = [x["dtype"] for x in chunk_samples]
		chunk_subtype_names = [x["subtype"] for x in chunk_samples]
		chunk_questions = [x["question"] for x in chunk_samples]
		chunk_answer_spaces = []
		for d, s in zip(chunk_dtype_names, chunk_subtype_names):
			raw_space = subtype_space.get(s, []) or dtype_space.get(d, []) or []
			if raw_space:
				allowed_norm = {normalize_text(x) for x in global_labels}
				filtered = [x for x in raw_space if normalize_text(x) in allowed_norm]
				chunk_answer_spaces.append(filtered if filtered else global_labels)
			else:
				chunk_answer_spaces.append(global_labels)

		chunk_preds = predictor.predict_batch(
			image_objs=chunk_images,
			questions=chunk_questions,
			dtype_names=chunk_dtype_names,
			subtype_names=chunk_subtype_names,
			answer_spaces=chunk_answer_spaces,
			max_new_tokens=max_new_tokens,
		)

		for offset, (sample, raw_pred, allowed_space) in enumerate(zip(chunk_samples, chunk_preds, chunk_answer_spaces), start=1):
			idx = start + offset
			dtype_name = sample["dtype"]
			pred = map_prediction_to_allowed(raw_pred, allowed_space)
			correct = is_correct(pred, sample["gold_answer"])

			if dtype_name not in per_dtype:
				per_dtype[dtype_name] = {"total": 0, "correct": 0, "accuracy": 0.0}
			per_dtype[dtype_name]["total"] += 1
			if correct:
				per_dtype[dtype_name]["correct"] += 1

			predictions.append(
				{
					"id": sample["id"],
					"image_path": sample["image_path"],
					"dtype": sample["dtype"],
					"subtype": sample["subtype"],
					"question": sample["question"],
					"gold_answer": sample["gold_answer"],
					"pred_raw": raw_pred,
					"pred_answer": pred,
					"correct": correct,
				}
			)

			if idx % 50 == 0 or idx == total_samples:
				print(f"[Qwen2B-origin] progress: {idx}/{total_samples}")

	for record in per_dtype.values():
		total = int(record["total"])
		correct = int(record["correct"])
		record["accuracy"] = (correct / total) if total > 0 else 0.0

	total = len(predictions)
	total_correct = sum(1 for x in predictions if x["correct"])
	return {
		"model_name": "Qwen2B-origin",
		"num_samples": total,
		"num_correct": total_correct,
		"overall_accuracy": (total_correct / total) if total > 0 else 0.0,
		"per_dtype": per_dtype,
		"predictions": predictions,
	}


def main() -> None:
	args = parse_args()
	device = resolve_device(args.device)
	configure_runtime(device, args.torch_threads)

	if not os.path.isfile(args.dataset_file):
		raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

	samples = load_eval_samples(args.dataset_file, args.max_samples)
	dtype_space, subtype_space = collect_answer_spaces(args.labels_root)
	global_labels = load_label_vocab(args.label_to_class_file)
	print(f"Loaded {len(samples)} samples. Device: {device}")

	predictor = BaseQwenPredictor(args.model_dir, device, args.attn_implementation)
	result = evaluate(
		predictor,
		samples,
		dtype_space,
		subtype_space,
		global_labels,
		args.image_size,
		args.max_new_tokens,
		args.io_workers,
		args.infer_batch_size,
	)

	os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
	with open(args.output_json, "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)

	print(f"Saved result to: {args.output_json}")
	print(f"overall_acc={result['overall_accuracy']:.4f}, correct={result['num_correct']}/{result['num_samples']}")


if __name__ == "__main__":
	main()