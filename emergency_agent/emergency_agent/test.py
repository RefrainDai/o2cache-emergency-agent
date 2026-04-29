import argparse
import gc
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare Qwen models on labels_Qwen valid set with type-constrained answer range."
	)
	parser.add_argument(
		"--dataset-file",
		type=str,
		default="/root/autodl-tmp/labels/labels_Qwen/valid/annotations.json",
		help="Validation annotations JSON file.",
	)
	parser.add_argument(
		"--labels-root",
		type=str,
		default="/root/autodl-tmp/labels/labels_Qwen",
		help="Root path containing train/valid/test annotations.",
	)
	parser.add_argument(
		"--models-root",
		type=str,
		default="/root/autodl-tmp/models",
		help="Root path containing Qwen model directories.",
	)
	parser.add_argument(
		"--classifier-dir",
		type=str,
		default="/root/autodl-tmp/models/Qwen3-VL-8B-FineTuning/ADD_CLASSFIER",
		help="Directory containing classifier_meta.json and classifier.pt.",
	)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument(
		"--attn-implementation",
		type=str,
		default="flash_attention_2",
		choices=["flash_attention_2", "sdpa", "eager"],
		help="Attention backend for base models.",
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=0,
		help="Evaluate first N samples from valid set; <=0 means all.",
	)
	parser.add_argument("--max-new-tokens", type=int, default=32)
	parser.add_argument(
		"--output-json",
		type=str,
		default="/root/autodl-tmp/Outputs/qwen_valid_eval_compare.json",
		help="Where to save detailed evaluation results.",
	)
	return parser.parse_args()


def resolve_device(device_arg: str) -> str:
	if device_arg == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return device_arg


def safe_read_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def normalize_text(text: str) -> str:
	clean = re.sub(r"[^a-z0-9\s/+-]", " ", str(text).lower())
	return " ".join(clean.strip().split())


def parse_answer_variants(answer: str) -> List[str]:
	text = str(answer).strip()
	if not text:
		return []

	parts = re.split(r"\s*(?:\||/|\\bor\\b|\\b或\\b)\s*", text, flags=re.IGNORECASE)
	variants = []
	for p in parts:
		norm = normalize_text(p)
		if norm:
			variants.append(norm)
	return list(dict.fromkeys(variants))


def is_correct(pred: str, gold: str) -> bool:
	pred_norm = normalize_text(pred)
	gold_norm = normalize_text(gold)
	if pred_norm == gold_norm:
		return True

	if pred_norm in {"yes", "no"} and gold_norm in {"yes", "no"}:
		return pred_norm == gold_norm

	variants = parse_answer_variants(gold)
	if variants and pred_norm in variants:
		return True

	# Handle cases where model includes brief explanation after the label.
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
			conversations = row.get("conversations", [])
			answer = extract_assistant_answer(conversations)
			if not answer:
				continue
			dtype = str(row.get("dtype", "unknown")).strip() or "unknown"
			subtype = str(row.get("subtype", "unknown")).strip() or "unknown"
			dtype_to_answers[dtype].add(answer)
			subtype_to_answers[subtype].add(answer)

	dtype_space = {k: sorted(v) for k, v in dtype_to_answers.items()}
	subtype_space = {k: sorted(v) for k, v in subtype_to_answers.items()}
	return dtype_space, subtype_space


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
		self.model_dir = model_dir
		self.device = device
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

		options = " ; ".join(answer_space)
		return (
			f"Question Type(dType): {dtype_name}\n"
			f"Question SubType: {subtype_name}\n"
			f"Question: {question}\n"
			f"Candidate Answer Range: {options}\n"
			"Choose exactly one answer from the candidate range. "
			"Return only the final answer text without explanation."
		)

	@torch.no_grad()
	def predict(self, image_path: str, question: str, dtype_name: str, subtype_name: str, answer_space: List[str], max_new_tokens: int) -> str:
		image = Image.open(image_path).convert("RGB")
		prompt = self.build_constrained_prompt(question, dtype_name, subtype_name, answer_space)
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image},
					{"type": "text", "text": prompt},
				],
			}
		]

		model_inputs = self.processor.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=True,
			return_dict=True,
			return_tensors="pt",
		)
		model_inputs = {
			k: v.to(self.model.device) if hasattr(v, "to") else v
			for k, v in model_inputs.items()
		}

		outputs = self.model.generate(
			**model_inputs,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			eos_token_id=self.tokenizer.eos_token_id,
			pad_token_id=self.tokenizer.eos_token_id,
		)

		generated_ids_trimmed = [
			out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], outputs)
		]
		decoded = self.processor.batch_decode(
			generated_ids_trimmed,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=False,
		)
		return decoded[0].strip() if decoded else ""


class ClassifierHead(nn.Module):
	def __init__(self, in_dim: int, hidden_dim: int, num_labels: int, dropout: float, feature_dropout: float):
		super().__init__()
		self.net = nn.Sequential(
			nn.Dropout(feature_dropout),
			nn.Linear(in_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_labels),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
	moved = {}
	for k, v in batch.items():
		if isinstance(v, torch.Tensor):
			moved[k] = v.to(device)
		else:
			moved[k] = v
	return moved


def gather_last_token_features(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	lengths = attention_mask.long().sum(dim=1) - 1
	lengths = torch.clamp(lengths, min=0)
	batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
	return hidden_states[batch_idx, lengths, :]


def apply_dtype_mask(logits: torch.Tensor, dtype_name: str, dtype_mask: Dict[str, torch.Tensor]) -> torch.Tensor:
	masked = logits.clone()
	mask = dtype_mask.get(dtype_name)
	if mask is None:
		return masked
	row_mask = mask.to(logits.device)
	masked[0, ~row_mask] = -1e9
	return masked


def qwen_forward_hidden(
	qwen_model,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	pixel_values: torch.Tensor,
	image_grid_thw: torch.Tensor,
) -> torch.Tensor:
	if hasattr(qwen_model, "model"):
		out = qwen_model.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			pixel_values=pixel_values,
			image_grid_thw=image_grid_thw,
			output_hidden_states=True,
			use_cache=False,
		)
	else:
		out = qwen_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			pixel_values=pixel_values,
			image_grid_thw=image_grid_thw,
			output_hidden_states=True,
			use_cache=False,
		)
	return out.hidden_states[-1]


class QwenClassifierPredictor:
	def __init__(self, base_model_dir: str, classifier_dir: str, device: str):
		meta_path = os.path.join(classifier_dir, "classifier_meta.json")
		cls_path = os.path.join(classifier_dir, "classifier.pt")
		if not os.path.isfile(meta_path) or not os.path.isfile(cls_path):
			raise FileNotFoundError("classifier_meta.json or classifier.pt not found.")

		self.device = device
		self.meta = safe_read_json(meta_path)
		attn_impl = self.meta.get("attn_implementation", "flash_attention_2")

		self.processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
		self.qwen = load_qwen_model(base_model_dir, device, attn_impl)
		for p in self.qwen.parameters():
			p.requires_grad = False
		self.qwen.eval()

		self.classifier = ClassifierHead(
			in_dim=int(self.meta["feature_dim"]),
			hidden_dim=int(self.meta["hidden_size"]),
			num_labels=int(self.meta["num_labels"]),
			dropout=float(self.meta["dropout"]),
			feature_dropout=float(self.meta.get("feature_dropout", 0.1)),
		).to(device)
		state = torch.load(cls_path, map_location=device)
		self.classifier.load_state_dict(state)
		self.classifier.eval()

		self.id2label = {int(k): v for k, v in self.meta["id2label"].items()}
		self.dtype_mask = {}
		for dtype_name, ids in self.meta["dtype_allowed_label_ids"].items():
			mask = torch.zeros(int(self.meta["num_labels"]), dtype=torch.bool)
			mask[torch.tensor(ids, dtype=torch.long)] = True
			self.dtype_mask[dtype_name] = mask

	@torch.no_grad()
	def predict(self, image_path: str, question: str, dtype_name: str) -> str:
		image_obj = Image.open(image_path).convert("RGB")
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image_obj},
					{"type": "text", "text": question},
				],
			}
		]
		batch = self.processor.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=True,
			return_dict=True,
			return_tensors="pt",
		)
		batch = move_batch_to_device(batch, self.device)

		hidden = qwen_forward_hidden(
			qwen_model=self.qwen,
			input_ids=batch["input_ids"],
			attention_mask=batch["attention_mask"],
			pixel_values=batch["pixel_values"],
			image_grid_thw=batch["image_grid_thw"],
		)
		feat = gather_last_token_features(hidden, batch["attention_mask"])
		if feat.dtype != next(self.classifier.parameters()).dtype:
			feat = feat.to(next(self.classifier.parameters()).dtype)

		logits = self.classifier(feat)
		masked = apply_dtype_mask(logits, dtype_name, self.dtype_mask)
		pred_id = int(torch.argmax(masked, dim=-1).item())
		return str(self.id2label[pred_id])


def init_stats_bucket() -> Dict[str, object]:
	return {"total": 0, "correct": 0, "accuracy": 0.0}


def update_stats(stats: Dict[str, Dict[str, object]], dtype_name: str, correct: bool) -> None:
	if dtype_name not in stats:
		stats[dtype_name] = init_stats_bucket()
	stats[dtype_name]["total"] += 1
	if correct:
		stats[dtype_name]["correct"] += 1


def finalize_stats(stats: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
	for dtype_name, record in stats.items():
		total = int(record["total"])
		correct = int(record["correct"])
		record["accuracy"] = (correct / total) if total > 0 else 0.0
	return stats


def evaluate_model(
	model_name: str,
	predictor,
	samples: List[dict],
	dtype_space: Dict[str, List[str]],
	subtype_space: Dict[str, List[str]],
	max_new_tokens: int,
	use_classifier: bool,
) -> Dict[str, object]:
	results = []
	per_dtype = {}

	for idx, sample in enumerate(samples, start=1):
		dtype_name = sample["dtype"]
		subtype_name = sample["subtype"]
		candidate_answers = subtype_space.get(subtype_name) or dtype_space.get(dtype_name) or []

		if use_classifier:
			pred = predictor.predict(
				image_path=sample["image_path"],
				question=sample["question"],
				dtype_name=dtype_name,
			)
		else:
			pred = predictor.predict(
				image_path=sample["image_path"],
				question=sample["question"],
				dtype_name=dtype_name,
				subtype_name=subtype_name,
				answer_space=candidate_answers,
				max_new_tokens=max_new_tokens,
			)

		correct = is_correct(pred, sample["gold_answer"])
		update_stats(per_dtype, dtype_name, correct)
		results.append(
			{
				"id": sample["id"],
				"image_path": sample["image_path"],
				"dtype": dtype_name,
				"subtype": subtype_name,
				"question": sample["question"],
				"gold_answer": sample["gold_answer"],
				"pred_answer": pred,
				"correct": correct,
			}
		)

		if idx % 50 == 0 or idx == len(samples):
			print(f"[{model_name}] progress: {idx}/{len(samples)}")

	per_dtype = finalize_stats(per_dtype)
	total = len(results)
	total_correct = sum(1 for item in results if item["correct"])
	overall_acc = (total_correct / total) if total > 0 else 0.0

	return {
		"model_name": model_name,
		"num_samples": total,
		"num_correct": total_correct,
		"overall_accuracy": overall_acc,
		"per_dtype": per_dtype,
		"predictions": results,
	}


def cleanup_cuda() -> None:
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def print_summary(report: Dict[str, object]) -> None:
	print("\n========== Comparison Summary ==========")
	for model_result in report["results"]:
		print(
			f"[{model_result['model_name']}] "
			f"samples={model_result['num_samples']} "
			f"correct={model_result['num_correct']} "
			f"acc={model_result['overall_accuracy']:.4f}"
		)
		for dtype_name, rec in sorted(model_result["per_dtype"].items()):
			print(
				f"  - {dtype_name}: total={rec['total']} "
				f"correct={rec['correct']} acc={rec['accuracy']:.4f}"
			)


def resolve_model_paths(models_root: str) -> Dict[str, str]:
	qwen_2b = os.path.join(models_root, "Qwen3-VL-2B-Instruct")

	qwen_8b = os.path.join(models_root, "Qwen3-VL-8B-Instruct")
	if not os.path.isdir(qwen_8b):
		raise FileNotFoundError(f"Qwen 8B model dir not found: {qwen_8b}")

	if not os.path.isdir(qwen_2b):
		raise FileNotFoundError(f"Qwen 2B model dir not found: {qwen_2b}")

	return {
		"qwen_2b": qwen_2b,
		"qwen_8b": qwen_8b,
	}


def main() -> None:
	args = parse_args()
	device = resolve_device(args.device)

	if not os.path.isfile(args.dataset_file):
		raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")
	if not os.path.isdir(args.models_root):
		raise FileNotFoundError(f"Models root not found: {args.models_root}")

	samples = load_eval_samples(args.dataset_file, args.max_samples)
	dtype_space, subtype_space = collect_answer_spaces(args.labels_root)
	model_paths = resolve_model_paths(args.models_root)

	print(f"Loaded {len(samples)} evaluation samples.")
	print(f"Device: {device}")

	all_results = []

	print("\n[1/3] Evaluating Qwen2B base model...")
	predictor_2b = BaseQwenPredictor(
		model_dir=model_paths["qwen_2b"],
		device=device,
		attn_implementation=args.attn_implementation,
	)
	res_2b = evaluate_model(
		model_name="Qwen2B-base",
		predictor=predictor_2b,
		samples=samples,
		dtype_space=dtype_space,
		subtype_space=subtype_space,
		max_new_tokens=args.max_new_tokens,
		use_classifier=False,
	)
	all_results.append(res_2b)
	del predictor_2b
	cleanup_cuda()

	print("\n[2/3] Evaluating Qwen8B base model...")
	predictor_8b = BaseQwenPredictor(
		model_dir=model_paths["qwen_8b"],
		device=device,
		attn_implementation=args.attn_implementation,
	)
	res_8b = evaluate_model(
		model_name="Qwen8B-base",
		predictor=predictor_8b,
		samples=samples,
		dtype_space=dtype_space,
		subtype_space=subtype_space,
		max_new_tokens=args.max_new_tokens,
		use_classifier=False,
	)
	all_results.append(res_8b)
	del predictor_8b
	cleanup_cuda()

	print("\n[3/3] Evaluating Qwen8B + classifier model...")
	predictor_cls = QwenClassifierPredictor(
		base_model_dir=model_paths["qwen_8b"],
		classifier_dir=args.classifier_dir,
		device=device,
	)
	res_cls = evaluate_model(
		model_name="Qwen8B+classifier",
		predictor=predictor_cls,
		samples=samples,
		dtype_space=dtype_space,
		subtype_space=subtype_space,
		max_new_tokens=args.max_new_tokens,
		use_classifier=True,
	)
	all_results.append(res_cls)
	del predictor_cls
	cleanup_cuda()

	report = {
		"dataset_file": args.dataset_file,
		"num_samples": len(samples),
		"device": device,
		"models": {
			"qwen_2b": model_paths["qwen_2b"],
			"qwen_8b": model_paths["qwen_8b"],
			"qwen_8b_classifier": args.classifier_dir,
		},
		"results": all_results,
	}

	os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
	with open(args.output_json, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)

	print_summary(report)
	print(f"\nSaved evaluation report to: {args.output_json}")


if __name__ == "__main__":
	main()
