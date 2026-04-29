import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL_DIR = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
DEFAULT_LABELS_DIR = "/root/autodl-tmp/labels/labels_Qwen"
DEFAULT_SAVE_DIR = "/root/autodl-tmp/models/Qwen3-VL-8B-FineTuning/ADD_CLASSFIER"
DEFAULT_TB_LOG_DIR = "/root/autodl-tmp/tb_plots"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train a classifier head with dtype mask on top of frozen Qwen3-VL features."
	)
	parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
	parser.add_argument("--labels-dir", type=str, default=DEFAULT_LABELS_DIR)
	parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--log-steps", type=int, default=50)
	parser.add_argument(
		"--early-stop-patience",
		type=int,
		default=8,
		help="Stop training when validation metric does not improve for N consecutive epochs. <=0 disables early stopping.",
	)
	parser.add_argument(
		"--early-stop-min-delta",
		type=float,
		default=1e-4,
		help="Minimum validation accuracy improvement to reset early stopping patience.",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Resume from save-dir/last_training_state.pt if exists.",
	)
	parser.add_argument(
		"--resume-from",
		type=str,
		default="",
		help="Path to a training state checkpoint .pt for continuing training.",
	)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-2)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--feature-dropout", type=float, default=0.1)
	parser.add_argument("--hidden-size", type=int, default=2048)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument(
		"--max-image-side",
		type=int,
		default=448,
		help="Resize image so the longer side is <= this value before feeding Qwen-VL. <=0 disables resize.",
	)
	parser.add_argument("--max-train-samples", type=int, default=0)
	parser.add_argument("--max-valid-samples", type=int, default=0)
	parser.add_argument("--max-test-samples", type=int, default=0)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument(
		"--attn-implementation",
		type=str,
		default="flash_attention_2",
		choices=["flash_attention_2", "sdpa", "eager"],
		help="Attention backend for Qwen model loading.",
	)
	parser.add_argument("--save-best-only", action="store_true")
	parser.add_argument(
		"--tb-log-dir",
		type=str,
		default=DEFAULT_TB_LOG_DIR,
		help="TensorBoard log root directory.",
	)
	parser.add_argument(
		"--tb-port",
		type=int,
		default=6007,
		help="Suggested TensorBoard port for viewing logs.",
	)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
	if device_arg == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return device_arg


def safe_read_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def normalize_text(text: str) -> str:
	return " ".join(str(text).strip().lower().split())


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


def extract_assistant_answer(conversations: List[dict]) -> Optional[str]:
	for turn in conversations:
		if turn.get("role") != "assistant":
			continue
		texts = [c.get("text", "") for c in turn.get("content", []) if c.get("type") == "text"]
		answer = " ".join(t for t in texts if t).strip()
		if answer:
			return answer
	return None


@dataclass
class Sample:
	image_path: str
	question: str
	dtype: str
	label_id: int


def load_split_samples(split_file: str) -> List[dict]:
	raw = safe_read_json(split_file)
	if not isinstance(raw, list):
		raise ValueError(f"Split file must be a list: {split_file}")
	return raw


def build_label_vocab(train_rows: List[dict]) -> Dict[str, int]:
	labels = set()
	for row in train_rows:
		ans = extract_assistant_answer(row.get("conversations", []))
		if ans:
			labels.add(normalize_text(ans))
	if not labels:
		raise ValueError("No labels found in training annotations.")
	return {label: idx for idx, label in enumerate(sorted(labels))}


def rows_to_samples(rows: List[dict], label2id: Dict[str, int], split_name: str) -> List[Sample]:
	samples: List[Sample] = []
	dropped = 0
	for row in rows:
		conversations = row.get("conversations", [])
		image_path = extract_user_image_path(conversations)
		answer = extract_assistant_answer(conversations)
		dtype = str(row.get("dtype", "unknown")).strip() or "unknown"
		subtype = str(row.get("subtype", "")).strip()

		if not image_path or not os.path.isfile(image_path) or not answer:
			dropped += 1
			continue

		label_key = normalize_text(answer)
		if label_key not in label2id:
			dropped += 1
			continue

		question = extract_user_question(conversations, fallback=subtype)
		if not question:
			question = subtype if subtype else "Describe this image."

		samples.append(
			Sample(
				image_path=image_path,
				question=question,
				dtype=dtype,
				label_id=label2id[label_key],
			)
		)

	print(f"[{split_name}] kept={len(samples)}, dropped={dropped}")
	if not samples:
		raise ValueError(f"No valid samples after filtering in split {split_name}.")
	return samples


def maybe_limit_samples(samples: List[Sample], limit: int, split_name: str) -> List[Sample]:
	if limit is None or limit <= 0:
		return samples
	limited = samples[:limit]
	print(f"[{split_name}] limited to {len(limited)} samples by user setting")
	return limited


def build_dtype_mask(samples: List[Sample], num_labels: int) -> Dict[str, torch.Tensor]:
	dtype_to_labels: Dict[str, set] = {}
	for s in samples:
		dtype_to_labels.setdefault(s.dtype, set()).add(s.label_id)

	masks: Dict[str, torch.Tensor] = {}
	for dtype, label_set in dtype_to_labels.items():
		mask = torch.zeros(num_labels, dtype=torch.bool)
		label_ids = torch.tensor(sorted(label_set), dtype=torch.long)
		mask[label_ids] = True
		masks[dtype] = mask
	return masks


class QwenClassifierDataset(Dataset):
	def __init__(self, samples: List[Sample]):
		self.samples = samples

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Sample:
		return self.samples[idx]


class Collator:
	def __init__(self, processor, max_length: int, max_image_side: int):
		self.processor = processor
		self.max_length = max_length
		self.max_image_side = max_image_side

	def _load_image(self, image_path: str) -> Image.Image:
		image = Image.open(image_path).convert("RGB")
		if self.max_image_side is None or self.max_image_side <= 0:
			return image

		w, h = image.size
		max_side = max(w, h)
		if max_side <= self.max_image_side:
			return image

		scale = self.max_image_side / float(max_side)
		new_w = max(1, int(round(w * scale)))
		new_h = max(1, int(round(h * scale)))
		return image.resize((new_w, new_h), Image.BILINEAR)

	def __call__(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
		input_id_list: List[torch.Tensor] = []
		attn_mask_list: List[torch.Tensor] = []
		pixel_values_list: List[torch.Tensor] = []
		image_grid_list: List[torch.Tensor] = []
		labels: List[int] = []
		dtypes: List[str] = []

		for sample in batch:
			image_obj = self._load_image(sample.image_path)
			messages = [
				{
					"role": "user",
					"content": [
						{"type": "image", "image": image_obj},
						{"type": "text", "text": sample.question},
					],
				}
			]
			features = self.processor.apply_chat_template(
				messages,
				tokenize=True,
				add_generation_prompt=True,
				return_dict=True,
				return_tensors="pt",
			)

			input_ids = features["input_ids"][0]
			attention_mask = features["attention_mask"][0]

			# Do not truncate multimodal tokens manually.
			# For Qwen-VL, cutting input_ids can remove image tokens while keeping
			# pixel features unchanged, causing token-feature mismatch errors.

			input_id_list.append(input_ids)
			attn_mask_list.append(attention_mask)
			pixel_values_list.append(features["pixel_values"])
			image_grid_list.append(features["image_grid_thw"])
			labels.append(sample.label_id)
			dtypes.append(sample.dtype)

		padded = self.processor.tokenizer.pad(
			{
				"input_ids": input_id_list,
				"attention_mask": attn_mask_list,
			},
			padding=True,
			return_tensors="pt",
		)

		return {
			"input_ids": padded["input_ids"],
			"attention_mask": padded["attention_mask"],
			"pixel_values": torch.cat(pixel_values_list, dim=0),
			"image_grid_thw": torch.cat(image_grid_list, dim=0),
			"labels": torch.tensor(labels, dtype=torch.long),
			"dtypes": dtypes,
		}


class ClassifierHead(nn.Module):
	def __init__(
		self,
		in_dim: int,
		hidden_dim: int,
		num_labels: int,
		dropout: float,
		feature_dropout: float,
	):
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


def apply_dtype_mask(logits: torch.Tensor, dtypes: List[str], dtype_mask: Dict[str, torch.Tensor]) -> torch.Tensor:
	masked = logits.clone()
	for i, dtype in enumerate(dtypes):
		mask = dtype_mask.get(dtype)
		if mask is None:
			continue
		row_mask = mask.to(logits.device)
		masked[i, ~row_mask] = -1e9
	return masked


def align_feature_dtype(features: torch.Tensor, module: nn.Module) -> torch.Tensor:
	target_dtype = next(module.parameters()).dtype
	if features.dtype != target_dtype:
		return features.to(target_dtype)
	return features


def qwen_forward_hidden(
	qwen_model,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	pixel_values: torch.Tensor,
	image_grid_thw: torch.Tensor,
) -> torch.Tensor:
	# Prefer backbone forward to avoid lm_head logits allocation and reduce VRAM.
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
	except Exception as exc:
		if attn_implementation != "eager":
			warnings.warn(
				f"Failed to load with attn_implementation={attn_implementation}: {exc}. "
				"Falling back to eager."
			)
			kwargs["attn_implementation"] = "eager"
			model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs)
		else:
			raise
	if device != "cuda":
		model.to(device)
	return model


def save_training_state(
	checkpoint_path: str,
	epoch: int,
	classifier: nn.Module,
	optimizer: torch.optim.Optimizer,
	best_state: Optional[Dict[str, torch.Tensor]],
	best_valid_acc: float,
	best_epoch: int,
	no_improve_epochs: int,
) -> None:
	state = {
		"epoch": epoch,
		"classifier_state": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
		"optimizer_state": optimizer.state_dict(),
		"best_state": best_state,
		"best_valid_acc": best_valid_acc,
		"best_epoch": best_epoch,
		"no_improve_epochs": no_improve_epochs,
	}
	torch.save(state, checkpoint_path)


def load_training_state(checkpoint_path: str, device: str):
	state = torch.load(checkpoint_path, map_location=device)
	if "classifier_state" not in state or "optimizer_state" not in state:
		raise ValueError(f"Invalid checkpoint file: {checkpoint_path}")
	return state


@torch.no_grad()
def evaluate(
	qwen_model,
	classifier,
	dataloader,
	dtype_mask: Dict[str, torch.Tensor],
	loss_fn,
	device: str,
) -> Dict[str, float]:
	classifier.eval()
	total_loss = 0.0
	total = 0
	correct = 0

	for batch in dataloader:
		batch = move_batch_to_device(batch, device)
		hidden = qwen_forward_hidden(
			qwen_model=qwen_model,
			input_ids=batch["input_ids"],
			attention_mask=batch["attention_mask"],
			pixel_values=batch["pixel_values"],
			image_grid_thw=batch["image_grid_thw"],
		)
		feats = gather_last_token_features(hidden, batch["attention_mask"])
		feats = align_feature_dtype(feats, classifier)
		logits = classifier(feats)
		masked_logits = apply_dtype_mask(logits, batch["dtypes"], dtype_mask)
		loss = loss_fn(masked_logits, batch["labels"])

		preds = torch.argmax(masked_logits, dim=-1)
		correct += (preds == batch["labels"]).sum().item()
		total += batch["labels"].shape[0]
		total_loss += loss.item() * batch["labels"].shape[0]

	avg_loss = total_loss / max(total, 1)
	acc = correct / max(total, 1)
	return {"loss": avg_loss, "acc": acc}


def train(args: argparse.Namespace) -> None:
	set_seed(args.seed)

	model_dir = os.path.abspath(args.model_dir)
	labels_dir = os.path.abspath(args.labels_dir)
	save_dir = os.path.abspath(args.save_dir)
	tb_log_root = os.path.abspath(args.tb_log_dir)
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(tb_log_root, exist_ok=True)

	run_name = time.strftime("qwen_cls_%Y%m%d_%H%M%S")
	tb_run_dir = os.path.join(tb_log_root, run_name)
	writer = SummaryWriter(log_dir=tb_run_dir)
	writer.add_text(
		"run/args",
		"```json\n" + json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n```",
		global_step=0,
	)

	train_file = os.path.join(labels_dir, "train", "annotations.json")
	valid_file = os.path.join(labels_dir, "valid", "annotations.json")
	test_file = os.path.join(labels_dir, "test", "annotations.json")

	for p in [model_dir, train_file, valid_file, test_file]:
		if not os.path.exists(p):
			raise FileNotFoundError(f"Required path not found: {p}")

	print(f"Loading model from: {model_dir}")
	device = resolve_device(args.device)
	print(f"Training device: {device}")
	print(f"Attention backend: {args.attn_implementation}")
	print(f"TensorBoard log dir: {tb_run_dir}")
	print(
		"TensorBoard start command: "
		f"tensorboard --logdir {tb_log_root} --port {args.tb_port} --host 0.0.0.0"
	)

	processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
	qwen_model = load_qwen_model(
		model_dir=model_dir,
		device=device,
		attn_implementation=args.attn_implementation,
	)

	# Freeze all base model params so only classifier is trained.
	for p in qwen_model.parameters():
		p.requires_grad = False
	qwen_model.eval()

	train_rows = load_split_samples(train_file)
	valid_rows = load_split_samples(valid_file)
	test_rows = load_split_samples(test_file)

	label2id = build_label_vocab(train_rows)
	id2label = {idx: label for label, idx in label2id.items()}
	num_labels = len(label2id)
	print(f"Num classes (multi-class labels): {num_labels}")

	train_samples = rows_to_samples(train_rows, label2id, "train")
	valid_samples = rows_to_samples(valid_rows, label2id, "valid")
	test_samples = rows_to_samples(test_rows, label2id, "test")

	train_samples = maybe_limit_samples(train_samples, args.max_train_samples, "train")
	valid_samples = maybe_limit_samples(valid_samples, args.max_valid_samples, "valid")
	test_samples = maybe_limit_samples(test_samples, args.max_test_samples, "test")

	dtype_mask = build_dtype_mask(train_samples, num_labels)
	print(f"Num dtype masks: {len(dtype_mask)}")

	collator = Collator(
		processor=processor,
		max_length=args.max_length,
		max_image_side=args.max_image_side,
	)
	train_loader = DataLoader(
		QwenClassifierDataset(train_samples),
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		collate_fn=collator,
		pin_memory=(device == "cuda"),
	)
	valid_loader = DataLoader(
		QwenClassifierDataset(valid_samples),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collator,
		pin_memory=(device == "cuda"),
	)
	test_loader = DataLoader(
		QwenClassifierDataset(test_samples),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collator,
		pin_memory=(device == "cuda"),
	)

	with torch.no_grad():
		probe_batch = next(iter(train_loader))
		probe_batch = move_batch_to_device(probe_batch, device)
		probe_hidden = qwen_forward_hidden(
			qwen_model=qwen_model,
			input_ids=probe_batch["input_ids"],
			attention_mask=probe_batch["attention_mask"],
			pixel_values=probe_batch["pixel_values"],
			image_grid_thw=probe_batch["image_grid_thw"],
		)
		feature_dim = probe_hidden.shape[-1]

	classifier = ClassifierHead(
		in_dim=feature_dim,
		hidden_dim=args.hidden_size,
		num_labels=num_labels,
		dropout=args.dropout,
		feature_dropout=args.feature_dropout,
	).to(device)

	optimizer = torch.optim.AdamW(
		classifier.parameters(),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)
	loss_fn = nn.CrossEntropyLoss()

	best_valid_acc = -1.0
	best_state = None
	best_epoch = -1
	no_improve_epochs = 0
	start_epoch = 1

	last_state_path = os.path.join(save_dir, "last_training_state.pt")
	resume_path = ""
	if args.resume_from:
		resume_path = os.path.abspath(args.resume_from)
	elif args.resume:
		resume_path = last_state_path

	if resume_path:
		if not os.path.isfile(resume_path):
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
		resume_state = load_training_state(resume_path, device)
		classifier.load_state_dict(resume_state["classifier_state"])
		optimizer.load_state_dict(resume_state["optimizer_state"])
		for state in optimizer.state.values():
			for key, value in state.items():
				if isinstance(value, torch.Tensor):
					state[key] = value.to(device)

		best_state = resume_state.get("best_state", None)
		best_valid_acc = float(resume_state.get("best_valid_acc", -1.0))
		best_epoch = int(resume_state.get("best_epoch", -1))
		no_improve_epochs = int(resume_state.get("no_improve_epochs", 0))
		last_epoch = int(resume_state.get("epoch", 0))
		start_epoch = last_epoch + 1
		print(
			f"Resumed from {resume_path} | last_epoch={last_epoch} "
			f"best_valid_acc={best_valid_acc:.4f} best_epoch={best_epoch}"
		)

	if start_epoch > args.epochs:
		print(
			f"Resume epoch {start_epoch} is already beyond target epochs={args.epochs}. "
			"No additional training steps will run."
		)

	for epoch in range(start_epoch, args.epochs + 1):
		classifier.train()
		total_loss = 0.0
		total = 0
		correct = 0
		epoch_start = time.time()
		num_steps = len(train_loader)
		print(f"Epoch {epoch}/{args.epochs} started: {num_steps} steps")

		for step, batch in enumerate(train_loader, start=1):
			global_step = (epoch - 1) * max(num_steps, 1) + step
			batch = move_batch_to_device(batch, device)

			with torch.no_grad():
				hidden = qwen_forward_hidden(
					qwen_model=qwen_model,
					input_ids=batch["input_ids"],
					attention_mask=batch["attention_mask"],
					pixel_values=batch["pixel_values"],
					image_grid_thw=batch["image_grid_thw"],
				)
				feats = gather_last_token_features(hidden, batch["attention_mask"])
				feats = align_feature_dtype(feats, classifier)

			logits = classifier(feats)
			masked_logits = apply_dtype_mask(logits, batch["dtypes"], dtype_mask)
			loss = loss_fn(masked_logits, batch["labels"])

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			preds = torch.argmax(masked_logits, dim=-1)
			correct += (preds == batch["labels"]).sum().item()
			bs = batch["labels"].shape[0]
			total += bs
			total_loss += loss.item() * bs

			if args.log_steps > 0 and (step == 1 or step % args.log_steps == 0 or step == num_steps):
				running_acc = correct / max(total, 1)
				running_loss = total_loss / max(total, 1)
				writer.add_scalar("train/loss_step", running_loss, global_step)
				writer.add_scalar("train/acc_step", running_acc, global_step)
				elapsed = time.time() - epoch_start
				steps_done = max(step, 1)
				eta = (elapsed / steps_done) * (num_steps - steps_done)
				print(
					f"  step {step}/{num_steps} | "
					f"loss={running_loss:.4f} acc={running_acc:.4f} | "
					f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
				)

		train_loss = total_loss / max(total, 1)
		train_acc = correct / max(total, 1)
		writer.add_scalar("train/loss_epoch", train_loss, epoch)
		writer.add_scalar("train/acc_epoch", train_acc, epoch)

		valid_metrics = evaluate(
			qwen_model=qwen_model,
			classifier=classifier,
			dataloader=valid_loader,
			dtype_mask=dtype_mask,
			loss_fn=loss_fn,
			device=device,
		)
		writer.add_scalar("valid/loss_epoch", valid_metrics["loss"], epoch)
		writer.add_scalar("valid/acc_epoch", valid_metrics["acc"], epoch)

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['acc']:.4f}"
		)

		improved = valid_metrics["acc"] > (best_valid_acc + args.early_stop_min_delta)
		if improved:
			best_valid_acc = valid_metrics["acc"]
			best_epoch = epoch
			best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
			no_improve_epochs = 0
			if not args.save_best_only:
				torch.save(best_state, os.path.join(save_dir, "classifier_best.pt"))
		else:
			no_improve_epochs += 1

		save_training_state(
			checkpoint_path=last_state_path,
			epoch=epoch,
			classifier=classifier,
			optimizer=optimizer,
			best_state=best_state,
			best_valid_acc=best_valid_acc,
			best_epoch=best_epoch,
			no_improve_epochs=no_improve_epochs,
		)

		if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
			print(
				f"Early stopping triggered at epoch {epoch}: "
				f"no improvement for {no_improve_epochs} epochs "
				f"(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})."
			)
			break

	if best_state is None:
		best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
		best_epoch = args.epochs

	classifier.load_state_dict(best_state)
	test_metrics = evaluate(
		qwen_model=qwen_model,
		classifier=classifier,
		dataloader=test_loader,
		dtype_mask=dtype_mask,
		loss_fn=loss_fn,
		device=device,
	)
	writer.add_scalar("test/loss", test_metrics["loss"], 0)
	writer.add_scalar("test/acc", test_metrics["acc"], 0)
	print(f"Best epoch: {best_epoch}, best_valid_acc={best_valid_acc:.4f}")
	print(f"Test metrics | loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f}")

	final_classifier_path = os.path.join(save_dir, "classifier.pt")
	torch.save(best_state, final_classifier_path)

	dtype_mask_save = {
		dtype_name: torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
		for dtype_name, mask in dtype_mask.items()
	}
	meta = {
		"base_model_dir": model_dir,
		"attn_implementation": args.attn_implementation,
		"feature_dim": feature_dim,
		"hidden_size": args.hidden_size,
		"dropout": args.dropout,
		"feature_dropout": args.feature_dropout,
		"num_labels": num_labels,
		"label2id": label2id,
		"id2label": {str(k): v for k, v in id2label.items()},
		"dtype_allowed_label_ids": dtype_mask_save,
		"best_epoch": best_epoch,
		"best_valid_acc": best_valid_acc,
		"test_loss": test_metrics["loss"],
		"test_acc": test_metrics["acc"],
		"run_args": vars(args),
		"tensorboard": {
			"log_root": tb_log_root,
			"run_dir": tb_run_dir,
			"port": args.tb_port,
		},
	}
	meta_path = os.path.join(save_dir, "classifier_meta.json")
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"Saved classifier to: {final_classifier_path}")
	print(f"Saved metadata to: {meta_path}")
	writer.close()


class QwenWithClassifier:
	def __init__(self, model_dir: str, classifier_dir: str, device: str = "auto"):
		self.model_dir = os.path.abspath(model_dir)
		self.classifier_dir = os.path.abspath(classifier_dir)
		self.device = resolve_device(device)

		meta_path = os.path.join(self.classifier_dir, "classifier_meta.json")
		cls_path = os.path.join(self.classifier_dir, "classifier.pt")
		if not os.path.isfile(meta_path) or not os.path.isfile(cls_path):
			raise FileNotFoundError("classifier_meta.json or classifier.pt not found.")

		self.meta = safe_read_json(meta_path)
		attn_impl = self.meta.get("attn_implementation", "flash_attention_2")
		self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
		self.qwen = load_qwen_model(
			model_dir=self.model_dir,
			device=self.device,
			attn_implementation=attn_impl,
		)
		for p in self.qwen.parameters():
			p.requires_grad = False
		self.qwen.eval()

		self.classifier = ClassifierHead(
			in_dim=int(self.meta["feature_dim"]),
			hidden_dim=int(self.meta["hidden_size"]),
			num_labels=int(self.meta["num_labels"]),
			dropout=float(self.meta["dropout"]),
			feature_dropout=float(self.meta.get("feature_dropout", 0.1)),
		).to(self.device)
		state = torch.load(cls_path, map_location=self.device)
		self.classifier.load_state_dict(state)
		self.classifier.eval()

		self.id2label = {int(k): v for k, v in self.meta["id2label"].items()}
		self.dtype_mask = {}
		for dtype_name, ids in self.meta["dtype_allowed_label_ids"].items():
			mask = torch.zeros(int(self.meta["num_labels"]), dtype=torch.bool)
			mask[torch.tensor(ids, dtype=torch.long)] = True
			self.dtype_mask[dtype_name] = mask

	@torch.no_grad()
	def predict(self, image_path: str, question: str, dtype_name: str) -> Dict[str, object]:
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image_path},
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
		feat = align_feature_dtype(feat, self.classifier)
		logits = self.classifier(feat)
		masked = apply_dtype_mask(logits, [dtype_name], self.dtype_mask)

		probs = torch.softmax(masked, dim=-1)[0]
		pred_id = int(torch.argmax(probs).item())
		pred_label = self.id2label[pred_id]
		pred_prob = float(probs[pred_id].item())

		return {
			"pred_id": pred_id,
			"pred_label": pred_label,
			"confidence": pred_prob,
		}


def main() -> None:
	args = parse_args()
	train(args)


if __name__ == "__main__":
	main()
