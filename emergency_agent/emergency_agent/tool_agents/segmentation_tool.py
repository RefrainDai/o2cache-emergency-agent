import os
import sys
import base64
import io
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class RescueNetSegmentationTool:
    """Reusable wrapper for RescueNet PSPNet segmentation inference."""

    def __init__(
        self,
        seg_exp_root: str,
        model_path: str,
        device: Optional[torch.device] = None,
        resize_hw: Tuple[int, int] = (713, 713),
    ) -> None:
        self.seg_exp_root = os.path.abspath(seg_exp_root)
        self.model_path = os.path.abspath(model_path)
        self.resize_hw = resize_hw
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.color_encoding = self.default_color_encoding()
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._image_transform = transforms.Compose(
            [
                transforms.Resize(self.resize_hw, Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

        self.model = self._load_model()

    @staticmethod
    def default_color_encoding() -> "OrderedDict[str, Tuple[int, int, int]]":
        return OrderedDict(
            [
                ("unlabeled", (0, 0, 0)),
                ("water", (61, 230, 250)),
                ("building-no-damage", (180, 120, 120)),
                ("building-medium-damage", (235, 255, 7)),
                ("building-major-damage", (255, 184, 6)),
                ("building-total-destruction", (255, 0, 0)),
                ("vehicle", (255, 0, 245)),
                ("road-clear", (140, 140, 140)),
                ("road-blocked", (160, 150, 20)),
                ("tree", (4, 250, 7)),
                ("pool", (255, 235, 0)),
            ]
        )

    @staticmethod
    def _array_to_base64(image_array: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image_array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _compute_class_statistics(self, mask: np.ndarray) -> List[Dict[str, object]]:
        total_pixels = int(mask.size)
        labels = list(self.color_encoding.keys())
        stats: List[Dict[str, object]] = []
        unique_ids, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
            if 0 <= class_id < len(labels):
                stats.append(
                    {
                        "class_id": class_id,
                        "label": labels[class_id],
                        "pixel_count": count,
                        "ratio": round(count / total_pixels, 6),
                    }
                )
        stats.sort(key=lambda item: item["pixel_count"], reverse=True)
        return stats

    def _load_model(self) -> torch.nn.Module:
        if not os.path.isdir(self.seg_exp_root):
            raise FileNotFoundError(f"Segmentation experiment directory not found: {self.seg_exp_root}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        if self.seg_exp_root not in sys.path:
            sys.path.insert(0, self.seg_exp_root)

        from models.pspnet import PSPNet

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = PSPNet(
            layers=101,
            classes=11,
            zoom_factor=8,
            criterion=criterion,
            BatchNorm=nn.BatchNorm2d,
            pretrained=False,
        )

        if torch.cuda.is_available() and self.device.type == "cuda":
            model = torch.nn.DataParallel(model).to(self.device)
        else:
            model = model.to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            if isinstance(model, torch.nn.DataParallel):
                cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                model.module.load_state_dict(cleaned)
            else:
                cleaned = {("module." + k): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned)

        model.eval()
        return model

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self._normalize(self._image_transform(image)).unsqueeze(0)
        display = np.array(image.resize((self.resize_hw[1], self.resize_hw[0]), Image.NEAREST), dtype=np.uint8)
        return tensor, display

    def predict_mask(self, input_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(input_tensor.to(self.device))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        return pred

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        colors = np.array(list(self.color_encoding.values()), dtype=np.uint8)
        return colors[mask]

    @staticmethod
    def blend_overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        return ((1.0 - alpha) * image + alpha * color_mask).astype(np.uint8)

    def infer_one(self, image_path: str) -> Dict[str, object]:
        input_tensor, original = self.preprocess_image(image_path)
        pred_mask = self.predict_mask(input_tensor)
        pred_color = self.colorize_mask(pred_mask)
        overlay = self.blend_overlay(original, pred_color)
        return {
            "image_path": image_path,
            "original_image_base64": self._array_to_base64(original),
            "overlay_image_base64": self._array_to_base64(overlay),
            "class_statistics": self._compute_class_statistics(pred_mask),
        }

    def infer_batch(self, image_paths: Iterable[str]) -> List[Dict[str, object]]:
        outputs: List[Dict[str, object]] = []

        for image_path in image_paths:
            result = self.infer_one(image_path)
            outputs.append(result)

        return outputs


def build_default_tool(workspace_root: Optional[str] = None) -> RescueNetSegmentationTool:
    fallback_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if workspace_root:
        candidate_root = os.path.abspath(workspace_root)
        root = candidate_root if os.path.isdir(candidate_root) else fallback_root
    else:
        root = fallback_root

    seg_exp_root = os.path.join(
        root,
        "tools",
        "diasters_segmentation",
        "RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation-main",
        "Segmentation-Experiments",
    )
    model_path = os.path.join(root, "models", "RescueNet-segmentation", "train_epoch_125.pth")
    return RescueNetSegmentationTool(seg_exp_root=seg_exp_root, model_path=model_path)
