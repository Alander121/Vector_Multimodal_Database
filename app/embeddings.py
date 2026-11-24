from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from .config import settings


def resolve_device(value: str) -> torch.device:
    value = value.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


class EmbeddingModels:
    def __init__(self) -> None:
        threads = max(1, settings.torch_num_threads)
        torch.set_num_threads(min(threads, os.cpu_count() or threads))
        self.device = resolve_device(settings.device)
        token = settings.hf_token

        print(f"[embeddings] device={self.device}")
        print(f"[embeddings] loading SigLIP2: {settings.siglip_model}")
        self.siglip_processor = AutoProcessor.from_pretrained(settings.siglip_model, token=token)
        self.siglip_model = AutoModel.from_pretrained(settings.siglip_model, token=token).to(self.device)
        self.siglip_model.eval()

        print(f"[embeddings] loading DINOv2: {settings.dino_model}")
        self.dino_processor = AutoImageProcessor.from_pretrained(settings.dino_model, token=token)
        self.dino_model = AutoModel.from_pretrained(settings.dino_model, token=token).to(self.device)
        self.dino_model.eval()

    @staticmethod
    def _tensor_from_output(output):
        if isinstance(output, torch.Tensor):
            return output
        pooler = getattr(output, "pooler_output", None)
        if pooler is not None:
            return pooler
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            return hidden[:, 0, :]
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    @torch.inference_mode()
    def encode_images(self, images: Iterable[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        images = list(images)
        if not images:
            return np.empty((0, 768), np.float32), np.empty((0, 768), np.float32)

        a = self.siglip_processor(images=images, return_tensors="pt")
        a = {k: v.to(self.device) for k, v in a.items() if hasattr(v, "to")}
        sig = self._normalize(self._tensor_from_output(self.siglip_model.get_image_features(**a)))

        b = self.dino_processor(images=images, return_tensors="pt")
        b = {k: v.to(self.device) for k, v in b.items() if hasattr(v, "to")}
        dino = self._normalize(self._tensor_from_output(self.dino_model(**b)))

        return sig.float().cpu().numpy().astype(np.float32), dino.float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        sig, dino = self.encode_images([image])
        return sig[0], dino[0]

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        a = self.siglip_processor(text=[text], padding="max_length", return_tensors="pt")
        a = {k: v.to(self.device) for k, v in a.items() if hasattr(v, "to")}
        out = self._normalize(self._tensor_from_output(self.siglip_model.get_text_features(**a)))
        return out.float().cpu().numpy()[0].astype(np.float32)
