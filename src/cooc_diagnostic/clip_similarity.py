"""CLIP image-text similarity between COCO val2017 images and category names.

Uses openai/clip-vit-large-patch14-336 -- the same vision tower LLaVA-1.5-7B
uses internally -- so this covariate lives in the same visual-semantic space
the diagnostic's hypothesis is actually about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _encode_images(model, processor, image_paths: dict[int, str], device: str, batch_size: int) -> tuple[list[int], torch.Tensor]:
    """Returns (image_ids sorted ascending, L2-normalized image_features[N, D])."""
    image_ids = sorted(image_paths.keys())
    image_feature_batches = []
    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        images = [Image.open(image_paths[i]).convert("RGB") for i in batch_ids]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        image_feature_batches.append(feats.cpu())
    return image_ids, torch.cat(image_feature_batches, dim=0)


def compute_image_text_similarity(
    image_paths: dict[int, str],
    category_names: list[str],
    model_id: str = "openai/clip-vit-large-patch14-336",
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[list[int], np.ndarray]:
    """Returns (image_ids sorted ascending, cosine_sim[len(image_ids), len(category_names)])."""
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    text_inputs = processor(
        text=[f"a photo of a {name}" for name in category_names],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu()

    image_ids, image_features = _encode_images(model, processor, image_paths, device, batch_size)
    sim_matrix = (image_features @ text_features.T).numpy()
    return image_ids, sim_matrix


def compute_image_embeddings(
    image_paths: dict[int, str],
    model_id: str = "openai/clip-vit-large-patch14-336",
    device: str = "cuda",
    batch_size: int = 64,
) -> tuple[list[int], np.ndarray]:
    """Returns (image_ids sorted ascending, L2-normalized embeddings[len(image_ids), D]) --
    the raw frozen CLIP visual feature used by Stage 5's linear probe (as opposed
    to compute_image_text_similarity's image-text cosine similarity scalar)."""
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    image_ids, image_features = _encode_images(model, processor, image_paths, device, batch_size)
    return image_ids, image_features.numpy()


def load_or_compute_image_embeddings(
    cache_path: str | Path,
    image_paths: dict[int, str],
    **kwargs,
) -> tuple[list[int], np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        return payload["image_ids"].tolist(), payload["embeddings"]

    image_ids, embeddings = compute_image_embeddings(image_paths, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, image_ids=np.array(image_ids), embeddings=embeddings)
    return image_ids, embeddings


def load_or_compute_similarity(
    cache_path: str | Path,
    image_paths: dict[int, str],
    category_names: list[str],
    **kwargs,
) -> tuple[list[int], np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_names = payload["category_names"].tolist()
        if cached_names == category_names:
            return payload["image_ids"].tolist(), payload["sim_matrix"]

    image_ids, sim_matrix = compute_image_text_similarity(image_paths, category_names, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        image_ids=np.array(image_ids),
        category_names=np.array(category_names),
        sim_matrix=sim_matrix,
    )
    return image_ids, sim_matrix
