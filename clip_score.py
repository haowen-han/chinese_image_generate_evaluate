from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import numpy as np
from typing import List
import torch

# Initialize model and processor as None
model = None
processor = None

def get_model_and_processor():
    """
    Instantiate the model and processor when first used and keep them in memory.
    """
    global model, processor
    if model is None or processor is None:
        model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14")
        processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14")
    return model, processor

def get_clip_score(images: List[Image.Image], texts: List[str]) -> torch.Tensor:
    """
    Calculate the CLIP score for the given images and texts.

    Parameters:
    images (List[Image.Image]): A list of images, each image corresponds to a text.
    texts (List[str]): A list of texts, each text corresponds to an image.

    Returns:
    torch.Tensor: The CLIP score for each pair of image and text.
    """
    assert len(images) == len(texts), "The number of images must be equal to the number of texts because each image should have a corresponding text for accurate comparison."
    model, processor = get_model_and_processor()
    inputs = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    # compute text features
    inputs = processor(text=texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    score = 100 * (image_features * text_features).sum(axis=-1)
    return torch.max(score, torch.zeros_like(score))
