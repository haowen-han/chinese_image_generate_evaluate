from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import numpy as np
from typing import List
import torch
from torchmetrics.functional.multimodal import clip_score
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor
from transformers import CLIPConfig

# Initialize model and processor as None
cn_clip_model = None
cn_clip_processor = None

# Initialize model and processor as None
en_clip_model = None
en_clip_processor = None

def get_chinese_model_and_processor():
    """
    Instantiate the model and processor when first used and keep them in memory.
    """
    global cn_clip_model, cn_clip_processor
    if cn_clip_model is None or cn_clip_processor is None:
        cn_clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14",device_map = 'cuda')
        cn_clip_processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14")
    return cn_clip_model, cn_clip_processor


def get_english_model_and_processor():
    """
    Instantiate the model and processor when first used and keep them in memory.
    """
    global en_clip_model, en_clip_processor
    if en_clip_model is None or en_clip_processor is None:
        # en_clip_model = _CLIPModel.from_pretrained("openai/clip-vit-base-patch16",device_map = 'cuda')
        # en_clip_processor =  _CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        maxtokens = 248
        model_id = "zer0int/LongCLIP-GmP-ViT-L-14"
        config = CLIPConfig.from_pretrained(model_id)
        config.text_config.max_position_embeddings = maxtokens

        en_clip_model = _CLIPModel.from_pretrained(model_id, torch_dtype=torch.bfloat16,config=config,device_map = 'cuda')
        en_clip_processor =  _CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=maxtokens, return_tensors="pt", truncation=True)

    return en_clip_model, en_clip_processor


def get_score(processor, model, images, texts):
    processed_input = processor(text=texts, images=images, return_tensors="pt", padding=True)
    img_features = model.get_image_features(processed_input["pixel_values"].to("cuda"))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    max_position_embeddings = model.config.text_config.max_position_embeddings
    print(max_position_embeddings)
    if processed_input["attention_mask"].shape[-1] > max_position_embeddings:
        print(
            f"Encountered caption longer than {max_position_embeddings=}. Will truncate captions to this length."
            "If longer captions are needed, initialize argument `model_name_or_path` with a model that supports"
            "longer sequences",
            UserWarning,
        )
        processed_input["attention_mask"] = processed_input["attention_mask"][..., :max_position_embeddings]
        processed_input["input_ids"] = processed_input["input_ids"][..., :max_position_embeddings]

    txt_features = model.get_text_features(
        processed_input["input_ids"].to("cuda"), processed_input["attention_mask"].to("cuda")
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity between feature vectors
    score = 100 * (img_features * txt_features).sum(axis=-1)
    return score

def get_clip_score(images: List[Image.Image], texts: List[str], neg_texts: List[str], type: str) -> torch.Tensor:
    """
    Calculate the CLIP score for the given images and texts.
    """
    assert type in ['cn', 'en'], "The type must be 'cn' or 'en'."
    assert len(images) == len(texts), "The number of images must be equal to the number of texts because each image should have a corresponding text for accurate comparison."
    if type == 'cn':
        model, processor = get_chinese_model_and_processor()
    else:
        model, processor = get_english_model_and_processor()
    if neg_texts is not None:
        assert len(images) == len(neg_texts), "The number of images must be equal to the number of neg_texts because each image should have a corresponding neg_text for accurate comparison."
        return get_score(model=model,processor=processor,images=images,texts=texts),get_score(model=model,processor=processor,images=images,texts=neg_texts)
    else:
        prompt_score = get_score(model=model,processor=processor,images=images,texts=texts)
        return prompt_score,torch.zeros_like(prompt_score)