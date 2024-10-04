from clip_score import get_clip_score
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_batches(data, batch_size=4):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        images = [Image.open(os.path.join(os.path.dirname(__file__), item['img_path'])) for item in batch]
        prompts = [item['prompt'] for item in batch]
        neg_prompt = [item['neg_prompt'] for item in batch]
        yield images, prompts, neg_prompt

def get_all_data_json_paths(root_dir):
    data_json_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'data.json' in filenames:
            data_json_paths.append(os.path.join(dirpath, 'data.json'))
    return data_json_paths

if __name__ == "__main__":
    with torch.no_grad():
        data_json_paths = get_all_data_json_paths(os.path.join(os.path.dirname(__file__), 'data'))
        for path in data_json_paths:
            data = load_data(path)
            scores = {
                'cn': {'pos': torch.tensor([]), 'neg': torch.tensor([])},
                'en': {'pos': torch.tensor([]), 'neg': torch.tensor([])}
            }
            
            for images, prompts, neg_prompt in process_batches(data):
                for lang in ['cn', 'en']:
                    pos, neg = get_clip_score(images, prompts, neg_prompt, lang)
                    scores[lang]['pos'] = torch.cat([scores[lang]['pos'], pos.cpu()])
                    scores[lang]['neg'] = torch.cat([scores[lang]['neg'], neg.cpu()])

            width = 10
            
            height = 7
            
            plt.figure(figsize=(width, height))

            markers = {'cn': 's', 'en': 'o'}
            for lang in ['cn', 'en']:
                scores[lang]['score'] = (scores[lang]['pos'] - scores[lang]['neg'])
                plt.plot(scores[lang]["score"].cpu().numpy(), marker=markers[lang], label=f'{lang} clip score')
                
            plt.title(f'CLIP Scores for Image-Text Pairs\n' + 
                      '\n'.join([f'{lang.upper()} {t} Score: {scores[lang][t].mean():.2f}' 
                                 for lang in ['en', 'cn'] for t in ['pos', 'neg']]),
                      fontsize=8)
            plt.xlabel('Image-Text Pair Index')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(os.path.dirname(path), 'clip_scores.png'))