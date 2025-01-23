import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from PIL import Image

from dataset_tag_editor.tagger import WaifuDiffusion

tag = WaifuDiffusion("wd-v1-4-convnext-tagger-v2",0.5)
tag.start()

def get_tags(image,thresh:float = 0.5):
    output = tag.predict(image,thresh)
    sorted_items = sorted(output.items(), key=lambda item: item[1],reverse=True)
    sorted_keys = [item[0] for item in sorted_items]
    return sorted_keys

image = Image.open("D:/datasets/1.jpg")
sorted_keys = get_tags(image)
print(sorted_keys)