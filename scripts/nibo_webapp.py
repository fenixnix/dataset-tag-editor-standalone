import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

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

if __name__ == "__main__":
    root_path = "Z:/comic_database/comic_grid/21世紀小福星"
    output_fn = "Z:/comic_database/comic_grid/21世紀小福星/tags.txt"
    with open(output_fn,'w',encoding="utf-8") as fw:
        for p,_,files in os.walk(root_path):
            for f in files:
                file_path = Path(f)
                if file_path.suffix in [".png",".jpg",".jpeg",".webp"]:
                    fn = os.path.join(p,f)
                    print(fn)
                    fw.write(fn+"\n")
                    img = Image.open(fn)
                    tags = get_tags(img)
                    fw.write(",".join(tags)+"\n")