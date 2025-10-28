import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel

def classify_with_hf_clip(
    image_path: str,
    candidate_labels: list,
    model_name: str = "openai/clip-vit-base-patch32",
    top_k: int = 3,
    output_image_path: str = None
):
    """
    使用 HuggingFace Transformers 中的 CLIPModel 对图片进行零样本分类，并在图片上标注 top-k 预测结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型和处理器
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 2. 打开图片
    image = Image.open(image_path).convert("RGB")

    # 3. 构造输入
    inputs = processor(
        text=[f"a photo of a {label}" for label in candidate_labels],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # 4. 前向计算
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_labels)
        probs = logits_per_image.softmax(dim=1)      # 归一化概率

    # 5. Top-k
    top_k = min(top_k, len(candidate_labels))
    top_probs, top_idxs = probs.topk(top_k, dim=1)
    top_probs = top_probs.squeeze().tolist()
    top_idxs = top_idxs.squeeze().tolist()
    if isinstance(top_probs, float):  # 如果只有一个label
        top_probs, top_idxs = [top_probs], [top_idxs]
    top_results = [(candidate_labels[i], p) for i, p in zip(top_idxs, top_probs)]

    # 6. 绘制预测结果
    if output_image_path is None:
        output_image_path = image_path.rsplit('.', 1)[0] + "_clip_top3.jpg"

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    margin = 10
    y = margin
    header = f"Top-{top_k} Predictions:"
    draw.rectangle([(0, 0), (annotated.width, 30 + len(top_results)*25)], fill=(0, 0, 0, 128))
    draw.text((margin, y), header, fill="white", font=font)
    y += 25

    for rank, (label, p) in enumerate(top_results, 1):
        draw.text((margin, y), f"{rank}. {label} ({p*100:.2f}%)", fill="white", font=font)
        y += 25

    # annotated.save(output_image_path)

    return {
        "topk": top_results,
        "output_image_path": output_image_path
    }


if __name__ == "__main__":
    labels = ["cat", "dog", "car", "person", "bicycle"]
    result = classify_with_hf_clip(
        image_path="/hdd0/ljn/new_sam2/my_fssam/vis_test_imgs/person_image/p5.jpg",
        candidate_labels=labels,
        model_name="openai/clip-vit-base-patch32",
        top_k=3
    )
    print("Top-3:", result["topk"])
    print("Saved to:", result["output_image_path"])
