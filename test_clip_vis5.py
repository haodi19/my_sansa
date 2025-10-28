# clip_token_similarity_map.py
import os
import argparse
import cv2
import numpy as np
import torch
from transformers import CLIPModel
from transformers import CLIPImageProcessor
import torch.nn.functional as F

def parse_args():
    p = argparse.ArgumentParser("CLIP patch↔patch similarity heatmap (no Grad-CAM)")
    p.add_argument("--imageA", type=str, default='vis_test_imgs/person/bb.png', help="Query image path (to visualize)")
    p.add_argument("--imageB", type=str, default='vis_test_imgs/person/dd.png', help="Reference image path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                   help="e.g. openai/clip-vit-base-patch32 or openai/clip-vit-base-patch16")
    p.add_argument("--size", type=int, default=224, help="CLIP vision input size (224 for these models)")
    p.add_argument("--sim_mode", type=str, default="max", choices=["max","mean"],
                   help="Aggregate similarity over ref patches")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian blur sigma in pixels after upsample (0 = no blur). Try ~patch_size/2")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--basename", type=str, default="token_sim")
    p.add_argument("--cls2dense", action="store_true",
               help="Use CLS token from imageA to compute similarity to dense tokens of imageB")
    return p.parse_args()

def load_rgb(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img_f = (img.astype(np.float32)/255.0).clip(0,1)
    # CLIP 归一化到 [-1,1]
    x = (img_f - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return img, img_f, x

def load_rgb2(path, size, processor=None, device="cuda"):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

    # 可视化用的 [0,1] 浮点图
    img_f = (img_rgb.astype(np.float32) / 255.0).clip(0, 1)

    # 标准 CLIP 预处理（强烈建议）
    if processor is None:
        x = torch.from_numpy(img_f).permute(2,0,1).unsqueeze(0)  # 退化备用
    else:
        # processor 会做 mean/std 归一化
        x = processor(images=img_rgb, return_tensors="pt")["pixel_values"]  # [1,3,H,W]

    x = x.to(device)
    return img_rgb, img_f, x

def load_rgb3(path, size, processor=None, device="cuda"):
    """
    加载RGB图像 → pad到正方形 → resize到 (size, size)
    返回:
        img_rgb: 原始RGB图像 (H0, W0, 3)
        img_f:   0~1浮点图 (H0, W0, 3)
        x:       模型输入 [1, 3, size, size]
        pad_info: dict，记录padding信息，用于可视化还原
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]

    # ---- 1) pad to square ----
    if H0 > W0:
        pad_left = (H0 - W0) // 2
        pad_right = H0 - W0 - pad_left
        pad_top = pad_bottom = 0
    else:
        pad_top = (W0 - H0) // 2
        pad_bottom = W0 - H0 - pad_top
        pad_left = pad_right = 0

    img_padded = cv2.copyMakeBorder(
        img_rgb,
        top=pad_top, bottom=pad_bottom,
        left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # black padding
    )

    # ---- 2) resize to target ----
    img_resized = cv2.resize(img_padded, (size, size), interpolation=cv2.INTER_AREA)

    # ---- 3) float & processor normalize ----
    img_f = (img_rgb.astype(np.float32) / 255.0).clip(0, 1)
    if processor is None:
        x = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
    else:
        x = processor(images=img_resized, return_tensors="pt")["pixel_values"]

    x = x.to(device)

    pad_info = {
        "orig_h": H0, "orig_w": W0,
        "pad_top": pad_top, "pad_bottom": pad_bottom,
        "pad_left": pad_left, "pad_right": pad_right,
        "padded_size": max(H0, W0),
        "target_size": size
    }

    return img_rgb, img_f, x, pad_info


@torch.no_grad()
def extract_patch_tokens(clip_model, x):
    """
    返回:
      patch_feats: [N, D]  (L2 normalized)
      grid: (Ht, Wt)
    """
    # out['pooler_output'].shape: torch.Size([1, 768])
    # out['last_hidden_state'].shape: torch.Size([1, 50, 768])
    out = clip_model.vision_model(pixel_values=x)
    tokens = out.last_hidden_state  # [1, 1+N, D]
    patch = tokens[:, 1:, :]        # drop CLS -> [1, N, D]
    patch = patch.squeeze(0)        # [N, D]
    patch = patch / (patch.norm(dim=-1, keepdim=True) + 1e-12)
    # 估计网格
    N = patch.shape[0]
    Ht = Wt = int(round(N**0.5))
    assert Ht*Wt == N, f"N={N} not a square grid; model might not be ViT/32/16 standard."
    return patch, (Ht, Wt)


@torch.no_grad()
def extract_patch_tokens2(clip_model, x):
    """
    提取 DenseCLIP 论文中使用的 dense 特征:
        F_dense = X^{L-1} W_v + b_v
    Args:
        clip_model: transformers.CLIPModel
        x: [B, 3, H, W]
    Returns:
        patch_feats: [N, D]  L2 normalized
        grid: (Ht, Wt)
    """
    hidden_container = {}

    def save_hidden_hook(module, inputs, outputs):
        # inputs[0] 是 TransformerBlock 的输入，即 X^{L-1}
        hidden_container['hidden'] = inputs[0].detach()

    # Hook 最后一层 TransformerBlock
    last_block = clip_model.vision_model.encoder.layers[-1]
    hook_handle = last_block.register_forward_hook(save_hidden_hook)

    # Forward 一次，触发hook
    _ = clip_model.vision_model(pixel_values=x)
    hook_handle.remove()

    # 拿到 X^{L-1}
    hidden_states = hidden_container['hidden']  # [B, N+1, D]

    # 拿 v_proj 权重
    self_attn = last_block.self_attn
    W_v = self_attn.v_proj.weight     # [D, D]
    b_v = self_attn.v_proj.bias       # [D]

    # 计算 V
    V = torch.matmul(hidden_states, W_v.T) + b_v  # [B, N+1, D]

    # 去掉 CLS
    patch = V[:, 1:, :]  # [B, N, D]
    patch = patch.squeeze(0)  # 假设 B=1
    patch = F.normalize(patch, dim=-1)  # L2 normalize

    # 计算网格大小
    N = patch.shape[0]
    Ht = Wt = int(round(N ** 0.5))
    assert Ht * Wt == N, f"N={N} 不是标准 ViT 网格"

    return patch, (Ht, Wt)

@torch.no_grad()
def extract_patch_tokens3(clip_model, x, apply_final_norm_on_v=False):
    """
    提取 DenseCLIP 中的 dense 特征，严格对齐你贴的 ViT 实现：
        y = layer_norm1(X^{L-1})
        V = y @ W_v^T + b_v
      （可选）若你的实现里在最后还对 v 应用了 norm1，这里用 apply_final_norm_on_v=True 对齐
    Args:
        clip_model: transformers.CLIPModel
        x: [B, 3, H, W]
        apply_final_norm_on_v: bool, 是否再对 V 过一遍同一个 layer_norm1（对应你代码中 final_norm 对 v 的处理）
    Returns:
        patch_feats: [N, D] (L2 normalized)
        grid: (Ht, Wt)
    """
    hidden_in = {}

    def save_block_input(module, inputs, outputs):
        # 这一层 block 的输入就是 X^{L-1}，形状 [B, N+1, D]
        hidden_in['x_l_minus_1'] = inputs[0].detach()

    # 最后一层 TransformerBlock
    last_block = clip_model.vision_model.encoder.layers[-1]
    h = last_block.register_forward_hook(save_block_input)

    # 前向一遍，触发 hook
    _ = clip_model.vision_model(pixel_values=x)
    h.remove()

    x_lm1 = hidden_in['x_l_minus_1']              # [B, N+1, D]

    # 取这一层的 pre-norm 与 v_proj
    # 注意：有的实现叫 layer_norm1，有的可能叫 ln_1；这里按你列的参数名 layer_norm1
    ln1 = last_block.layer_norm1
    v_proj = last_block.self_attn.v_proj          # [D, D] 仿射层

    # 先做 pre-norm（严格对齐你贴的实现）
    y = ln1(x_lm1)                                # [B, N+1, D]

    # 再投影为 V
    V = v_proj(y)                                 # 等价于 y @ W_v^T + b_v, shape [B, N+1, D]

    # （可选）若你的实现里在最后对 v 又做了一次 norm1，这里可打开
    if apply_final_norm_on_v:
        V = ln1(V)

    # 去 CLS，仅保留 patch
    patch = V[:, 1:, :]                           # [B, N, D]
    patch = patch.squeeze(0)                      # 假设 B=1 → [N, D]
    patch = F.normalize(patch, dim=-1)            # L2 norm，方便余弦相似度

    # 还原网格
    N = patch.shape[0]
    Ht = Wt = int(round(N ** 0.5))
    assert Ht * Wt == N, f"N={N} 不是标准 ViT 网格"
    return patch, (Ht, Wt)

@torch.no_grad()
def extract_key_tokens(clip_model, x):
    """
    严格对齐最后一层 pre-norm 的 K： y = ln1(X^{L-1});  K = y @ W_k^T + b_k
    返回: [N, D] (L2 normalized), (Ht, Wt)
    """
    hidden_in = {}
    def save_block_input(module, inputs, outputs):
        hidden_in['x_l_minus_1'] = inputs[0].detach()

    last_block = clip_model.vision_model.encoder.layers[-1]
    h = last_block.register_forward_hook(save_block_input)
    _ = clip_model.vision_model(pixel_values=x)
    h.remove()

    x_lm1 = hidden_in['x_l_minus_1']
    ln1 = last_block.layer_norm1
    k_proj = last_block.self_attn.k_proj

    y = ln1(x_lm1)     # pre-norm
    K = k_proj(y)      # [B, N+1, D]
    K = K[:, 1:, :].squeeze(0)
    K = F.normalize(K, dim=-1)

    N = K.shape[0]
    Ht = Wt = int(round(N ** 0.5))
    return K, (Ht, Wt)

@torch.no_grad()
def extract_cls_token(clip_model, x):
    """
    使用 CLIP 原生的 CLS token (pooler_output)
    Args:
        clip_model: transformers.CLIPModel
        x: [B, 3, H, W]
    Returns:
        cls_feat: [D]  (L2 normalized)
    """
    out = clip_model.vision_model(pixel_values=x)
    # pooler_output = LayerNorm(last_hidden_state[:, 0, :])
    cls_feat = out.pooler_output  # [B, D]
    cls_feat = F.normalize(cls_feat, dim=-1)
    cls_feat = cls_feat.squeeze(0)  # [D]
    return cls_feat



def key_smoothing(scores_1d, key_feats, Ht, Wt, alpha=20, ksize=3):
    """
    scores_1d: [N] 或 [N, C]；这里你用的是 [N]
    key_feats: [N, D] (L2 norm)
    返回: [N]
    """
    if scores_1d.dim() == 1:
        scores = scores_1d.unsqueeze(1)  # [N,1]
    else:
        scores = scores_1d               # [N,C]

    B = 1
    C = scores.shape[1]
    H, W = Ht, Wt
    scores_2d = scores.view(B, H, W, C)
    keys_2d = key_feats.view(B, H, W, -1)

    pad = ksize // 2
    out = torch.zeros_like(scores_2d)

    for i in range(H):
        for j in range(W):
            kp = keys_2d[0, i, j]  # [D]
            i0, i1 = max(0, i - pad), min(H, i + pad + 1)
            j0, j1 = max(0, j - pad), min(W, j + pad + 1)

            neigh_k = keys_2d[0, i0:i1, j0:j1].reshape(-1, keys_2d.shape[-1])
            neigh_s = scores_2d[0, i0:i1, j0:j1].reshape(-1, C)

            w = torch.matmul(neigh_k, kp)             # cos 因为已 L2
            w = torch.exp(alpha * w)
            w = w / (w.sum() + 1e-8)
            out[0, i, j] = (w.unsqueeze(1) * neigh_s).sum(0)

    out = out.view(-1, C)
    if scores_1d.dim() == 1:
        out = out.squeeze(1)
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    clip_model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(args.model)
    # for name, param in clip_model.vision_model.named_parameters():
        # print(f"Name: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")
    # exit(0)
    # patch_size 从名字大致判断（也可通过 config.patch_size 读取）
    patch_size = 32 if "patch32" in args.model else 16
    
    # # 2) 读图并预处理
    # imgA_u8, imgA_f, xA = load_rgb(args.imageA, args.size)
    # imgB_u8, imgB_f, xB = load_rgb(args.imageB, args.size)
    imgA_u8, imgA_f, xA = load_rgb2(args.imageA, args.size, processor, 'cuda')
    imgB_u8, imgB_f, xB = load_rgb2(args.imageB, args.size, processor, 'cuda')
    # imgA_u8, imgA_f, xA, pad_info_A = load_rgb3(args.imageA, args.size, processor, 'cuda')
    # imgB_u8, imgB_f, xB, pad_info_B = load_rgb3(args.imageB, args.size, processor, 'cuda')

    xA = xA.to(device)
    xB = xB.to(device)

    if args.cls2dense:
        print("🔸 Using CLS-to-dense similarity")
        clsB = extract_cls_token(clip_model, xB)      # [D]
        patchA, (HtA, WtA) = extract_patch_tokens(clip_model, xA)   # [Nb, D]

        # 计算 CLS(A) 与 Dense(B) 的相似度
        S_cls = torch.matmul(patchA, clsB.unsqueeze(1)).squeeze(1)  # [Nb]
        sim_map = S_cls.view(HtA, WtA).detach().cpu().numpy().astype(np.float32)

        vmin, vmax = np.percentile(sim_map, 1), np.percentile(sim_map, 99)
        sim_map = (sim_map - vmin) / max(1e-6, (vmax - vmin))
        sim_map = sim_map.clip(0, 1)

        sim_up = cv2.resize(sim_map, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        if args.smooth > 0:
            sim_up = cv2.GaussianBlur(sim_up, ksize=(0, 0),
                                    sigmaX=args.smooth, sigmaY=args.smooth)

        heatmap_color = cv2.applyColorMap((sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = (imgA_f * 255).astype(np.uint8)
        overlay = cv2.addWeighted(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), 0.5, heatmap_color, 0.5, 0)

        base = f"{args.basename}_cls2dense_{'p'+str(patch_size)}"
        out_heat = os.path.join(args.out_dir, f"{base}_heatmap.png")
        out_overlay = os.path.join(args.out_dir, f"{base}_overlay.png")
        cv2.imwrite(out_heat, heatmap_color)
        cv2.imwrite(out_overlay, overlay)
        print("✅ Saved (CLS→Dense):")
        print(" -", out_heat)
        print(" -", out_overlay)
        return  # 注意直接返回，跳过后续 dense↔dense 逻辑


    # 3) 提取 A/B 的 patch token（归一化）patchA: torch.Size([49, 768])
    patchA, (HtA, WtA) = extract_patch_tokens3(clip_model, xA)   # [Na, D], Na = HtA*WtA
    patchB, (HtB, WtB) = extract_patch_tokens3(clip_model, xB)   # [Nb, D]
    # import pdb
    # pdb.set_trace()
    # 4) 计算相似度矩阵 S = patchA @ patchB^T -> [Na, Nb]
    S = torch.matmul(patchA, patchB.t())  # 余弦相似（因为已 L2 norm）

    # 5) 沿 ref 维度做聚合：每个 A patch 一个分数
    if args.sim_mode == "max":
        # print(111)
        sA, _ = torch.max(S, dim=1)    # [Na]
    else:
        # print(333)
        sA = torch.mean(S, dim=1)      # [Na]

    # （可选）Key Smoothing
    # K只需要对A图提一次；若想对称 KS，可对 B 也算，在相似度上再做一次。
    # 例如启用一个命令行参数 --ks_alpha > 0 来打开
    ks_alpha = 20  # 改成 args.ks_alpha if you add argparse
    if ks_alpha > 0:
        keyA, (Ht_k, Wt_k) = extract_key_tokens(clip_model, xA)
        assert Ht_k == HtA and Wt_k == WtA
        sA = key_smoothing(sA, keyA, HtA, WtA, alpha=ks_alpha, ksize=3)

    # 6) 还原到 (HtA, WtA) 网格并归一化到 [0,1]
    sim_map = sA.view(HtA, WtA).detach().cpu().numpy().astype(np.float32)
    # 线性归一化（保持相对对比）
    vmin, vmax = np.percentile(sim_map, 1), np.percentile(sim_map, 99)
    sim_map = (sim_map - vmin) / max(1e-6, (vmax - vmin))
    sim_map = sim_map.clip(0,1)

    # 7) 上采样到图像大小
    sim_up = cv2.resize(sim_map, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    
    if False:
        # 8) 反pad到原图比例
        pad = pad_info_A   # 对应 imageA 的 pad_info
        padded_size = pad["padded_size"]
        pad_top, pad_bottom = pad["pad_top"], pad["pad_bottom"]
        pad_left, pad_right = pad["pad_left"], pad["pad_right"]

        # 把 224×224 热力图还原到 pad 后的正方形
        heatmap_padded = cv2.resize(sim_up, (padded_size, padded_size), interpolation=cv2.INTER_LINEAR)

        # 裁掉 padding，回到原图大小
        heatmap_cropped = heatmap_padded[
            pad_top : padded_size - pad_bottom,
            pad_left : padded_size - pad_right
        ]

        # 最终 resize 回原始大小
        sim_up = cv2.resize(
            heatmap_cropped, 
            (pad["orig_w"], pad["orig_h"]), 
            interpolation=cv2.INTER_LINEAR
        )


    # 可选平滑（缓解“菱形/块状”视觉）
    if args.smooth > 0:
        sim_up = cv2.GaussianBlur(sim_up, ksize=(0,0), sigmaX=args.smooth, sigmaY=args.smooth)

    # heatmap_color = cv2.applyColorMap((sim_up*255).astype(np.uint8), cv2.COLORMAP_JET)
    # base = f"{args.basename}_{'p'+str(patch_size)}_{args.sim_mode}"
    # out_heat = os.path.join(args.out_dir, f"{base}_heatmap.png")
    # cv2.imwrite(out_heat, heatmap_color)


    # 8) 叠加 & 保存
    heatmap_color = cv2.applyColorMap((sim_up*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (imgA_f*255).astype(np.uint8)
    overlay = cv2.addWeighted(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), 0.5, heatmap_color, 0.5, 0)

    base = f"{args.basename}_{'p'+str(patch_size)}_{args.sim_mode}"
    out_heat = os.path.join(args.out_dir, f"{base}_heatmap.png")
    out_overlay = os.path.join(args.out_dir, f"{base}_overlay.png")
    cv2.imwrite(out_heat, heatmap_color)
    cv2.imwrite(out_overlay, overlay)
    print("✅ Saved:")
    print(" -", out_heat)
    print(" -", out_overlay)

if __name__ == "__main__":
    main()
