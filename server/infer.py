# #!/usr/bin/env python3
# """
# infer.py - CLI to run inference with the AttentionUNet from your notebook.

# New features:
#  - fixes albumentations PadIfNeeded warning
#  - --alpha FLOAT: mask opacity (0.0..1.0). Default 0.4 (i.e., 40% mask, 60% image)
#  - --legend: draw legend on overlay
#  - --labels "bg,healthy,unhealthy": comma-separated class labels
# """
# import argparse, os, glob, json
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from albumentations import Compose, LongestMaxSize, PadIfNeeded, Normalize
# from albumentations.pytorch import ToTensorV2

# # ---------------------------
# # Constants / defaults
# # ---------------------------
# N_CLASSES = 3  # 0 bg, 1 healthy, 2 unhealthy
# DEFAULT_LABELS = ["background", "healthy", "unhealthy"]
# PALETTE = np.array([[0,0,0],[0,255,0],[255,0,0]], dtype=np.uint8)  # class -> color

# # ---------------------------
# # Utilities
# # ---------------------------
# def colorize_mask(mask):
#     """mask: HxW with class indices 0..N_CLASSES-1"""
#     return PALETTE[mask]

# def draw_legend(img, class_labels=DEFAULT_LABELS, palette=PALETTE, xpos=10, ypos=10, spacing=8):
#     """
#     Draws legend (colored boxes + labels) onto img (RGB uint8).
#     xpos,ypos = top-left start position.
#     """
#     h, w = img.shape[:2]
#     x = xpos
#     y = ypos
#     box_h = 18
#     box_w = 28
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     thickness = 1
#     for i, label in enumerate(class_labels):
#         if i >= len(palette): break
#         cv2.rectangle(img, (x, y), (x+box_w, y+box_h), tuple(int(c) for c in palette[i]), -1)
#         text_x = x + box_w + 8
#         text_y = y + box_h - 3
#         # put a thin black outline for readability
#         cv2.putText(img, label, (text_x+1, text_y+1), font, font_scale, (0,0,0), thickness=2, lineType=cv2.LINE_AA)
#         cv2.putText(img, label, (text_x, text_y), font, font_scale, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
#         y += box_h + spacing
#     return img

# def overlay_rgb(img_t, mask_pred, mask_alpha=0.4, legend=False, class_labels=None):
#     """
#     img_t: tensor CxHxW in [0..1] (float) OR numpy HxWxC in [0..255]
#     mask_pred: HxW (numpy or tensor) with class indices
#     mask_alpha: float in [0,1] -> alpha for mask (0=transparent mask, 1=full mask)
#     legend: if True draw legend
#     class_labels: list of label strings
#     returns uint8 HxWx3 (RGB)
#     """
#     if isinstance(img_t, torch.Tensor):
#         img = img_t.permute(1,2,0).cpu().numpy()
#         img = (img*255).clip(0,255).astype(np.uint8)
#     else:
#         img = img_t.copy()
#         if img.dtype != np.uint8:
#             img = (img*255).clip(0,255).astype(np.uint8)

#     if isinstance(mask_pred, torch.Tensor):
#         mp = mask_pred.cpu().numpy()
#     else:
#         mp = mask_pred
#     cm  = colorize_mask(mp)  # HxWx3 uint8

#     alpha = float(mask_alpha)
#     alpha = max(0.0, min(1.0, alpha))
#     img_float = img.astype(np.float32)
#     cm_float = cm.astype(np.float32)
#     out = (img_float * (1.0 - alpha) + cm_float * alpha).clip(0,255).astype(np.uint8)

#     if legend:
#         labels = class_labels if class_labels is not None else DEFAULT_LABELS
#         out = draw_legend(out, class_labels=labels, palette=PALETTE.tolist())

#     return out

# # ---------------------------
# # Model (from your notebook) - keep same as before
# # ---------------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=False),
#         )
#     def forward(self,x): return self.net(x)

# class AttentionBlock(nn.Module):
#     def __init__(self, in_channels, gating_channels, inter_channels):
#         super().__init__()
#         # match the checkpoint structure: convs without bias and BatchNorm
#         self.W_g = nn.Sequential(
#             nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(inter_channels)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(inter_channels)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, g):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class UpBlock(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
#         self.att = AttentionBlock(in_channels=mid_ch, gating_channels=mid_ch, inter_channels=max(mid_ch//2,1))
#         self.conv = ConvBlock(mid_ch + mid_ch, out_ch)
#     def forward(self, x, bridge):
#         up = self.up(x)
#         if up.shape[2:] != bridge.shape[2:]:
#             bridge = F.interpolate(bridge, size=up.shape[2:], mode="bilinear", align_corners=False)
#         att = self.att(bridge, up)
#         out = torch.cat([up, att], dim=1)
#         out = self.conv(out)
#         return out

# class AttentionUNet(nn.Module):
#     def __init__(self, in_ch=3, n_classes=3, base=32):
#         super().__init__()
#         self.c1 = ConvBlock(in_ch, base)
#         self.p1 = nn.MaxPool2d(2)
#         self.c2 = ConvBlock(base, base*2)
#         self.p2 = nn.MaxPool2d(2)
#         self.c3 = ConvBlock(base*2, base*4)
#         self.p3 = nn.MaxPool2d(2)
#         self.c4 = ConvBlock(base*4, base*8)
#         self.p4 = nn.MaxPool2d(2)
#         self.c5 = ConvBlock(base*8, base*16)

#         self.u6 = UpBlock(base*16, base*8, base*8)
#         self.u7 = UpBlock(base*8,  base*4, base*4)
#         self.u8 = UpBlock(base*4,  base*2, base*2)
#         self.u9 = UpBlock(base*2,  base,    base)
#         self.outc = nn.Conv2d(base, n_classes, 1)

#     def forward(self, x):
#         c1 = self.c1(x); p1 = self.p1(c1)
#         c2 = self.c2(p1); p2 = self.p2(c2)
#         c3 = self.c3(p2); p3 = self.p3(c3)
#         c4 = self.c4(p3); p4 = self.p4(c4)
#         c5 = self.c5(p4)

#         x  = self.u6(c5, c4)
#         x  = self.u7(x,  c3)
#         x  = self.u8(x,  c2)
#         x  = self.u9(x,  c1)
#         return self.outc(x)

# # ---------------------------
# # Inference helpers
# # ---------------------------
# def build_transform(img_size):
#     # fixed: remove deprecated 'value' and 'mask_value' parameters to avoid warnings
#     return Compose([
#         LongestMaxSize(img_size),
#         PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
#         Normalize(), ToTensorV2()
#     ])

# def load_checkpoint(model, ckpt_path, device):
#     """
#     Loads checkpoint and remaps key names if needed.
#     - supports dict with keys 'model' or 'state_dict' or raw state_dict
#     - strips 'module.' prefix
#     - remaps attention block keys: Wg -> W_g and Wx -> W_x (common naming diff)
#     - loads state_dict with strict=False and prints missing/unexpected keys for debugging
#     """
#     ck = torch.load(ckpt_path, map_location=device)

#     if isinstance(ck, dict) and "model" in ck:
#         sd = ck["model"]
#     elif isinstance(ck, dict) and "state_dict" in ck:
#         sd = ck["state_dict"]
#     else:
#         sd = ck

#     new_sd = {}
#     for k, v in sd.items():
#         new_k = k
#         if new_k.startswith("module."):
#             new_k = new_k.replace("module.", "", 1)
#         if ".att.Wg." in new_k:
#             new_k = new_k.replace(".att.Wg.", ".att.W_g.")
#         if ".att.Wx." in new_k:
#             new_k = new_k.replace(".att.Wx.", ".att.W_x.")
#         if ".att.WG." in new_k:
#             new_k = new_k.replace(".att.WG.", ".att.W_g.")
#         if ".att.WX." in new_k:
#             new_k = new_k.replace(".att.WX.", ".att.W_x.")
#         new_sd[new_k] = v

#     load_res = model.load_state_dict(new_sd, strict=False)
#     if hasattr(load_res, "missing_keys") or hasattr(load_res, "unexpected_keys"):
#         missing = getattr(load_res, "missing_keys", [])
#         unexpected = getattr(load_res, "unexpected_keys", [])
#         print("Loaded checkpoint with non-strict mode.")
#         if missing:
#             print(f"Missing keys ({len(missing)}): {missing[:20]}{'...' if len(missing)>20 else ''}")
#         else:
#             print("Missing keys: None")
#         if unexpected:
#             print(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}{'...' if len(unexpected)>20 else ''}")
#         else:
#             print("Unexpected keys: None")
#     else:
#         print("Checkpoint loaded (no compatibility report available).")

#     return model

# def infer_single(model, img_path, tf, device, alpha=0.4, legend=False, labels=None):
#     rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
#     aug = tf(image=rgb)
#     x = aug["image"].unsqueeze(0).to(device).float()
#     model.eval()
#     with torch.no_grad():
#         logits = model(x)  # BxC x H x W
#         pred = logits.softmax(1).argmax(1)[0].cpu()  # H x W
#     ov = overlay_rgb(x[0].cpu(), pred, mask_alpha=alpha, legend=legend, class_labels=labels)
#     return ov, pred

# # ---------------------------
# # CLI
# # ---------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Run inference with AttentionUNet (notebook model).")
#     p.add_argument("--model", required=True, help="path to .pth checkpoint")
#     p.add_argument("--input", required=True, help="image file or folder containing images")
#     p.add_argument("--out", required=True, help="output folder to save overlays")
#     p.add_argument("--img-size", type=int, default=512, help="square image size used in transforms")
#     p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
#     p.add_argument("--alpha", type=float, default=0.4, help="mask alpha (0.0..1.0), default 0.4")
#     p.add_argument("--legend", action="store_true", help="draw legend on overlay")
#     p.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS), help="comma-separated class labels (in order from 0..N-1)")
#     return p.parse_args()

# def main():
#     args = parse_args()
#     device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
#     os.makedirs(args.out, exist_ok=True)

#     model = AttentionUNet(in_ch=3, n_classes=N_CLASSES, base=32)
#     model = model.to(device)

#     print("Loading checkpoint:", args.model)
#     model = load_checkpoint(model, args.model, device)
#     print("Model loaded.")

#     tf = build_transform(args.img_size)

#     labels = [l.strip() for l in args.labels.split(",")] if args.labels else DEFAULT_LABELS

#     inp = Path(args.input)
#     if inp.is_dir():
#         files = sorted([str(p) for p in inp.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp", ".tif", ".tiff"]])
#     else:
#         files = [str(inp)]
#     if not files:
#         print("No images found in input.")
#         return

#     for p in files:
#         try:
#             ov, pred = infer_single(model, p, tf, device, alpha=args.alpha, legend=args.legend, labels=labels)
#             fname = Path(p).stem
#             out_overlay = Path(args.out) / f"{fname}_overlay.png"
#             out_mask = Path(args.out) / f"{fname}_mask.png"
#             cv2.imwrite(str(out_overlay), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(str(out_mask), cv2.cvtColor(colorize_mask(pred.cpu().numpy()), cv2.COLOR_RGB2BGR))
#             print("Saved:", out_overlay, out_mask)
#         except Exception as e:
#             print("Failed to process", p, ":", e)

# if __name__ == "__main__":
#     main()


















# #!/usr/bin/env python3
# """
# infer.py - CLI to run inference with the AttentionUNet from your notebook.

# Features / fixes:
#  - use a separate visual transform (no Normalize) so overlay matches the actual image
#  - unnormalize utility and CLI options to set mean/std if different from ImageNet
#  - safer cv2.imread handling
#  - clearer device selection and optional --fp16 (CUDA only)
#  - save both colorized overlay and raw class-index mask
#  - improved checkpoint loading + compatibility report
# """
# import argparse
# import os
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from albumentations import Compose, LongestMaxSize, PadIfNeeded, Normalize
# from albumentations.pytorch import ToTensorV2

# # ---------------------------
# # Constants / defaults
# # ---------------------------
# N_CLASSES = 3  # 0 bg, 1 healthy, 2 unhealthy
# DEFAULT_LABELS = ["background", "healthy", "unhealthy"]
# PALETTE = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)  # class -> color

# # ImageNet defaults (change via CLI if your training used different)
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)


# # ---------------------------
# # Utilities
# # ---------------------------
# def colorize_mask(mask):
#     """mask: HxW with class indices 0..N_CLASSES-1 -> HxWx3 uint8"""
#     # clip to palette length to avoid index errors
#     m = np.asarray(mask, dtype=np.int32)
#     m = np.clip(m, 0, len(PALETTE) - 1)
#     return PALETTE[m]


# def draw_legend(img, class_labels=DEFAULT_LABELS, palette=PALETTE, xpos=10, ypos=10, spacing=8):
#     """
#     Draws legend (colored boxes + labels) onto img (RGB uint8).
#     xpos,ypos = top-left start position.
#     """
#     h, w = img.shape[:2]
#     x = xpos
#     y = ypos
#     box_h = 18
#     box_w = 28
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     thickness = 1
#     for i, label in enumerate(class_labels):
#         if i >= len(palette):
#             break
#         color = tuple(int(c) for c in palette[i])
#         cv2.rectangle(img, (x, y), (x + box_w, y + box_h), color, -1)
#         text_x = x + box_w + 8
#         text_y = y + box_h - 3
#         # put a thin black outline for readability
#         cv2.putText(img, label, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
#         cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
#         y += box_h + spacing
#     return img


# def overlay_rgb(img_t, mask_pred, mask_alpha=0.4, legend=False, class_labels=None):
#     """
#     img_t: tensor CxHxW in [0..1] (float) OR numpy HxWxC in [0..255]
#     mask_pred: HxW (numpy or tensor) with class indices
#     mask_alpha: float in [0,1] -> alpha for mask (0=transparent mask, 1=full mask)
#     legend: if True draw legend
#     class_labels: list of label strings
#     returns uint8 HxWx3 (RGB)
#     """
#     if isinstance(img_t, torch.Tensor):
#         img = img_t.permute(1, 2, 0).cpu().numpy()
#         img = (img * 255.0).clip(0, 255).astype(np.uint8)
#     else:
#         img = img_t.copy()
#         if img.dtype != np.uint8:
#             img = (img * 255.0).clip(0, 255).astype(np.uint8)

#     if isinstance(mask_pred, torch.Tensor):
#         mp = mask_pred.cpu().numpy()
#     else:
#         mp = np.asarray(mask_pred)

#     cm = colorize_mask(mp)  # HxWx3 uint8

#     alpha = float(mask_alpha)
#     alpha = max(0.0, min(1.0, alpha))
#     img_float = img.astype(np.float32)
#     cm_float = cm.astype(np.float32)
#     out = (img_float * (1.0 - alpha) + cm_float * alpha).clip(0, 255).astype(np.uint8)

#     if legend:
#         labels = class_labels if class_labels is not None else DEFAULT_LABELS
#         out = draw_legend(out, class_labels=labels, palette=PALETTE)

#     return out


# def unnormalize_tensor(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
#     """
#     img_t: torch tensor CxHxW (normalized with mean/std)
#     returns HxWx3 uint8
#     """
#     arr = img_t.cpu().permute(1, 2, 0).numpy().astype("float32")
#     mean = np.array(mean).reshape(1, 1, 3)
#     std = np.array(std).reshape(1, 1, 3)
#     arr = arr * std + mean
#     arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
#     return arr


# # ---------------------------
# # Model (from your notebook)
# # ---------------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=False),
#         )

#     def forward(self, x):
#         return self.net(x)


# class AttentionBlock(nn.Module):
#     def __init__(self, in_channels, gating_channels, inter_channels):
#         super().__init__()
#         # match the checkpoint structure: convs without bias and BatchNorm
#         self.W_g = nn.Sequential(
#             nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(inter_channels),
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(inter_channels),
#         )
#         self.psi = nn.Sequential(
#             nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid(),
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, g):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi


# class UpBlock(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
#         self.att = AttentionBlock(in_channels=mid_ch, gating_channels=mid_ch, inter_channels=max(mid_ch // 2, 1))
#         self.conv = ConvBlock(mid_ch + mid_ch, out_ch)

#     def forward(self, x, bridge):
#         up = self.up(x)
#         if up.shape[2:] != bridge.shape[2:]:
#             bridge = F.interpolate(bridge, size=up.shape[2:], mode="bilinear", align_corners=False)
#         att = self.att(bridge, up)
#         out = torch.cat([up, att], dim=1)
#         out = self.conv(out)
#         return out


# class AttentionUNet(nn.Module):
#     def __init__(self, in_ch=3, n_classes=3, base=32):
#         super().__init__()
#         self.c1 = ConvBlock(in_ch, base)
#         self.p1 = nn.MaxPool2d(2)
#         self.c2 = ConvBlock(base, base * 2)
#         self.p2 = nn.MaxPool2d(2)
#         self.c3 = ConvBlock(base * 2, base * 4)
#         self.p3 = nn.MaxPool2d(2)
#         self.c4 = ConvBlock(base * 4, base * 8)
#         self.p4 = nn.MaxPool2d(2)
#         self.c5 = ConvBlock(base * 8, base * 16)

#         self.u6 = UpBlock(base * 16, base * 8, base * 8)
#         self.u7 = UpBlock(base * 8, base * 4, base * 4)
#         self.u8 = UpBlock(base * 4, base * 2, base * 2)
#         self.u9 = UpBlock(base * 2, base, base)
#         self.outc = nn.Conv2d(base, n_classes, 1)

#     def forward(self, x):
#         c1 = self.c1(x)
#         p1 = self.p1(c1)
#         c2 = self.c2(p1)
#         p2 = self.p2(c2)
#         c3 = self.c3(p2)
#         p3 = self.p3(c3)
#         c4 = self.c4(p3)
#         p4 = self.p4(c4)
#         c5 = self.c5(p4)

#         x = self.u6(c5, c4)
#         x = self.u7(x, c3)
#         x = self.u8(x, c2)
#         x = self.u9(x, c1)
#         return self.outc(x)


# # ---------------------------
# # Inference helpers
# # ---------------------------
# def build_transforms(img_size, mean, std):
#     """
#     Return (tf_vis, tf_model)
#     - tf_vis: resizes/pads the image but does NOT normalize; returns HxWxC uint8
#     - tf_model: same resize/pad but then Normalize + ToTensorV2 for model input
#     """
#     tf_vis = Compose([LongestMaxSize(img_size), PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)])
#     tf_model = Compose([LongestMaxSize(img_size), PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
#                         Normalize(mean=mean, std=std), ToTensorV2()])
#     return tf_vis, tf_model


# def load_checkpoint(model, ckpt_path, device):
#     """
#     Loads checkpoint and remaps key names if needed.
#     - supports dict with keys 'model' or 'state_dict' or raw state_dict
#     - strips 'module.' prefix
#     - remaps common attention naming differences
#     - loads state_dict with strict=False and prints missing/unexpected keys for debugging
#     """
#     ck = torch.load(ckpt_path, map_location=device)

#     if isinstance(ck, dict) and "model" in ck:
#         sd = ck["model"]
#     elif isinstance(ck, dict) and "state_dict" in ck:
#         sd = ck["state_dict"]
#     else:
#         sd = ck

#     new_sd = {}
#     for k, v in sd.items():
#         new_k = k
#         if new_k.startswith("module."):
#             new_k = new_k.replace("module.", "", 1)
#         # remap a few naming variants often seen in attention implementations
#         new_k = new_k.replace(".att.Wg.", ".att.W_g.")
#         new_k = new_k.replace(".att.Wx.", ".att.W_x.")
#         new_k = new_k.replace(".att.WG.", ".att.W_g.")
#         new_k = new_k.replace(".att.WX.", ".att.W_x.")
#         new_sd[new_k] = v

#     load_res = model.load_state_dict(new_sd, strict=False)
#     # load_res may be a NamedTuple / object with missing/unexpected_keys attributes or an OrderedDict
#     missing = getattr(load_res, "missing_keys", None)
#     unexpected = getattr(load_res, "unexpected_keys", None)
#     if missing is None and unexpected is None and isinstance(load_res, dict):
#         # older torch returns dict with these keys sometimes
#         missing = load_res.get("missing_keys", [])
#         unexpected = load_res.get("unexpected_keys", [])

#     print("Loaded checkpoint with non-strict mode.")
#     if missing:
#         print(f"Missing keys ({len(missing)}): {missing[:50]}{'...' if len(missing) > 50 else ''}")
#     else:
#         print("Missing keys: None")
#     if unexpected:
#         print(f"Unexpected keys ({len(unexpected)}): {unexpected[:50]}{'...' if len(unexpected) > 50 else ''}")
#     else:
#         print("Unexpected keys: None")

#     return model


# def infer_single(model, img_path, tf_vis, tf_model, device, mean, std, use_fp16=False, alpha=0.4, legend=False, labels=None):
#     """Run inference on a single image path and return (overlay_rgb_uint8, pred_mask_numpy)"""
#     # read image robustly
#     img_bgr = cv2.imread(str(img_path))
#     if img_bgr is None:
#         raise RuntimeError(f"cv2.imread failed for {img_path}")
#     rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#     # visual transform (no normalize) -> HxWxC uint8
#     aug_vis = tf_vis(image=rgb)
#     vis_img = aug_vis["image"]
#     if vis_img.dtype != np.uint8:
#         # albumentations may return float if original image float; ensure uint8
#         vis_img = (vis_img * 255.0).clip(0, 255).astype(np.uint8)

#     # model transform -> tensor CHW normalized
#     aug = tf_model(image=rgb)
#     x = aug["image"].unsqueeze(0).to(device)
#     if use_fp16:
#         # only do fp16 on CUDA; torch will error on cpu
#         x = x.half()
#         model = model.half()

#     model.eval()
#     with torch.no_grad():
#         logits = model(x)  # BxC x H x W
#         # softmax then argmax
#         pred = logits.softmax(1).argmax(1)[0].cpu().numpy().astype(np.uint8)  # H x W (numpy)

#     # produce overlay using the visual uint8 image (prefer this because it matches resizing/padding)
#     ov = overlay_rgb(vis_img, pred, mask_alpha=alpha, legend=legend, class_labels=labels)

#     return ov, pred


# # ---------------------------
# # CLI
# # ---------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Run inference with AttentionUNet (notebook model).")
#     p.add_argument("--model", required=True, help="path to .pth checkpoint")
#     p.add_argument("--input", required=True, help="image file or folder containing images")
#     p.add_argument("--out", required=True, help="output folder to save overlays")
#     p.add_argument("--img-size", type=int, default=512, help="square image size used in transforms")
#     p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (e.g. cuda or cuda:0)")
#     p.add_argument("--alpha", type=float, default=0.4, help="mask alpha (0.0..1.0), default 0.4")
#     p.add_argument("--legend", action="store_true", help="draw legend on overlay")
#     p.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS), help="comma-separated class labels (in order from 0..N-1)")
#     p.add_argument("--base", type=int, default=32, help="base width used in AttentionUNet (must match training)")
#     p.add_argument("--n-classes", type=int, default=N_CLASSES, help="number of classes (must match checkpoint)")
#     p.add_argument("--mean", type=str, default=",".join(map(str, IMAGENET_MEAN)), help="mean (comma separated) used in Normalize, e.g. 0.485,0.456,0.406")
#     p.add_argument("--std", type=str, default=",".join(map(str, IMAGENET_STD)), help="std (comma separated) used in Normalize, e.g. 0.229,0.224,0.225")
#     p.add_argument("--fp16", action="store_true", help="use fp16 (only on CUDA; enabled if checkpoint trained in fp16)")
#     return p.parse_args()


# def str_to_tuple_floats(s):
#     parts = [x.strip() for x in s.split(",") if x.strip() != ""]
#     return tuple(float(x) for x in parts)


# def main():
#     args = parse_args()

#     # parse mean/std
#     mean = str_to_tuple_floats(args.mean)
#     std = str_to_tuple_floats(args.std)
#     if len(mean) != 3 or len(std) != 3:
#         raise ValueError("mean and std must be 3 comma-separated floats each (R,G,B).")

#     # device selection: honor requested device but fallback safely
#     requested = args.device.lower()
#     if "cuda" in requested and torch.cuda.is_available():
#         device = torch.device(requested)
#     else:
#         device = torch.device("cpu")

#     os.makedirs(args.out, exist_ok=True)

#     model = AttentionUNet(in_ch=3, n_classes=args.n_classes, base=args.base)
#     model = model.to(device)

#     print("Loading checkpoint:", args.model)
#     model = load_checkpoint(model, args.model, device)
#     print("Model loaded.")

#     tf_vis, tf_model = build_transforms(args.img_size, mean=mean, std=std)

#     labels = [l.strip() for l in args.labels.split(",")] if args.labels else DEFAULT_LABELS

#     inp = Path(args.input)
#     if inp.is_dir():
#         files = sorted([str(p) for p in inp.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
#     else:
#         files = [str(inp)]
#     if not files:
#         print("No images found in input:", args.input)
#         return

#     use_fp16 = args.fp16 and ("cuda" in str(device).lower() and torch.cuda.is_available())
#     if args.fp16 and not use_fp16:
#         print("Warning: --fp16 requested but CUDA not available; running in fp32.")

#     # run inference across files
#     for p in files:
#         try:
#             ov, pred = infer_single(model, p, tf_vis, tf_model, device, mean=mean, std=std,
#                                     use_fp16=use_fp16, alpha=args.alpha, legend=args.legend, labels=labels)
#             fname = Path(p).stem
#             out_overlay = Path(args.out) / f"{fname}_overlay.png"
#             out_mask_color = Path(args.out) / f"{fname}_mask_color.png"
#             out_mask_raw = Path(args.out) / f"{fname}_mask_raw.png"

#             # save overlay and colorized mask
#             cv2.imwrite(str(out_overlay), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(str(out_mask_color), cv2.cvtColor(colorize_mask(pred), cv2.COLOR_RGB2BGR))
#             # save raw class indices as single-channel PNG (uint8)
#             cv2.imwrite(str(out_mask_raw), pred.astype("uint8"))

#             print("Saved:", out_overlay, out_mask_color, out_mask_raw)
#         except Exception as e:
#             print("Failed to process", p, ":", repr(e))


# if __name__ == "__main__":
#     main()















#!/usr/bin/env python3
"""
infer.py - CLI to run inference with the AttentionUNet from your notebook.

Features / fixes:
 - use a separate visual transform (no Normalize) so overlay matches the actual image
 - unnormalize utility and CLI options to set mean/std if different from ImageNet
 - safer cv2.imread handling
 - clearer device selection and optional --fp16 (CUDA only)
 - save both colorized overlay and raw class-index mask
 - improved checkpoint loading + compatibility report

Default img-size changed to 1024 for higher recall on small objects.
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, LongestMaxSize, PadIfNeeded, Normalize
from albumentations.pytorch import ToTensorV2

# ---------------------------
# Constants / defaults
# ---------------------------
N_CLASSES = 3  # 0 bg, 1 healthy, 2 unhealthy
DEFAULT_LABELS = ["background", "healthy", "unhealthy"]
PALETTE = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)  # class -> color

# ImageNet defaults (change via CLI if your training used different)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------
# Utilities
# ---------------------------
def colorize_mask(mask):
    """mask: HxW with class indices 0..N_CLASSES-1 -> HxWx3 uint8"""
    m = np.asarray(mask, dtype=np.int32)
    m = np.clip(m, 0, len(PALETTE) - 1)
    return PALETTE[m]


def draw_legend(img, class_labels=DEFAULT_LABELS, palette=PALETTE, xpos=10, ypos=10, spacing=8):
    """
    Draws legend (colored boxes + labels) onto img (RGB uint8).
    xpos,ypos = top-left start position.
    """
    h, w = img.shape[:2]
    x = xpos
    y = ypos
    box_h = 18
    box_w = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    for i, label in enumerate(class_labels):
        if i >= len(palette):
            break
        color = tuple(int(c) for c in palette[i])
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), color, -1)
        text_x = x + box_w + 8
        text_y = y + box_h - 3
        # put a thin black outline for readability
        cv2.putText(img, label, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        y += box_h + spacing
    return img


def overlay_rgb(img_t, mask_pred, mask_alpha=0.4, legend=False, class_labels=None):
    """
    img_t: tensor CxHxW in [0..1] (float) OR numpy HxWxC in [0..255]
    mask_pred: HxW (numpy or tensor) with class indices
    mask_alpha: float in [0,1] -> alpha for mask (0=transparent mask, 1=full mask)
    legend: if True draw legend
    class_labels: list of label strings
    returns uint8 HxWx3 (RGB)
    """
    if isinstance(img_t, torch.Tensor):
        img = img_t.permute(1, 2, 0).cpu().numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img_t.copy()
        if img.dtype != np.uint8:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

    if isinstance(mask_pred, torch.Tensor):
        mp = mask_pred.cpu().numpy()
    else:
        mp = np.asarray(mask_pred)

    cm = colorize_mask(mp)  # HxWx3 uint8

    alpha = float(mask_alpha)
    alpha = max(0.0, min(1.0, alpha))
    img_float = img.astype(np.float32)
    cm_float = cm.astype(np.float32)
    out = (img_float * (1.0 - alpha) + cm_float * alpha).clip(0, 255).astype(np.uint8)

    if legend:
        labels = class_labels if class_labels is not None else DEFAULT_LABELS
        out = draw_legend(out, class_labels=labels, palette=PALETTE)

    return out


def unnormalize_tensor(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    img_t: torch tensor CxHxW (normalized with mean/std)
    returns HxWx3 uint8
    """
    arr = img_t.cpu().permute(1, 2, 0).numpy().astype("float32")
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    arr = arr * std + mean
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


# ---------------------------
# Model (from your notebook)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        # match the checkpoint structure: convs without bias and BatchNorm
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
        self.att = AttentionBlock(in_channels=mid_ch, gating_channels=mid_ch, inter_channels=max(mid_ch // 2, 1))
        self.conv = ConvBlock(mid_ch + mid_ch, out_ch)

    def forward(self, x, bridge):
        up = self.up(x)
        if up.shape[2:] != bridge.shape[2:]:
            bridge = F.interpolate(bridge, size=up.shape[2:], mode="bilinear", align_corners=False)
        att = self.att(bridge, up)
        out = torch.cat([up, att], dim=1)
        out = self.conv(out)
        return out


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=3, base=32):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(base * 8, base * 16)

        self.u6 = UpBlock(base * 16, base * 8, base * 8)
        self.u7 = UpBlock(base * 8, base * 4, base * 4)
        self.u8 = UpBlock(base * 4, base * 2, base * 2)
        self.u9 = UpBlock(base * 2, base, base)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        p4 = self.p4(c4)
        c5 = self.c5(p4)

        x = self.u6(c5, c4)
        x = self.u7(x, c3)
        x = self.u8(x, c2)
        x = self.u9(x, c1)
        return self.outc(x)


# ---------------------------
# Inference helpers
# ---------------------------
def build_transforms(img_size, mean, std):
    """
    Return (tf_vis, tf_model)
    - tf_vis: resizes/pads the image but does NOT normalize; returns HxWxC uint8
    - tf_model: same resize/pad but then Normalize + ToTensorV2 for model input
    """
    tf_vis = Compose([LongestMaxSize(img_size), PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)])
    tf_model = Compose([LongestMaxSize(img_size), PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
                        Normalize(mean=mean, std=std), ToTensorV2()])
    return tf_vis, tf_model


def load_checkpoint(model, ckpt_path, device):
    """
    Loads checkpoint and remaps key names if needed.
    - supports dict with keys 'model' or 'state_dict' or raw state_dict
    - strips 'module.' prefix
    - remaps common attention naming differences
    - loads state_dict with strict=False and prints missing/unexpected keys for debugging
    """
    ck = torch.load(ckpt_path, map_location=device)

    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    else:
        sd = ck

    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k.replace("module.", "", 1)
        # remap a few naming variants often seen in attention implementations
        new_k = new_k.replace(".att.Wg.", ".att.W_g.")
        new_k = new_k.replace(".att.Wx.", ".att.W_x.")
        new_k = new_k.replace(".att.WG.", ".att.W_g.")
        new_k = new_k.replace(".att.WX.", ".att.W_x.")
        new_sd[new_k] = v

    load_res = model.load_state_dict(new_sd, strict=False)
    missing = getattr(load_res, "missing_keys", None)
    unexpected = getattr(load_res, "unexpected_keys", None)
    if missing is None and unexpected is None and isinstance(load_res, dict):
        missing = load_res.get("missing_keys", [])
        unexpected = load_res.get("unexpected_keys", [])

    print("Loaded checkpoint with non-strict mode.")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:50]}{'...' if len(missing) > 50 else ''}")
    else:
        print("Missing keys: None")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:50]}{'...' if len(unexpected) > 50 else ''}")
    else:
        print("Unexpected keys: None")

    return model


def infer_single(model, img_path, tf_vis, tf_model, device, mean, std, use_fp16=False, alpha=0.4, legend=False, labels=None):
    """Run inference on a single image path and return (overlay_rgb_uint8, pred_mask_numpy)"""
    # read image robustly
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed for {img_path}")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # visual transform (no normalize) -> HxWxC uint8
    aug_vis = tf_vis(image=rgb)
    vis_img = aug_vis["image"]
    if vis_img.dtype != np.uint8:
        vis_img = (vis_img * 255.0).clip(0, 255).astype(np.uint8)

    # model transform -> tensor CHW normalized
    aug = tf_model(image=rgb)
    x = aug["image"].unsqueeze(0).to(device)
    if use_fp16:
        x = x.half()
        model = model.half()

    model.eval()
    with torch.no_grad():
        logits = model(x)  # BxC x H x W
        pred = logits.softmax(1).argmax(1)[0].cpu().numpy().astype(np.uint8)  # H x W (numpy)

    # produce overlay using the visual uint8 image (prefer this because it matches resizing/padding)
    ov = overlay_rgb(vis_img, pred, mask_alpha=alpha, legend=legend, class_labels=labels)

    return ov, pred


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run inference with AttentionUNet (notebook model).")
    p.add_argument("--model", required=True, help="path to .pth checkpoint")
    p.add_argument("--input", required=True, help="image file or folder containing images")
    p.add_argument("--out", required=True, help="output folder to save overlays")
    p.add_argument("--img-size", type=int, default=1024, help="square image size used in transforms (default 1024)")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (e.g. cuda or cuda:0)")
    p.add_argument("--alpha", type=float, default=0.4, help="mask alpha (0.0..1.0), default 0.4")
    p.add_argument("--legend", action="store_true", help="draw legend on overlay")
    p.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS), help="comma-separated class labels (in order from 0..N-1)")
    p.add_argument("--base", type=int, default=32, help="base width used in AttentionUNet (must match training)")
    p.add_argument("--n-classes", type=int, default=N_CLASSES, help="number of classes (must match checkpoint)")
    p.add_argument("--mean", type=str, default=",".join(map(str, IMAGENET_MEAN)), help="mean (comma separated) used in Normalize, e.g. 0.485,0.456,0.406")
    p.add_argument("--std", type=str, default=",".join(map(str, IMAGENET_STD)), help="std (comma separated) used in Normalize, e.g. 0.229,0.224,0.225")
    p.add_argument("--fp16", action="store_true", help="use fp16 (only on CUDA; enabled if checkpoint trained in fp16)")
    return p.parse_args()


def str_to_tuple_floats(s):
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    return tuple(float(x) for x in parts)


def main():
    args = parse_args()

    # parse mean/std
    mean = str_to_tuple_floats(args.mean)
    std = str_to_tuple_floats(args.std)
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must be 3 comma-separated floats each (R,G,B).")

    # device selection: honor requested device but fallback safely
    requested = args.device.lower()
    if "cuda" in requested and torch.cuda.is_available():
        device = torch.device(requested)
    else:
        device = torch.device("cpu")

    os.makedirs(args.out, exist_ok=True)

    model = AttentionUNet(in_ch=3, n_classes=args.n_classes, base=args.base)
    model = model.to(device)

    print("Loading checkpoint:", args.model)
    model = load_checkpoint(model, args.model, device)
    print("Model loaded.")

    tf_vis, tf_model = build_transforms(args.img_size, mean=mean, std=std)

    labels = [l.strip() for l in args.labels.split(",")] if args.labels else DEFAULT_LABELS

    inp = Path(args.input)
    if inp.is_dir():
        files = sorted([str(p) for p in inp.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
    else:
        files = [str(inp)]
    if not files:
        print("No images found in input:", args.input)
        return

    use_fp16 = args.fp16 and ("cuda" in str(device).lower() and torch.cuda.is_available())
    if args.fp16 and not use_fp16:
        print("Warning: --fp16 requested but CUDA not available; running in fp32.")

    # run inference across files
    for p in files:
        try:
            ov, pred = infer_single(model, p, tf_vis, tf_model, device, mean=mean, std=std,
                                    use_fp16=use_fp16, alpha=args.alpha, legend=args.legend, labels=labels)
            fname = Path(p).stem
            out_overlay = Path(args.out) / f"{fname}_overlay.png"
            out_mask_color = Path(args.out) / f"{fname}_mask_color.png"
            out_mask_raw = Path(args.out) / f"{fname}_mask_raw.png"

            # save overlay and colorized mask
            cv2.imwrite(str(out_overlay), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_mask_color), cv2.cvtColor(colorize_mask(pred), cv2.COLOR_RGB2BGR))
            # save raw class indices as single-channel PNG (uint8)
            cv2.imwrite(str(out_mask_raw), pred.astype("uint8"))

            print("Saved:", out_overlay, out_mask_color, out_mask_raw)
        except Exception as e:
            print("Failed to process", p, ":", repr(e))


if __name__ == "__main__":
    main()
