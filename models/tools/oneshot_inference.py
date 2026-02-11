import os
import cv2
import pickle
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.smpl._smpl import SMPL, Mesh
from models.bert.modeling_gabert import GAZEFROMBODY
from models.utils.miscellaneous import load_from_state_dict


# ============================
# transform（学習時と同一）
# ============================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================
# 矢印描画
# ============================
def draw_arrow(img, direction, head_bb, color, length=80):
    h, w, _ = img.shape

    cx = int((head_bb[0] + head_bb[2] * 0.5) * w)
    cy = int((head_bb[1] + head_bb[3] * 0.5) * h)

    direction = direction / (np.linalg.norm(direction) + 1e-8)

    dx = int(direction[0] * length)
    dy = int(-direction[1] * length)

    cv2.arrowedLine(
        img,
        (cx, cy),
        (cx + dx, cy + dy),
        color=color,
        thickness=2,
        tipLength=0.2
    )


# ============================
# メイン処理
# ============================
def main(args):
    assert args.n_frames % 2 == 1, "n_frames must be odd"
    center = args.n_frames // 2
    device = torch.device(args.device)

    # --- load annotation ---
    with open(args.annotation_path, "rb") as f:
        anno = pickle.load(f)

    img_indices = anno["index"]
    head_bb_all = np.asarray(anno["head_bb"], dtype=np.float32)
    gaze_gt_all = np.asarray(anno["gaze"], dtype=np.float32)

    # --- find target index ---
    target_name = os.path.basename(args.target_image_path)
    target_idx = int(os.path.splitext(target_name)[0])

    if target_idx not in img_indices:
        raise ValueError("Target image not found in annotation")

    i = img_indices.index(target_idx)

    if i - center < 0 or i + center >= len(img_indices):
        raise ValueError("Not enough frames around target image")

    # --- load sequence ---
    frame_ids = img_indices[i-center:i+center+1]
    img_paths = [
        os.path.join(args.image_dir, f"{idx:06}.jpg")
        for idx in frame_ids
    ]

    imgs = torch.stack(
        [transform(Image.open(p).convert("RGB")) for p in img_paths],
        dim=0
    ).unsqueeze(0).to(device)

    # --- load model ---
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()
    smpl.eval()

    metro = load_from_state_dict(None, smpl, mesh_sampler)
    metro.to(device)
    metro.eval()

    class ArgsTmp: pass
    args_tmp = ArgsTmp()
    args_tmp.n_frames = args.n_frames

    model = GAZEFROMBODY(args_tmp, metro)
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location=device)
    )
    model.to(device)
    model.eval()

    # --- inference ---
    with torch.no_grad():
        directions, _ = model(
            imgs,
            smpl,
            mesh_sampler,
            is_train=False
        )

    pred_gaze = directions[0, center].cpu().numpy()
    gt_gaze = gaze_gt_all[i]
    head_bb = head_bb_all[i]

    # --- visualization ---
    img_cv = cv2.imread(args.target_image_path)
    img_cv = cv2.resize(img_cv, (224, 224))

    # GT: green
    draw_arrow(img_cv, gt_gaze, head_bb, (0, 255, 0))
    # Pred: red
    draw_arrow(img_cv, pred_gaze, head_bb, (0, 0, 255))

    cv2.putText(
        img_cv,
        "GT (green) / Pred (red)",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imwrite(args.output_jpeg, img_cv)
    print(f"[OK] Saved: {args.output_jpeg}")


# ============================
# argparse
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay predicted and GT gaze on image"
    )
    parser.add_argument("--target_image_path", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--output_jpeg", required=True)

    parser.add_argument("--image_dir", required=True,
                        help="dataset image directory")
    parser.add_argument("--annotation_path", required=True)

    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # ============================
    # path existence check
    # ============================
    import os

    def check_file(path, name):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[ERROR] {name} not found: {path}"
            )

    def check_dir(path, name):
        if not os.path.isdir(path):
            raise NotADirectoryError(
                f"[ERROR] {name} is not a directory: {path}"
            )

    check_file(args.target_image_path, "target_image_path")
    check_file(args.model_checkpoint, "model_checkpoint")
    check_file(args.annotation_path, "annotation_path")

    check_dir(args.image_dir, "image_dir")

    main(args)
