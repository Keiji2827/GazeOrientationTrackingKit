import os
import glob
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

    #print("head_bb:", head_bb)
    #print(f"Draw arrow: ({cx}, {cy}) -> ({cx + dx}, {cy + dy})")



    cv2.arrowedLine(
        img,
        (cx, cy),
        (cx + dx, cy + dy),
        color=color,
        thickness=2,
        tipLength=0.2
    )

def load_model(args):
    # --- load model ---
    device = args.device
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()
    smpl.eval()

    metro = load_from_state_dict(args, smpl, mesh_sampler)
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
    return model, smpl, mesh_sampler


def expand_target_images(target_inputs):
    target_images = []
    #print(target_inputs)
    for item in target_inputs:
        #print(item)
        if os.path.isdir(item):
            target_images += sorted(
                glob.glob(os.path.join(item, "*.jpg"))
            )
        else:
            target_images += glob.glob(item)

    if len(target_images) == 0:
        raise ValueError("No valid target images found.")

    return sorted(target_images)

# ============================
# メイン処理
# ============================
def main(args):
    assert args.n_frames % 2 == 1, "n_frames must be odd"
    center = args.n_frames // 2
    device = torch.device(args.device)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("[INFO] Loading model...")
    model, smpl, mesh_sampler = load_model(args)


    # --------------------------------------------------------
    # Expand target images
    # --------------------------------------------------------
    target_images = expand_target_images(args.target_image_path)
    print(f"[INFO] {len(target_images)} images to process")


    # --- find target index ---
    for target_image_path in target_images:

        image_dir = os.path.dirname(target_image_path)
        pickle_path = os.path.join(image_dir, "annotation.pickle")

        if not os.path.exists(pickle_path):
            print(f"[WARN] Pickle not found: {pickle_path}")
            continue

        with open(pickle_path, "rb") as f:
            anno = pickle.load(f)

        img_indices = np.asarray(anno["index"])
        gaze_gt_all = np.asarray(anno["gazes"], dtype=np.float32)
        head_bb_all = np.vstack(anno['head_bb']).astype(np.float32)
        body_bb_all = np.vstack(anno['body_bb']).astype(np.float32)


        #print(f"\n[INFO] Processing: {target_image_path}")
        target_name = os.path.basename(target_image_path)
        target_idx = int(os.path.splitext(target_name)[0])

        matches = np.where(img_indices == target_idx)[0]
        if len(matches) == 0:
            print(f"[WARN] {target_idx} not in annotation. Skip.")
            continue

        i = int(matches[0])

        if i - center < 0 or i + center >= len(img_indices):
            print(f"[WARN] {target_idx} not in annotation. Skip.")
            continue

        # Load multi-frames
        frame_ids = img_indices[i - center : i + center + 1]
        img_paths = [
            os.path.join(image_dir, f"{idx:06}.jpg")
            for idx in frame_ids
        ]

        imgs = torch.stack(
            [transform(Image.open(p).convert("RGB")) for p in img_paths],
            dim=0
        ).unsqueeze(0).to(device)

        # --- inference ---
        with torch.no_grad():
            directions, S_diag = model(
                imgs,
                smpl,
                mesh_sampler,
                is_train=False
            )

        pred_gaze = directions[0].cpu().numpy()
        gt_gaze = gaze_gt_all[i]
        # ----------------------------------------------------
        # Head bbox (Dataset と同じ正規化)
        # ----------------------------------------------------
        head_bb = head_bb_all[i].astype(np.float32).copy()
        body_bb = body_bb_all[i].astype(np.float32)

        # body-relative
        head_bb[0] -= body_bb[0]
        head_bb[1] -= body_bb[1]

        head_bb[0] /= body_bb[2]
        head_bb[2] /= body_bb[2]

        head_bb[1] /= body_bb[3]
        head_bb[3] /= body_bb[3]

        # --- visualization ---
        img_cv = cv2.imread(target_image_path)
        img_cv = cv2.resize(img_cv, (224, 224))

        # GT: green
        draw_arrow(img_cv, gt_gaze, head_bb, (0, 255, 0))
        # Pred: red
        draw_arrow(img_cv, pred_gaze, head_bb, (0, 0, 255))

        #cv2.putText(
        #    img_cv,
        #    "GT (green) / Pred (red)",
        #    (5, 18),
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    0.5,
        #    (255, 255, 255),
        #    1
        #)

        base = os.path.basename(target_image_path)
        save_path = os.path.join(
            args.output_path,
            f"overlay_{base}"
        )

        cv2.imwrite(save_path, img_cv)
        print(f"[OK] Saved: {save_path}")


# ============================
# argparse
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay predicted and GT gaze on image"
    )
    # ============================
    # parameter for existing model (bert)
    # ============================
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', 
                        type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default='models/weights/metro/metro_3dpw_state_dict.bin', 
                        type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    #parser.add_argument("--model_metro_checkpoint", default='models/weights/metro/metro_for_gaze.pth', 
    #                    type=str, required=False,
    #                    help="Path to metro all checkpoint.")

    parser.add_argument("--target_image_path", required=True, nargs="+")
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--output_path", required=True)

    #parser.add_argument("--image_dir", required=True,
    #                    help="dataset image directory")
    #parser.add_argument("--annotation_path", required=True)

    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # ============================
    # path existence check
    # ============================

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

    #check_file(args.target_image_path, "target_image_path")
    check_file(args.model_checkpoint, "model_checkpoint")
    #check_file(args.annotation_path, "annotation_path")

    #check_dir(args.image_dir, "image_dir")

    main(args)
