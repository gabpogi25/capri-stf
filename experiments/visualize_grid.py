import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import random

def get_match_img(img1_path, img2_path, kp1, kp2, matches, title, capri_rejected=None):
    img1 = cv2.cvtColor(cv2.imread(str(img1_path)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(str(img2_path)), cv2.COLOR_BGR2RGB)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    margin = 5
    out_img = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8)
    out_img[:h1, :w1, :] = img1
    out_img[:h2, w1+margin:, :] = img2
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(out_img)
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=5)
    
    if capri_rejected is not None:
        for idx1, idx2 in capri_rejected:
            pt1 = tuple(map(int, kp1[idx1]))
            pt2 = tuple(map(int, kp2[idx2]))
            pt2_shifted = (pt2[0] + w1 + margin, pt2[1])
            ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color='red', alpha=0.3, linewidth=0.5)

    for idx1, idx2 in matches:
        pt1 = tuple(map(int, kp1[idx1]))
        pt2 = tuple(map(int, kp2[idx2]))
        pt2_shifted = (pt2[0] + w1 + margin, pt2[1])
        c = 'lime' if capri_rejected is not None else 'lime'
        ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color=c, alpha=0.6, linewidth=1.0)
        ax.scatter([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color=c, s=1)
        
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img

def main():
    dataset_dir = Path("data/real_images/image-matching-challenge-2025/train/pt_sacrecoeur_trevi_tajmahal")
    out_dir = Path("outputs/benchmark_real")
    
    sift_matches = out_dir / "sift" / "matches.h5"
    sift_features = out_dir / "sift" / "features.h5"
    sg_matches = out_dir / "superglue" / "matches.h5"
    sg_features = out_dir / "superglue" / "features.h5"
    capri_matches = out_dir / "capri" / "matches_capri.h5"
    
    if not (sg_matches.exists() and sift_matches.exists() and capri_matches.exists()):
        print("Required HDF5 files missing.")
        return
        
    f_sift_m = h5py.File(sift_matches, 'r')
    f_sift_f = h5py.File(sift_features, 'r')
    f_sg_m = h5py.File(sg_matches, 'r')
    f_sg_f = h5py.File(sg_features, 'r')
    f_capri = h5py.File(capri_matches, 'r')
    
    img1_keys = list(f_sg_m.keys())
    
    rows = []
    
    for i in range(10, len(img1_keys)):
        if len(rows) >= 4:
            break
            
        img1_name = img1_keys[i]
        if len(f_sg_m[img1_name].keys()) == 0: continue
        img2_name = list(f_sg_m[img1_name].keys())[0]
        
        # Check if pair exists in sift
        if img1_name not in f_sift_m or img2_name not in f_sift_m[img1_name]: continue
        
        sg_m0 = f_sg_m[img1_name][img2_name]['matches0'][:]
        valid_sg = sg_m0 > -1
        sg_idx = np.stack([np.where(valid_sg)[0], sg_m0[valid_sg]], -1)
        if len(sg_idx) < 100: continue
            
        sift_m0 = f_sift_m[img1_name][img2_name]['matches0'][:]
        valid_sift = sift_m0 > -1
        sift_idx = np.stack([np.where(valid_sift)[0], sift_m0[valid_sift]], -1)
        
        capri_m0 = f_capri[img1_name][img2_name]['matches0'][:]
        valid_capri = capri_m0 > -1
        capri_idx = np.stack([np.where(valid_capri)[0], capri_m0[valid_capri]], -1)
        
        kp_sg_1 = f_sg_f[img1_name]['keypoints'][:]
        kp_sg_2 = f_sg_f[img2_name]['keypoints'][:]
        
        kp_sift_1 = f_sift_f[img1_name]['keypoints'][:]
        kp_sift_2 = f_sift_f[img2_name]['keypoints'][:]
        
        capri_set = set(map(tuple, capri_idx))
        sg_set = set(map(tuple, sg_idx))
        rejected = list(sg_set - capri_set)
        
        img_sift = get_match_img(dataset_dir/img1_name, dataset_dir/img2_name, kp_sift_1, kp_sift_2, sift_idx, f"SIFT + NN ({len(sift_idx)})")
        img_sg = get_match_img(dataset_dir/img1_name, dataset_dir/img2_name, kp_sg_1, kp_sg_2, sg_idx, f"SuperPoint + SuperGlue ({len(sg_idx)})")
        img_capri = get_match_img(dataset_dir/img1_name, dataset_dir/img2_name, kp_sg_1, kp_sg_2, capri_idx, f"SuperPoint + CAPRI ({len(capri_idx)})", capri_rejected=rejected)
        
        rows.append((img_sift, img_sg, img_capri))
        
        # Save individually
        out_base = Path(__file__).resolve().parent.parent / 'outputs' / 'figures'
        out_base.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_base / f"row{len(rows)}_sift.png"), cv2.cvtColor(img_sift, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_base / f"row{len(rows)}_sg.png"), cv2.cvtColor(img_sg, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_base / f"row{len(rows)}_capri.png"), cv2.cvtColor(img_capri, cv2.COLOR_RGB2BGR))
        
        print(f"Processed row {len(rows)}")
        
    f_sift_m.close()
    f_sift_f.close()
    f_sg_m.close()
    f_sg_f.close()
    f_capri.close()
    
    row_imgs = []
    for (i1, i2, i3) in rows:
        h = max(i1.shape[0], i2.shape[0], i3.shape[0])
        w1, w2, w3 = i1.shape[1], i2.shape[1], i3.shape[1]
        
        row_img = np.zeros((h, w1+w2+w3+20, 3), dtype=np.uint8) + 255
        
        row_img[:i1.shape[0], :w1, :] = i1
        row_img[:i2.shape[0], w1+10:w1+10+w2, :] = i2
        row_img[:i3.shape[0], w1+w2+20:w1+w2+20+w3, :] = i3
        row_imgs.append(row_img)
        
    max_w = max(r.shape[1] for r in row_imgs)
    total_h = sum(r.shape[0] for r in row_imgs) + 10*(len(row_imgs)-1)
    
    final_img = np.zeros((total_h, max_w, 3), dtype=np.uint8) + 255
    
    y = 0
    for r in row_imgs:
        final_img[y:y+r.shape[0], :r.shape[1], :] = r
        y += r.shape[0] + 10
        
    out_path = Path(__file__).resolve().parent.parent / 'outputs' / 'figures' / 'capri_grid_presentation.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
