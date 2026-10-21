import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import random
from pathlib import Path as _Path
FIGURES_DIR = _Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def draw_matches(img1_path, img2_path, kp1, kp2, matches, title, out_path, capri_rejected=None):
    img1 = cv2.cvtColor(cv2.imread(str(img1_path)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(str(img2_path)), cv2.COLOR_BGR2RGB)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a blank image to hold both images side-by-side
    margin = 10
    out_img = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8)
    out_img[:h1, :w1, :] = img1
    out_img[:h2, w1+margin:, :] = img2
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(out_img)
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=10)
    
    # If there are rejected matches, draw them in red underneath
    if capri_rejected is not None:
        for idx1, idx2 in capri_rejected:
            pt1 = tuple(map(int, kp1[idx1]))
            pt2 = tuple(map(int, kp2[idx2]))
            pt2_shifted = (pt2[0] + w1 + margin, pt2[1])
            ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color='red', alpha=0.3, linewidth=0.5)

    # Draw accepted matches in green
    for idx1, idx2 in matches:
        pt1 = tuple(map(int, kp1[idx1]))
        pt2 = tuple(map(int, kp2[idx2]))
        pt2_shifted = (pt2[0] + w1 + margin, pt2[1])
        ax.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color='lime', alpha=0.6, linewidth=1.0)
        ax.scatter([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color='lime', s=2)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    dataset_dir = Path("data/real_images/image-matching-challenge-2025/train/pt_sacrecoeur_trevi_tajmahal")
    out_dir = Path("outputs/benchmark_real")
    
    # Find a good pair with matches
    sg_matches = out_dir / "superglue" / "matches.h5"
    sg_features = out_dir / "superglue" / "features.h5"
    
    if not sg_matches.exists() or not sg_features.exists():
        print("Matches not generated yet.")
        return
        
    # Generate 5 examples
    num_examples = 5
    generated = 0
    capri_matches_path = out_dir / "capri" / "matches_capri.h5"
    
    with h5py.File(sg_matches, 'r') as f_match, h5py.File(sg_features, 'r') as f_feat:
        capri_f = None
        if capri_matches_path.exists():
            capri_f = h5py.File(capri_matches_path, 'r')
            
        img1_keys = list(f_match.keys())
        
        for i in range(10, len(img1_keys)):
            if generated >= num_examples:
                break
                
            img1_name = img1_keys[i]
            if len(f_match[img1_name].keys()) == 0:
                continue
                
            img2_name = list(f_match[img1_name].keys())[0]
            
            matches0 = f_match[img1_name][img2_name]['matches0'][:]
            valid = matches0 > -1
            sg_match_indices = np.stack([np.where(valid)[0], matches0[valid]], -1)
            
            # We want pairs with a decent amount of matches
            if len(sg_match_indices) < 100:
                continue
                
            kp1 = f_feat[img1_name]['keypoints'][:]
            kp2 = f_feat[img2_name]['keypoints'][:]
            
            out_sg = FIGURES_DIR / f"superglue_matches_{generated}.png"
            draw_matches(dataset_dir / img1_name, dataset_dir / img2_name, kp1, kp2, sg_match_indices, f"SuperGlue Matches ({len(sg_match_indices)})", out_sg)
            print(f"Saved {out_sg}")
            
            if capri_f is not None and img1_name in capri_f and img2_name in capri_f[img1_name]:
                matches0_capri = capri_f[img1_name][img2_name]['matches0'][:]
                valid_capri = matches0_capri > -1
                capri_match_indices = np.stack([np.where(valid_capri)[0], matches0_capri[valid_capri]], -1)
                
                capri_set = set(map(tuple, capri_match_indices))
                sg_set = set(map(tuple, sg_match_indices))
                rejected = list(sg_set - capri_set)
                
                out_capri = FIGURES_DIR / f"capri_matches_{generated}.png"
                draw_matches(dataset_dir / img1_name, dataset_dir / img2_name, kp1, kp2, capri_match_indices, f"CAPRI Filtered Matches ({len(capri_match_indices)} accepted, {len(rejected)} rejected)", out_capri, capri_rejected=rejected)
                print(f"Saved {out_capri}")
            
            generated += 1
            
        if capri_f is not None:
            capri_f.close()

if __name__ == '__main__':
    main()
