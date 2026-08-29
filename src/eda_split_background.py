import argparse
import json
import glob
import os
from collections import defaultdict, Counter
import config

import numpy as np
import matplotlib.pyplot as plt

#use: python eda_split_background.py --data-dir ./dataset_npz --out-dir ./eda_out


# --------------------------------------------------------------------------- #
# Caricamento dati
# --------------------------------------------------------------------------- #

def list_data_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"Nessun .npz/.npy trovato in {data_dir}")
    return files


def load_window(path):
    d = np.load(path)
    cir = d["radar_cir_iq"]   # (T, 6, 3, 120, 2)
    xy = d["people_xy"]       # (T, 4, 2)
    mask = d["people_mask"]   # (T, 4)
    ts = d["timestamps"] if "timestamps" in d else np.arange(cir.shape[0])
    return cir, xy, mask, ts


def load_background_window(data_dir, bg_idx=config.EMA_BG_WINDOW):
    """Carica i dati della finestra della stanza vuota dato il suo ID numerico."""
    filename = f"window_{bg_idx:06d}.npz"
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath_npy = os.path.join(data_dir, f"window_{bg_idx:06d}.npy")
        if os.path.exists(filepath_npy):
            filepath = filepath_npy
        else:
            raise FileNotFoundError(f"File della stanza vuota (finestra {bg_idx}) non trovato: {filepath}")
    
    cir_bg, _, _, _ = load_window(filepath)
    return cir_bg


# --------------------------------------------------------------------------- #
# EDA & Declutter
# --------------------------------------------------------------------------- #

def static_background_declutter(target_window, background_window):
    """
    Rimuove il clutter statico sottraendo il profilo medio calcolato
    dalla registrazione della stanza vuota.
    """
    bg_static_profile = np.mean(background_window, axis=0, keepdims=True)
    return target_window - bg_static_profile


def analyze_dataset(files, bg_cir, room_w=4.8, room_h=7.2, grid_res=0.1):
    """
    Percorre tutti i file e accumula le statistiche EDA usando
    il decluttering della stanza vuota.
    """
    n_people_per_frame_hist = Counter()
    file_n_people = {}                 # file -> n persone dominante nel file
    file_n_frames = {}                 # file -> numero di frame
    per_file_xy_mask = {}              # file -> (xy, mask) per densità spaziale per-file

    grid_w = int(round(room_w / grid_res))
    grid_h = int(round(room_h / grid_res))
    density_overall = np.zeros((grid_h, grid_w), dtype=np.float64)

    radar_variance_sum = None
    radar_variance_count = 0

    for path in files:
        cir, xy, mask, ts = load_window(path)
        T = cir.shape[0]
        file_n_frames[path] = T

        # --- persone per frame ---
        n_people_frame = mask.sum(axis=1).astype(int)   # (T,)
        n_people_per_frame_hist.update(n_people_frame.tolist())
        
        vals, counts = np.unique(n_people_frame, return_counts=True)
        file_n_people[path] = int(vals[np.argmax(counts)])
        per_file_xy_mask[path] = (xy, mask)

        # --- densità spaziale ---
        valid_xy = xy[mask.astype(bool)]   # (N_valid, 2)
        if valid_xy.size > 0:
            gx = np.clip((valid_xy[:, 0] / room_w * grid_w).astype(int), 0, grid_w - 1)
            gy = np.clip((valid_xy[:, 1] / room_h * grid_h).astype(int), 0, grid_h - 1)
            np.add.at(density_overall, (gy, gx), 1)

        # --- varianza temporale per radar (su segnale decluttered da stanza vuota) ---
        declut = static_background_declutter(cir, bg_cir)
        mag = np.sqrt(declut[..., 0] ** 2 + declut[..., 1] ** 2)   # (T, R, A, B)
        var_per_radar = mag.var(axis=0).mean(axis=(1, 2))          # (R,)
        if radar_variance_sum is None:
            radar_variance_sum = np.zeros_like(var_per_radar)
        radar_variance_sum += var_per_radar
        radar_variance_count += 1

    radar_variance_mean = radar_variance_sum / max(radar_variance_count, 1)

    return {
        "n_people_per_frame_hist": n_people_per_frame_hist,
        "file_n_people": file_n_people,
        "file_n_frames": file_n_frames,
        "per_file_xy_mask": per_file_xy_mask,
        "density_overall": density_overall,
        "radar_variance_mean": radar_variance_mean,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "room_w": room_w,
        "room_h": room_h,
    }


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

def split_windows_from_config(data_dir, 
                              train_wins=config.TRAIN_WINDOWS, 
                              val_wins=config.VAL_WINDOWS, 
                              test_wins=config.TEST_WINDOWS):
    """
    Divide le finestre formattate come 'window_XXXXXX.npz' nei tre subset 
    costruendo direttamente i file path a partire dagli ID numerici.
    """
    def build_filepath(win_idx):
        filename = f"window_{win_idx:06d}.npz"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            filepath_npy = os.path.join(data_dir, f"window_{win_idx:06d}.npy")
            if os.path.exists(filepath_npy):
                return filepath_npy
            raise FileNotFoundError(f"File non trovato per la finestra {win_idx}: {filepath}")
            
        return filepath

    return {
        "train": [build_filepath(i) for i in train_wins],
        "val": [build_filepath(i) for i in val_wins],
        "test": [build_filepath(i) for i in test_wins]
    }


# --------------------------------------------------------------------------- #
# Copertura spaziale
# --------------------------------------------------------------------------- #

def spatial_density_for_files(per_file_xy_mask, files, grid_w, grid_h, room_w, room_h):
    density = np.zeros((grid_h, grid_w), dtype=np.float64)
    for f in files:
        xy, mask = per_file_xy_mask[f]
        valid_xy = xy[mask.astype(bool)]
        if valid_xy.size == 0:
            continue
        gx = np.clip((valid_xy[:, 0] / room_w * grid_w).astype(int), 0, grid_w - 1)
        gy = np.clip((valid_xy[:, 1] / room_h * grid_h).astype(int), 0, grid_h - 1)
        np.add.at(density, (gy, gx), 1)
    return density


def density_kl_divergence(p_counts, q_counts, eps=1e-6):
    p = p_counts.astype(np.float64) + eps
    q = q_counts.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_persons_per_frame(hist, out_path):
    xs = sorted(hist.keys())
    ys = [hist[x] for x in xs]
    plt.figure(figsize=(6, 4))
    plt.bar([str(x) for x in xs], ys)
    plt.title("Distribuzione del numero di persone per frame")
    plt.xlabel("Numero di persone")
    plt.ylabel("Numero di frame")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_density(density, room_w, room_h, title, out_path):
    plt.figure(figsize=(5, 6))
    plt.imshow(density, origin="lower", extent=[0, room_w, 0, room_h],
               aspect="auto", cmap="viridis")
    plt.colorbar(label="Densità di ground-truth")
    plt.title(title)
    plt.xlabel("X (metri)")
    plt.ylabel("Y (metri)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_radar_variance(var_per_radar, out_path):
    labels = [f"Radar {i+1}" for i in range(len(var_per_radar))]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, var_per_radar, color="skyblue")
    plt.title("Varianza media temporale del segnale per ciascun radar")
    plt.ylabel("Varianza media del segnale")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_persons_per_file(file_n_people, out_path):
    files = sorted(file_n_people.keys(), key=lambda f: os.path.basename(f))
    ys = [file_n_people[f] for f in files]
    labels = [f"File {i}" for i in range(len(files))]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, ys, color="salmon")
    plt.title("Numero di persone (dominante) in ciascun file")
    plt.ylabel("Numero di persone")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_split_density(per_file_xy_mask, split, grid_w, grid_h, room_w, room_h, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, name in zip(axes, ["train", "val", "test"]):
        d = spatial_density_for_files(per_file_xy_mask, split[name], grid_w, grid_h, room_w, room_h)
        im = ax.imshow(d, origin="lower", extent=[0, room_w, 0, room_h], aspect="auto", cmap="viridis")
        ax.set_title(f"{name} ({len(split[name])} file)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Densità di ground-truth")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Cartella con i file .npz/.npy")
    ap.add_argument("--out-dir", default="./eda_out")
    ap.add_argument("--grid-res", type=float, default=0.1, help="Risoluzione griglia densità in metri")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = list_data_files(args.data_dir)
    print(f"Trovati {len(files)} file in {args.data_dir}")

    # Caricamento della finestra di background (stanza vuota)
    print(f"Caricamento background dalla finestra #{config.EMA_BG_WINDOW}...")
    bg_cir = load_background_window(args.data_dir, bg_idx=config.EMA_BG_WINDOW)

    stats = analyze_dataset(
        files,
        bg_cir=bg_cir,
        grid_res=args.grid_res,
    )

    # --- Plot EDA ---
    plot_persons_per_frame(stats["n_people_per_frame_hist"],
                            os.path.join(args.out_dir, "persons_per_frame_hist.png"))
    plot_density(stats["density_overall"], stats["room_w"], stats["room_h"],
                 "Densità di ground-truth delle posizioni delle persone nella stanza",
                 os.path.join(args.out_dir, "spatial_density_overall.png"))
    plot_radar_variance(stats["radar_variance_mean"],
                         os.path.join(args.out_dir, "radar_variance.png"))
    plot_persons_per_file(stats["file_n_people"],
                           os.path.join(args.out_dir, "persons_per_file.png"))

    # --- Split da config.py ---
    split = split_windows_from_config(
        args.data_dir,
        train_wins=config.TRAIN_WINDOWS,
        val_wins=config.VAL_WINDOWS,
        test_wins=config.TEST_WINDOWS
    )

    plot_split_density(stats["per_file_xy_mask"], split,
                        stats["grid_w"], stats["grid_h"],
                        stats["room_w"], stats["room_h"],
                        os.path.join(args.out_dir, "spatial_density_train_val_test.png"))

    kl_train_val = density_kl_divergence(
        spatial_density_for_files(stats["per_file_xy_mask"], split["train"], stats["grid_w"], stats["grid_h"], stats["room_w"], stats["room_h"]),
        spatial_density_for_files(stats["per_file_xy_mask"], split["val"], stats["grid_w"], stats["grid_h"], stats["room_w"], stats["room_h"]),
    )
    kl_train_test = density_kl_divergence(
        spatial_density_for_files(stats["per_file_xy_mask"], split["train"], stats["grid_w"], stats["grid_h"], stats["room_w"], stats["room_h"]),
        spatial_density_for_files(stats["per_file_xy_mask"], split["test"], stats["grid_w"], stats["grid_h"], stats["room_w"], stats["room_h"]),
    )
    coverage_note = (
        f"KL(train||val) = {kl_train_val:.3f}, KL(train||test) = {kl_train_test:.3f} "
        "(valori piu' alti = distribuzione spaziale piu' diversa dal train)"
    )

    split_serializable = {k: [os.path.basename(f) for f in v] for k, v in split.items()}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({
            "mode": "config_fixed",
            "bg_window": config.EMA_BG_WINDOW,
            "n_files": len(files),
            "file_n_people": {os.path.basename(k): v for k, v in stats["file_n_people"].items()},
            "split": split_serializable,
        }, f, indent=2)

    # --- Report ---
    hist = stats["n_people_per_frame_hist"]
    total_frames = sum(hist.values())
    lines = []
    lines.append(f"File analizzati: {len(files)}")
    lines.append(f"Finestra Stanza Vuota (Background): window_{config.EMA_BG_WINDOW:06d}.npz")
    lines.append(f"Frame totali: {total_frames}")
    lines.append("Distribuzione persone/frame:")
    for n in sorted(hist.keys()):
        lines.append(f"  {n} persone: {hist[n]} frame ({100*hist[n]/total_frames:.1f}%)")
    lines.append("")
    lines.append("Varianza media per radar (dopo sottrazione stanza vuota):")
    for i, v in enumerate(stats["radar_variance_mean"]):
        lines.append(f"  Radar {i+1}: {v:.1f}")
    ranked = np.argsort(-stats["radar_variance_mean"])
    lines.append(f"  Ranking (decrescente): {[f'Radar {i+1}' for i in ranked]}")
    lines.append("")
    lines.append("Persone per file (fascia dominante):")
    for f in sorted(stats["file_n_people"].keys(), key=lambda p: os.path.basename(p)):
        lines.append(f"  {os.path.basename(f)}: {stats['file_n_people'][f]} persone, "
                      f"{stats['file_n_frames'][f]} frame")
    lines.append("")
    lines.append("Modalita' split: config_fixed (da config.py)")
    lines.append(coverage_note)

    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "eda_report.txt"), "w") as f:
        f.write(report)

    print(report)
    print(f"\nOutput salvati in: {args.out_dir}")


if __name__ == "__main__":
    main()