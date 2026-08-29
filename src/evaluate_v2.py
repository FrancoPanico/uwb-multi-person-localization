"""
evaluate_v2.py
==============
Calcola l'F1 con Hungarian matching (soglia 1.0m o custom) del modello v2
(backprojection + heatmap), confrontando le rilevazioni grezze (SENZA Tracking)
e quelle filtrate dal Filtro di Kalman (CON Tracking).

Integrazione Dual-Thresholding & Maschera Spaziale Anti-Fantasma:
- decode_heatmap estrae [x, y, score] usando thresh_low (0.20 da CLI).
- Applica una soglia dinamica maggiorata (thresh_low + center_thresh_boost)
  esattamente nella bolla centrale attorno al centro stanza (ROOM_W/2, ROOM_H/2).
- HeatmapTracker gestisce l'isteresi applicando thresh_high per le nuove tracce.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf

from generate_dataset_ablation import (
    ROOM_H,
    ROOM_W,
    apply_norm_v2,
    build_windows_v2,
    dual_ema_declutter,
)
from tracking import HeatmapTracker


# --------------------------------------------------------------------------- #
# Decodifica: heatmap -> punti [x, y, score], con soppressione al centro stanza
# --------------------------------------------------------------------------- #

def decode_heatmap(
    heat,
    grid_w,
    grid_h,
    thresh=0.20,             # Soglia d'ingresso base (thresh_low)
    max_people=4,
    nms_radius=0.50,
    refine=True,
    center_radius=0.80,      # Raggio di soppressione centrale (metri)
    center_thresh_boost=0.10 # Offset di soglia applicato al centro stanza
):
    cell_w, cell_h = ROOM_W / grid_w, ROOM_H / grid_h

    # Centro geometrico reale della stanza (2.0m, 2.0m per griglia 4x4m)
    center_x = ROOM_W / 2.0
    center_y = ROOM_H / 2.0

    # Mappa delle coordinate metriche per ogni cella della griglia
    gy_idx, gx_idx = np.indices(heat.shape)
    x_coords = (gx_idx + 0.5) * cell_w
    y_coords = (gy_idx + 0.5) * cell_h

    # Distanza euclidea dal centro reale della stanza
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

    # Soglia dinamica: maggiorata nella bolla centrale
    dynamic_thresh = np.where(dist_from_center <= center_radius, thresh + center_thresh_boost, thresh)

    ys, xs = np.where(heat > dynamic_thresh)
    if len(ys) == 0:
        return []

    scores = heat[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    points = []
    for gy, gx, sc in zip(ys, xs, scores):
        dx = dy = 0.0
        if refine:
            if 0 < gx < grid_w - 1:
                L, C, R = heat[gy, gx - 1], heat[gy, gx], heat[gy, gx + 1]
                denom_x = 2.0 * (2.0 * C - L - R)
                if denom_x > 1e-9:
                    dx = np.clip((R - L) / denom_x, -0.5, 0.5)
            if 0 < gy < grid_h - 1:
                B, C, T = heat[gy - 1, gx], heat[gy, gx], heat[gy + 1, gx]
                denom_y = 2.0 * (2.0 * C - B - T)
                if denom_y > 1e-9:
                    dy = np.clip((T - B) / denom_y, -0.5, 0.5)

        x = float(np.clip((gx + dx + 0.5) * cell_w, 0.0, ROOM_W))
        y = float(np.clip((gy + dy + 0.5) * cell_h, 0.0, ROOM_H))

        too_close = any(
            (x - px) ** 2 + (y - py) ** 2 < nms_radius**2 for px, py, _ in points
        )
        if not too_close:
            points.append((x, y, float(sc)))
        if len(points) >= max_people:
            break

    return points


def match_frame(pred, gt, thresh=1.0):
    if len(pred) == 0 and len(gt) == 0:
        return 0, 0, 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    D = np.linalg.norm(
        np.array(pred)[:, None, :] - np.array(gt)[None, :, :], axis=-1
    )
    r, c = linear_sum_assignment(D)
    tp = sum(1 for i, j in zip(r, c) if D[i, j] <= thresh)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


# --------------------------------------------------------------------------- #
# Inferenza su un intero file
# --------------------------------------------------------------------------- #

def run_file_inference(
    model,
    cir,
    radar_idx,
    radars_1based,
    k,
    mean,
    std,
    ema_mode="v2",
    alpha_slow=0.02,
    alpha_fast=0.15,
    use_log1p=False,
    batch_size=256,
    static_ref=None,
):
    channels = dual_ema_declutter(
        cir, radar_idx, alpha_slow, alpha_fast, ema_mode, use_log1p, static_ref
    )
    if static_ref is None:
        windows, frame_idx = build_windows_v2(channels[0], channels[1], k)
    else:
        windows, frame_idx = build_windows_v2(
            channels[0], channels[1], k, mag_abs=channels[2]
        )
    if windows.shape[0] == 0:
        return np.empty((0,)), None

    n_win = windows.shape[0]
    heat_all = []
    for i in range(0, n_win, batch_size):
        batch = windows[i : i + batch_size]
        normed = np.stack(
            [apply_norm_v2(batch[j], mean, std, k) for j in range(batch.shape[0])]
        )
        normed = normed.astype(np.float32)
        inputs = {f"radar_{r}": normed[:, ri] for ri, r in enumerate(radars_1based)}
        _, prob = model(inputs, training=False)
        heat_all.append(prob.numpy()[..., 0])
    heat_all = np.concatenate(heat_all, axis=0)
    return frame_idx, heat_all


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--dataset-meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument(
        "--thresholds",
        default="0.20",
        help="Soglia d'ingresso per la decodifica (thresh_low)",
    )
    ap.add_argument("--nms-radius", type=float, default=0.50)
    ap.add_argument("--max-people", type=int, default=4)
    ap.add_argument("--match-dist", type=float, default=1.40)
    ap.add_argument(
        "--center-radius",
        type=float,
        default=0.80,
        help="Raggio (m) attorno al centro della stanza per la soppressione",
    )
    ap.add_argument(
        "--center-thresh-boost",
        type=float,
        default=0.10,
        help="Incremento di soglia richiesto nella bolla centrale",
    )
    ap.add_argument(
        "--no-refine",
        action="store_true",
        help="Disattiva il raffinamento sub-cella",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="./eval_v2_report.json")
    args = ap.parse_args()

    with open(args.dataset_meta) as f:
        meta = json.load(f)
    with open(args.split_json) as f:
        split_data = json.load(f)

    split = split_data["split"]
    if "folds" in split:
        raise ValueError(
            "split.json in modalita' k-fold: passare uno split 'single_split'."
        )
    files = split[args.split]

    radars_1based = meta["radars_1based"]
    radar_idx = [r - 1 for r in radars_1based]
    k = meta["k"]
    grid_h, grid_w = meta["grid_shape"]
    mean = np.array(meta["norm_mean"], dtype=np.float32)
    std = np.array(meta["norm_std"], dtype=np.float32)

    ema_mode = meta.get("ema_mode", "v2")
    use_log1p = meta.get("use_log1p", False)
    alpha_slow = meta.get("alpha_slow", 0.02)
    alpha_fast = meta.get("alpha_fast", 0.15)
    use_abs_ref = meta.get("use_abs_ref", False)
    static_ref = (
        np.array(meta["static_ref"], dtype=np.float32) if use_abs_ref else None
    )

    print(
        f"Valutazione su split='{args.split}' ({len(files)} file),"
        f" radar={radars_1based}, K={k}"
    )
    print(
        "Configurazione preprocessing (da dataset_meta_v2.json):"
        f" ema_mode={ema_mode}, use_log1p={use_log1p}, alpha_slow={alpha_slow},"
        f" alpha_fast={alpha_fast}, use_abs_ref={use_abs_ref}"
    )

    model = tf.keras.models.load_model(args.model, compile=False)

    thresholds = [float(t) for t in args.thresholds.split(",")]
    refine = not args.no_refine

    cache = []
    for fname in files:
        path = os.path.join(args.data_dir, fname)
        d = np.load(path)
        cir = d["radar_cir_iq"]
        xy = d["people_xy"]
        mask = d["people_mask"]

        frame_idx, heat_all = run_file_inference(
            model,
            cir,
            radar_idx,
            radars_1based,
            k,
            mean,
            std,
            ema_mode,
            alpha_slow,
            alpha_fast,
            use_log1p,
            args.batch_size,
            static_ref,
        )
        cache.append((fname, frame_idx, heat_all, xy, mask))
        print(f"  inferenza completata: {fname} ({len(frame_idx)} frame)")

    results = {}
    best_f1 = -1.0
    best_thresh = None

    for thresh in thresholds:
        tot_raw_tp = tot_raw_fp = tot_raw_fn = 0
        tot_tr_tp = tot_tr_fp = tot_tr_fn = 0
        per_file_metrics = {}

        for fname, frame_idx, heat_all, xy, mask in cache:
            tracker = HeatmapTracker(
                max_tracks=args.max_people,
                max_match_dist=args.match_dist,
                max_missed=4,
                min_hits_to_report=5,
                thresh_high=0.33,
                process_noise_pos=0.0935,
                process_noise_vel=0.0158,
                meas_noise=0.4000,
            )

            f_raw_tp = f_raw_fp = f_raw_fn = 0
            f_tr_tp = f_tr_fp = f_tr_fn = 0

            for i, t in enumerate(frame_idx):
                heat_frame = heat_all[i]
                gt_pts = xy[t][mask[t].astype(bool)].tolist()

                raw_pts_with_scores = decode_heatmap(
                    heat_frame,
                    grid_w,
                    grid_h,
                    thresh,
                    args.max_people,
                    args.nms_radius,
                    refine,
                    center_radius=args.center_radius,
                    center_thresh_boost=args.center_thresh_boost,
                )

                tracked_pts = tracker.update(raw_pts_with_scores)
                raw_pts = [[p[0], p[1]] for p in raw_pts_with_scores]

                tp_r, fp_r, fn_r = match_frame(raw_pts, gt_pts, args.match_dist)
                f_raw_tp += tp_r
                f_raw_fp += fp_r
                f_raw_fn += fn_r

                tp_t, fp_t, fn_t = match_frame(tracked_pts, gt_pts, args.match_dist)
                f_tr_tp += tp_t
                f_tr_fp += fp_t
                f_tr_fn += fn_t

            f_p_tr = f_tr_tp / (f_tr_tp + f_tr_fp + 1e-9)
            f_r_tr = f_tr_tp / (f_tr_tp + f_tr_fn + 1e-9)
            f_f1_tr = 2 * f_p_tr * f_r_tr / (f_p_tr + f_r_tr + 1e-9)

            f_p_raw = f_raw_tp / (f_raw_tp + f_raw_fp + 1e-9)
            f_r_raw = f_raw_tp / (f_raw_tp + f_raw_fn + 1e-9)
            f_f1_raw = 2 * f_p_raw * f_r_raw / (f_p_raw + f_raw_fn + 1e-9)

            per_file_metrics[fname] = {
                "raw": {
                    "tp": f_raw_tp,
                    "fp": f_raw_fp,
                    "fn": f_raw_fn,
                    "precision": round(f_p_raw, 3),
                    "recall": round(f_r_raw, 3),
                    "f1": round(f_f1_raw, 3),
                },
                "tracked": {
                    "tp": f_tr_tp,
                    "fp": f_tr_fp,
                    "fn": f_tr_fn,
                    "precision": round(f_p_tr, 3),
                    "recall": round(f_r_tr, 3),
                    "f1": round(f_f1_tr, 3),
                },
            }

            tot_raw_tp += f_raw_tp
            tot_raw_fp += f_raw_fp
            tot_raw_fn += f_raw_fn
            tot_tr_tp += f_tr_tp
            tot_tr_fp += f_tr_fp
            tot_tr_fn += f_tr_fn

        p_raw = tot_raw_tp / (tot_raw_tp + tot_raw_fp + 1e-9)
        r_raw = tot_raw_tp / (tot_raw_tp + tot_raw_fn + 1e-9)
        f1_raw = 2 * p_raw * r_raw / (p_raw + r_raw + 1e-9)

        p_tr = tot_tr_tp / (tot_tr_tp + tot_tr_fp + 1e-9)
        r_tr = tot_tr_tp / (tot_tr_tp + tot_tr_fn + 1e-9)
        f1_tr = 2 * p_tr * r_tr / (p_tr + r_tr + 1e-9)

        results[thresh] = {
            "raw": {
                "tp": tot_raw_tp,
                "fp": tot_raw_fp,
                "fn": tot_raw_fn,
                "precision": p_raw,
                "recall": r_raw,
                "f1": f1_raw,
            },
            "tracked": {
                "tp": tot_tr_tp,
                "fp": tot_tr_fp,
                "fn": tot_tr_fn,
                "precision": p_tr,
                "recall": r_tr,
                "f1": f1_tr,
            },
            "by_file": per_file_metrics,
        }

        print("\n" + "=" * 80)
        print(f"DETTAGLIO FILE PER FILE (Soglia {thresh:.2f})")
        print("=" * 80)
        for fname, m in per_file_metrics.items():
            tr_m = m["tracked"]
            print(
                f"File: {fname:<22} | P={tr_m['precision']:.3f} |"
                f" R={tr_m['recall']:.3f} | F1={tr_m['f1']:.3f}"
            )

        print("\nTOTALE SPLIT (Soglia {:.2f}):".format(thresh))
        print(
            f"{'Modalità':<15} | {'TP':<6} {'FP':<6} {'FN':<6} | {'Precision':<9}"
            f" {'Recall':<9} {'F1-Score':<9}"
        )
        print("-" * 80)
        print(
            f"{'SENZA Tracking':<15} | {tot_raw_tp:<6d} {tot_raw_fp:<6d}"
            f" {tot_raw_fn:<6d} | {p_raw:<9.3f} {r_raw:<9.3f} {f1_raw:<9.3f}"
        )
        print(
            f"{'CON Tracking':<15} | {tot_tr_tp:<6d} {tot_tr_fp:<6d} {tot_tr_fn:<6d} |"
            f" {p_tr:<9.3f} {r_tr:<9.3f} {f1_tr:<9.3f}"
        )
        print("=" * 80)

        if f1_tr > best_f1:
            best_f1 = f1_tr
            best_thresh = thresh

    print(f"\nMiglior soglia (con Tracking): {best_thresh} -> F1={best_f1:.3f}")

    with open(args.out, "w") as f:
        json.dump(
            {
                "split": args.split,
                "n_files": len(files),
                "preprocessing_config": {
                    "ema_mode": ema_mode,
                    "use_log1p": use_log1p,
                    "alpha_slow": alpha_slow,
                    "alpha_fast": alpha_fast,
                },
                "match_dist_m": args.match_dist,
                "nms_radius_m": args.nms_radius,
                "subcell_refine": refine,
                "center_suppression": {
                    "radius_m": args.center_radius,
                    "boost": args.center_thresh_boost,
                },
                "results_per_threshold": results,
                "best_threshold": best_thresh,
                "best_f1_tracked": best_f1,
            },
            f,
            indent=2,
        )
    print(f"Report salvato in {args.out}")


if __name__ == "__main__":
    main()