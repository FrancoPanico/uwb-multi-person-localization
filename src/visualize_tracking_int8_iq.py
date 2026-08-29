"""visualize_tracking_int8_iq.py

Uso tipico (tutti e 24 i file, train+val+test, configurazione finale I/Q):
    python visualize_tracking_int8_iq.py --data-dir ./dataset_uwb/data \
        --split-json ./eda_out/split.json \
        --out-dir ./tracking_videos_iq_final

Anteprima veloce su un solo file di test, un frame ogni 4:
    python visualize_tracking_int8_iq.py --data-dir ./dataset_uwb/data \
        --split-json ./eda_out/split.json --splits test \
        --files window_000015.npz --stride 4 --out-dir ./tracking_videos_iq_final

Solo split di test (i 5 file usati per la valutazione F1=0.8778):
    python visualize_tracking_int8_iq.py --data-dir ./dataset_uwb/data \
        --split-json ./eda_out/split.json --splits test \
        --out-dir ./tracking_videos_iq_final
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from evaluate_v2 import decode_heatmap
from generate_dataset_iq_experiment import (
    ROOM_H,
    ROOM_W,
    apply_norm_v2,
    build_windows_multi,
    dual_ema_declutter_iq,
)
from tracking import HeatmapTracker

try:
    from geometry_and_projection import RADAR_GEOMETRY
except Exception:
    RADAR_GEOMETRY = None

try:
    import imageio.v2 as imageio
except ImportError:
    raise SystemExit(
        "Manca 'imageio'"
    )


def quantize_data(x_float, scale, zero_point):
    if scale == 0:
        return x_float.astype(np.int8)
    q = np.round(x_float / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_data(x_int8, scale, zero_point):
    return (x_int8.astype(np.float32) - zero_point) * scale


def run_file_inference_tflite_iq(
    interpreter,
    input_map,
    out_detail,
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
):
    channels = dual_ema_declutter_iq(
        cir, radar_idx, alpha_slow, alpha_fast, ema_mode, use_log1p
    )
    windows, frame_idx = build_windows_multi(channels, k)
    if windows.shape[0] == 0:
        return np.empty((0,)), None

    out_scale, out_zp = out_detail["quantization"]
    heat_all = []
    for i in range(windows.shape[0]):
        window = windows[i]
        normed = apply_norm_v2(window, mean, std, k).astype(np.float32)

        for ri, r in enumerate(radars_1based):
            inp_det = input_map[r]
            inp_scale, inp_zp = inp_det["quantization"]
            radar_feat = normed[ri][None, ...]
            quant_feat = quantize_data(radar_feat, inp_scale, inp_zp)
            interpreter.set_tensor(inp_det["index"], quant_feat)

        interpreter.invoke()
        raw_out = interpreter.get_tensor(out_detail["index"])
        heat_prob = dequantize_data(raw_out, out_scale, out_zp)[0, ..., 0]
        heat_all.append(heat_prob)

    heat_all = np.stack(heat_all, axis=0)
    return frame_idx, heat_all


def render_file_video(
    fname,
    interpreter,
    input_map,
    out_detail,
    cir,
    xy,
    mask,
    radar_idx,
    radars_1based,
    k,
    mean,
    std,
    ema_mode,
    alpha_slow,
    alpha_fast,
    use_log1p,
    grid_w,
    grid_h,
    thresh,
    max_people,
    nms_radius,
    match_dist,
    center_radius,
    center_thresh_boost,
    refine,
    stride,
    fps,
    out_path,
    dpi,
    thresh_high,
    max_missed,
    min_hits,
    proc_pos,
    proc_vel,
    meas_noise,
):
    frame_idx, heat_all = run_file_inference_tflite_iq(
        interpreter,
        input_map,
        out_detail,
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
    )
    if heat_all is None or len(frame_idx) == 0:
        print(f"  [{fname}] nessun frame valido, salto.")
        return

    tracker = HeatmapTracker(
        max_tracks=max_people,
        max_match_dist=match_dist,
        max_missed=max_missed,
        min_hits_to_report=min_hits,
        thresh_high=thresh_high,
        process_noise_pos=proc_pos,
        process_noise_vel=proc_vel,
        meas_noise=meas_noise,
    )

    # --- Figura creata UNA volta sola, poi solo aggiornata. ---
    fig, ax = plt.subplots(figsize=(6, 8), dpi=dpi)
    im = ax.imshow(
        np.zeros((grid_h, grid_w)),
        origin="lower",
        extent=[0, ROOM_W, 0, ROOM_H],
        vmin=0.0,
        vmax=1.0,
        cmap="inferno",
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_xlim(0, ROOM_W)
    ax.set_ylim(0, ROOM_H)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    gt_scatter = ax.scatter(
        [], [], s=140, facecolors="none", edgecolors="cyan", linewidths=2.0,
        label="Ground truth",
    )
    raw_scatter = ax.scatter(
        [], [], s=30, c="lightgray", alpha=0.7, marker="x",
        label="Rilevazione grezza (pre-tracking)",
    )
    tracked_scatter = ax.scatter(
        [], [], s=90, c="red", marker="o", edgecolors="black", linewidths=0.8,
        label="Output tracker",
    )

    if RADAR_GEOMETRY is not None:
        for rid, info in RADAR_GEOMETRY.items():
            rx, ry = info["pos"]
            ax.plot(rx, ry, marker="^", color="deepskyblue", markersize=9,
                     markeredgecolor="black", zorder=1)
            ax.annotate(str(rid), (rx, ry), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="deepskyblue")

    title_text = ax.set_title("")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)

    n_written = 0
    t0 = time.time()
    for i, t in enumerate(frame_idx):
        heat_frame = heat_all[i]
        gt_pts = xy[t][mask[t].astype(bool)]

        raw_pts_with_scores = decode_heatmap(
            heat_frame, grid_w, grid_h, thresh, max_people, nms_radius,
            refine, center_radius=center_radius,
            center_thresh_boost=center_thresh_boost,
        )
        tracked_pts = tracker.update(raw_pts_with_scores)

        if stride > 1 and (i % stride != 0):
            continue

        im.set_data(np.clip(heat_frame, 0.0, 1.0))

        if len(gt_pts) > 0:
            gt_scatter.set_offsets(gt_pts)
        else:
            gt_scatter.set_offsets(np.empty((0, 2)))

        if len(raw_pts_with_scores) > 0:
            raw_scatter.set_offsets([[p[0], p[1]] for p in raw_pts_with_scores])
        else:
            raw_scatter.set_offsets(np.empty((0, 2)))

        if len(tracked_pts) > 0:
            tracked_scatter.set_offsets(tracked_pts)
        else:
            tracked_scatter.set_offsets(np.empty((0, 2)))

        title_text.set_text(
            f"{fname} (I/Q trial015)  |  frame reale t={t}  (indice finestra {i}/{len(frame_idx)})\n"
            f"GT={len(gt_pts)}  raw={len(raw_pts_with_scores)}  tracciate={len(tracked_pts)}"
        )

        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(frame_rgba[..., :3])
        n_written += 1

        if n_written % 500 == 0:
            elapsed = time.time() - t0
            print(f"    ...{n_written} frame scritti ({elapsed:.1f}s)")

    writer.close()
    plt.close(fig)
    elapsed = time.time() - t0
    print(
        f"  [{fname}] video salvato: {out_path}  "
        f"({n_written} frame scritti su {len(frame_idx)} reali, {elapsed:.1f}s)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True")
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--dataset-meta", default="./dataset_k5_m2_iq/dataset_meta_v2.json")
    ap.add_argument("--tflite-model", default="./model_iq_trial15_int8.tflite")
    ap.add_argument("--splits", default="test",
                     help="Split da processare, separati da virgola (train/val/test)")
    ap.add_argument("--files", default=None,
                     help="Se specificato (lista separata da virgola), limita ai soli file indicati "
                          "invece di tutti quelli dello split")
    ap.add_argument("--thresh", type=float, default=0.27)
    ap.add_argument("--nms-radius", type=float, default=0.70)
    ap.add_argument("--max-people", type=int, default=4)
    ap.add_argument("--match-dist", type=float, default=2.0)
    ap.add_argument("--center-radius", type=float, default=1.1)
    ap.add_argument("--center-thresh-boost", type=float, default=0.02)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--thresh-high", type=float, default=0.43,
                     help="Soglia alta per aprire nuove tracce (t_high, T_HIGH del candidato I/Q trial015)")
    ap.add_argument("--max-missed", type=int, default=9)
    ap.add_argument("--min-hits", type=int, default=9)
    ap.add_argument("--proc-pos", type=float, default=0.028308131373182843)
    ap.add_argument("--proc-vel", type=float, default=0.0012012262662818589)
    ap.add_argument("--meas-noise", type=float, default=0.70)
    ap.add_argument("--stride", type=int, default=1,
                     help="Disegna 1 frame ogni N (il tracker avanza comunque su tutti i frame reali)")
    ap.add_argument("--fps", type=int, default=15, help="FPS del video generato (velocita' di riproduzione)")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument(
        "--length-frac", type=float, default=0.25,
        help="Frazione della lunghezza di ciascun file da processare (default "
             "0.25 = un quarto, preso dall'INIZIO del file, PRIMA di qualunque "
             "preprocessing/inferenza). Passa 1.0 per il file intero."
    )
    ap.add_argument("--out-dir", default="./tracking_videos_iq")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.dataset_meta) as f:
        meta = json.load(f)
    with open(args.split_json) as f:
        split_data = json.load(f)
    split = split_data["split"]
    if "folds" in split:
        raise ValueError("split.json in modalita' k-fold: passare uno split 'single_split'.")

    if meta.get("use_abs_ref", False):
        raise ValueError(
            "Questo dataset-meta ha use_abs_ref=True, ma la variante I/Q "
            "(dual_ema_declutter_iq) non supporta static_ref -- usare "
            "visualize_tracking_int8.py (pipeline a magnitudine) per questo meta."
        )
    if not meta.get("iq_mode", False):
        print(
            "ATTENZIONE: il dataset-meta passato non ha iq_mode=True. "
            "Verificare di aver passato dataset_k5_m2_iq/dataset_meta_v2.json "
            "e non il meta del candidato a magnitudine."
        )

    radars_1based = meta["radars_1based"]
    radar_idx = [r - 1 for r in radars_1based]
    k = meta["k"]
    grid_h, grid_w = meta["grid_shape"]
    mean = np.array(meta["norm_mean"], dtype=np.float32)
    std = np.array(meta["norm_std"], dtype=np.float32)
    ema_mode = meta.get("ema_mode", "v2")
    use_log1p = meta.get("use_log1p", True)
    alpha_slow = meta.get("alpha_slow", 0.02)
    alpha_fast = meta.get("alpha_fast", 0.15)

    print(f"Carico modello TFLite da {args.tflite_model} ...")
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_map = {}
    for detail in input_details:
        tensor_name = detail["name"]
        for r in radars_1based:
            if f"radar_{r}" in tensor_name:
                input_map[r] = detail
                break
    out_detail = output_details[0]

    requested_files = None
    if args.files:
        requested_files = set(x.strip() for x in args.files.split(","))

    refine = not args.no_refine

    for split_name in [s.strip() for s in args.splits.split(",")]:
        files = split[split_name]
        if requested_files is not None:
            files = [f for f in files if f in requested_files]

        split_out_dir = os.path.join(args.out_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        print(f"\n=== Split '{split_name}': {len(files)} file -> {split_out_dir} ===")
        for fname in files:
            path = os.path.join(args.data_dir, fname)
            d = np.load(path)
            cir = d["radar_cir_iq"]
            xy = d["people_xy"]
            mask = d["people_mask"]

            if args.length_frac < 1.0:
                t_full = cir.shape[0]
                t_use = max(6, int(t_full * args.length_frac))  # almeno k+1 frame
                cir = cir[:t_use]
                xy = xy[:t_use]
                mask = mask[:t_use]
                print(
                    f"    ({fname}: uso i primi {t_use}/{t_full} frame, "
                    f"length-frac={args.length_frac})"
                )

            out_path = os.path.join(
                split_out_dir, f"{fname.replace('.npz', '')}.mp4"
            )
            print(f"  elaboro {fname} -> {out_path}")
            render_file_video(
                fname, interpreter, input_map, out_detail, cir, xy, mask,
                radar_idx, radars_1based, k,
                mean, std, ema_mode, alpha_slow, alpha_fast, use_log1p,
                grid_w, grid_h, args.thresh, args.max_people,
                args.nms_radius, args.match_dist, args.center_radius,
                args.center_thresh_boost, refine, args.stride, args.fps,
                out_path, args.dpi,
                args.thresh_high, args.max_missed, args.min_hits,
                args.proc_pos, args.proc_vel, args.meas_noise,
            )

    print("\nFatto.")


if __name__ == "__main__":
    main()
