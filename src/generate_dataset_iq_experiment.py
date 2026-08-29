"""generate_dataset_iq_experiment.py

USO
---
python generate_dataset_iq_experiment.py --data-dir ./dataset_uwb/data \
    --split-json ./eda_out/split.json \
    --out-dir ./dataset_k5_m2_iq \
    --k 5 --grid-w 12 --grid-h 18 --sigma-m 0.2
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

ROOM_W, ROOM_H = 4.8, 7.2


def _signed_log1p(x):
    """Compressione log1p 'con segno', definita anche per x negativi.

    f(x) = sign(x) * log1p(|x|)

    Scelta di default non confermata dai paper specifici dello studente
    (vedi docstring del modulo) -- unico punto da modificare se serve
    un'altra normalizzazione.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def dual_ema_declutter_iq(cir, radar_idx, alpha_slow=0.02, alpha_fast=0.15,
                           ema_mode='v2', use_log1p=True):
    sel = cir[:, radar_idx].astype(np.float32)  # (T, R, 3, 120, 2)
    T = sel.shape[0]

    decl_slow = np.empty_like(sel)
    decl_fast = np.empty_like(sel)

    if ema_mode == 'v2':
        y_slow = np.empty_like(sel)
        y_fast = np.empty_like(sel)
        y_slow[0] = sel[0]
        y_fast[0] = sel[0]
        decl_slow[0] = np.zeros_like(sel[0])
        decl_fast[0] = np.zeros_like(sel[0])

        for t in range(1, T):
            y_slow[t] = alpha_slow * sel[t] + (1.0 - alpha_slow) * y_slow[t - 1]
            y_fast[t] = alpha_fast * sel[t] + (1.0 - alpha_fast) * y_fast[t - 1]
            decl_slow[t] = sel[t] - y_slow[t]
            decl_fast[t] = sel[t] - y_fast[t]

    elif ema_mode == 'v3':
        bg_slow = sel[0].copy()
        bg_fast = sel[0].copy()

        for t in range(T):
            decl_slow[t] = sel[t] - bg_slow
            decl_fast[t] = sel[t] - bg_fast

            bg_slow = alpha_slow * sel[t] + (1.0 - alpha_slow) * bg_slow
            bg_fast = alpha_fast * sel[t] + (1.0 - alpha_fast) * bg_fast
    else:
        raise ValueError(f"Modalita EMA non valida: {ema_mode}. Usare 'v2' o 'v3'.")

    # QUI la differenza rispetto all'originale: NON calcoliamo sqrt(I^2+Q^2).
    # Teniamo I (indice 0) e Q (indice 1) separati.
    i_slow = decl_slow[..., 0]   # (T, R, 3, 120)
    q_slow = decl_slow[..., 1]
    i_fast = decl_fast[..., 0]
    q_fast = decl_fast[..., 1]

    if use_log1p:
        i_slow = _signed_log1p(i_slow)
        q_slow = _signed_log1p(q_slow)
        i_fast = _signed_log1p(i_fast)
        q_fast = _signed_log1p(q_fast)

    # Trasposizione per convenzione RangeNet: (T, R, 120, 3), come nell'originale
    i_slow = np.transpose(i_slow, (0, 1, 3, 2)).astype(np.float32)
    q_slow = np.transpose(q_slow, (0, 1, 3, 2)).astype(np.float32)
    i_fast = np.transpose(i_fast, (0, 1, 3, 2)).astype(np.float32)
    q_fast = np.transpose(q_fast, (0, 1, 3, 2)).astype(np.float32)

    return i_slow, q_slow, i_fast, q_fast


def build_windows_multi(channels, k):
    n_var = len(channels)
    T, R, B, A = channels[0].shape
    n_win = T - k + 1
    if n_win <= 0:
        return (np.empty((0, R, B, A, n_var * k), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    out = np.empty((n_win, R, B, A, n_var * k), dtype=np.float32)
    for v, mag in enumerate(channels):
        for i in range(k):
            out[..., v * k + i] = mag[i:i + n_win]
    frame_idx = np.arange(k - 1, T)
    return out, frame_idx


def build_gaussian_heatmap(xy_frame, mask_frame, grid_w, grid_h, sigma_m=0.2):
    cell_w, cell_h = ROOM_W / grid_w, ROOM_H / grid_h
    sigma_x_cells = sigma_m / cell_w
    sigma_y_cells = sigma_m / cell_h

    gx_idx, gy_idx = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    heat = np.zeros((grid_h, grid_w), dtype=np.float32)

    for p in range(xy_frame.shape[0]):
        if not mask_frame[p]:
            continue
        x, y = xy_frame[p]
        cx_cont = np.clip(x, 0, ROOM_W - 1e-6) / cell_w - 0.5
        cy_cont = np.clip(y, 0, ROOM_H - 1e-6) / cell_h - 0.5
        cx_center = int(np.clip(round(cx_cont), 0, grid_w - 1))
        cy_center = int(np.clip(round(cy_cont), 0, grid_h - 1))

        g = np.exp(-0.5 * (((gx_idx - cx_center) / sigma_x_cells) ** 2 +
                            ((gy_idx - cy_center) / sigma_y_cells) ** 2))
        heat = np.maximum(heat, g)

    return heat.astype(np.float32)


def compute_train_stats_iq(data_dir, train_files, radar_idx, alpha_slow, alpha_fast,
                            ema_mode, use_log1p):
    n_var = 4
    n = np.zeros((n_var, 3), dtype=np.float64)
    s = np.zeros((n_var, 3), dtype=np.float64)
    ss = np.zeros((n_var, 3), dtype=np.float64)

    for fname in train_files:
        path = os.path.join(data_dir, fname)
        d = np.load(path)
        cir = d["radar_cir_iq"]
        channels = dual_ema_declutter_iq(cir, radar_idx, alpha_slow, alpha_fast,
                                          ema_mode, use_log1p)
        for variant_i, mag in enumerate(channels):
            flat = mag.reshape(-1, 3)
            n[variant_i] += flat.shape[0]
            s[variant_i] += flat.sum(axis=0)
            ss[variant_i] += (flat ** 2).sum(axis=0)

    mean = s / n
    var = ss / n - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm_v2(window, mean, std, k):
    n_var = mean.shape[0]
    out = window.copy()
    for v in range(n_var):
        for i in range(k):
            out[..., v * k + i] = (window[..., v * k + i] - mean[v]) / std[v]
    return out.astype(np.float16)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feat_f16, heat_f16, file_id, frame_idx, people_count):
    features = {
        "feature": _bytes_feature(feat_f16.tobytes()),
        "heatmap": _bytes_feature(heat_f16.astype(np.float16).tobytes()),
        "file_id": _bytes_feature(file_id.encode("utf-8")),
        "frame_idx": _int64_feature(int(frame_idx)),
        "people_count": _int64_feature(int(people_count)),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def write_split(data_dir, files, split_name, out_dir, radar_idx, k,
                 grid_w, grid_h, sigma_m, alpha_slow, alpha_fast, ema_mode, use_log1p,
                 mean, std, frames_per_shard=15000):
    shard_idx = 0
    writer = None
    written_in_shard = 0
    total_written = 0

    def new_writer():
        nonlocal writer, shard_idx, written_in_shard
        if writer is not None:
            writer.close()
        path = os.path.join(out_dir, f"{split_name}_shard_{shard_idx:04d}.tfrecord")
        writer = tf.io.TFRecordWriter(path)
        shard_idx += 1
        written_in_shard = 0

    new_writer()

    for fname in files:
        path = os.path.join(data_dir, fname)
        d = np.load(path)
        cir = d["radar_cir_iq"]
        xy = d["people_xy"]
        mask = d["people_mask"]

        channels = dual_ema_declutter_iq(cir, radar_idx, alpha_slow, alpha_fast,
                                          ema_mode, use_log1p)
        windows, frame_idx = build_windows_multi(channels, k)

        for w_i, t in enumerate(frame_idx):
            feat_norm = apply_norm_v2(windows[w_i], mean, std, k)
            heat = build_gaussian_heatmap(xy[t], mask[t], grid_w, grid_h, sigma_m)
            # AGGIUNTO (26/8): conteggio persone reali nel frame, da
            # people_mask -- stessa fonte di verita' gia' usata per la
            # heatmap target (mask[t]), nessun nuovo dato richiesto.
            people_count = int(mask[t].sum())

            if written_in_shard >= frames_per_shard:
                new_writer()

            ex = serialize_example(feat_norm, heat, fname, int(t), people_count)
            writer.write(ex)
            written_in_shard += 1
            total_written += 1

    if writer is not None:
        writer.close()
    return total_written


def main():
    ap = argparse.ArgumentParser(
        description="Generazione dataset con I/Q separati (non collassati "
                     "in magnitudine)"
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--radars", default="1,2,3,4,5,6")
    ap.add_argument("--k", type=int, default=5, help="Finestra temporale causale")
    ap.add_argument("--grid-w", type=int, default=12)
    ap.add_argument("--grid-h", type=int, default=18)
    ap.add_argument("--sigma-m", type=float, default=0.2,
                     help="Sigma gaussiana target in metri")
    ap.add_argument("--ema-mode", choices=["v2", "v3"], default="v2")
    ap.add_argument("--alpha-slow", type=float, default=0.02)
    ap.add_argument("--alpha-fast", type=float, default=0.15)
    ap.add_argument("--no-log1p", action="store_true", default=False,
                     help="Disattiva la compressione signed-log1p (si affida solo alla "
                          "BatchNormalization gia' presente dopo conv_joint1)")
    ap.add_argument("--frames-per-shard", type=int, default=15000)
    args = ap.parse_args()
    use_log1p = not args.no_log1p

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.split_json) as f:
        split_data = json.load(f)
    split = split_data["split"]
    if "folds" in split:
        raise ValueError("split.json e' in modalita' k-fold. Fornire uno split 'single_split'.")

    radar_idx = [int(r) - 1 for r in args.radars.split(",")]
    radars_1based = [int(r) for r in args.radars.split(",")]

    print("=========================================================")
    print("   DATASET CON I/Q SEPARATI ")
    print("=========================================================")
    print(f"Radar selezionati (1-based): {radars_1based}")
    print(f"Modalita EMA          : {args.ema_mode}")
    print(f"Compressione          : {'signed log1p' if use_log1p else 'nessuna (solo BatchNorm)'}")
    print(f"Alpha Slow / Fast      : {args.alpha_slow} / {args.alpha_fast}")
    print(f"Finestra K             : {args.k}")
    print(f"Griglia Target         : {args.grid_w}x{args.grid_h} (Sigma = {args.sigma_m}m)")
    print(f"N. varianti canale     : 4 (I_slow, Q_slow, I_fast, Q_fast) -- vs 2 dell'originale")
    print(f"Output Directory       : {args.out_dir}")
    print("---------------------------------------------------------")

    print("\n[1/2] Calcolo statistiche di normalizzazione sul set di train...")
    mean, std = compute_train_stats_iq(
        args.data_dir, split["train"], radar_idx,
        args.alpha_slow, args.alpha_fast, args.ema_mode, use_log1p
    )
    print(f"  Mean shape {mean.shape} =\n{mean}")
    print(f"  Std  shape {std.shape} =\n{std}")

    counts = {}
    for split_name in ["train", "val", "test"]:
        files = split[split_name]
        print(f"\n[2/2] Processing '{split_name}' ({len(files)} file)...")
        n = write_split(
            args.data_dir, files, split_name, args.out_dir,
            radar_idx, args.k, args.grid_w, args.grid_h, args.sigma_m,
            args.alpha_slow, args.alpha_fast, args.ema_mode, use_log1p,
            mean, std, args.frames_per_shard
        )
        counts[split_name] = n
        print(f"    -> {n} frame generati per '{split_name}'.")

    n_var = 4
    feature_shape = [len(radar_idx), 120, 3, n_var * args.k]
    grid_shape = [args.grid_h, args.grid_w]

    meta = {
        "radars_1based": radars_1based,
        "k": args.k,
        "grid_w": args.grid_w,
        "grid_h": args.grid_h,
        "cell_w_m": ROOM_W / args.grid_w,
        "cell_h_m": ROOM_H / args.grid_h,
        "sigma_m": args.sigma_m,
        "ema_mode": args.ema_mode,
        "use_log1p": use_log1p,
        "log1p_variant": "signed (sign(x)*log1p(|x|))" if use_log1p else None,
        "alpha_slow": args.alpha_slow,
        "alpha_fast": args.alpha_fast,
        "iq_mode": True,
        "iq_channel_order": ["I_slow", "Q_slow", "I_fast", "Q_fast"],
        "use_abs_ref": False,
        "feature_shape": feature_shape,
        "grid_shape": grid_shape,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "frame_counts": counts,
    }

    meta_path = os.path.join(args.out_dir, "dataset_meta_v2.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCompletato con successo! Metadata salvati in: {meta_path}")


if __name__ == "__main__":
    main()