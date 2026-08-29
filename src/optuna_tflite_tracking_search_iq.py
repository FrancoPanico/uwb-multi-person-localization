"""optuna_tflite_tracking_search_iq.py
=======================================
dà al modello IQ una propria ricerca Optuna sui parametri di
decodifica/tracking/maschera anti-fantasma

Uso:
    python optuna_tflite_tracking_search_iq.py --tflite-model ./model_iq_experiment_int8.tflite --data-dir ./dataset_uwb/data --split-json ./eda_out/split.json --dataset-meta ./dataset_k5_m2_iq/dataset_meta_v2.json --split val --n-trials 60 --out-dir ./optuna_tflite_spatial_out_iq
"""

import argparse
import json
import os

from evaluate_v2 import decode_heatmap, match_frame
from generate_dataset_iq_experiment import build_windows_multi, dual_ema_declutter_iq
import numpy as np
import optuna
import tensorflow as tf
from tracking import HeatmapTracker


def quantize_data(x_float, scale, zero_point):
  if scale == 0:
    return x_float.astype(np.int8)
  q = np.round(x_float / scale) + zero_point
  return np.clip(q, -128, 127).astype(np.int8)


def dequantize_data(x_int8, scale, zero_point):
  return (x_int8.astype(np.float32) - zero_point) * scale


def apply_norm_v2(window, mean, std, k):
  n_var = mean.shape[0]
  out = window.copy()
  for v in range(n_var):
    for i in range(k):
      out[..., v * k + i] = (window[..., v * k + i] - mean[v]) / std[v]
  return out.astype(np.float16)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tflite-model", required=True)
  ap.add_argument("--data-dir", default="./dataset_uwb/data")
  ap.add_argument("--split-json", default="./eda_out/split.json")
  ap.add_argument("--dataset-meta", default="./dataset_k5_m2_iq/dataset_meta_v2.json")
  ap.add_argument("--split", default="val")
  ap.add_argument("--n-trials", type=int, default=60)
  ap.add_argument("--out-dir", default="./optuna_tflite_spatial_out_iq")
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)

  with open(args.dataset_meta) as f:
    meta = json.load(f)
  with open(args.split_json) as f:
    split_data = json.load(f)

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

  files = split_data["split"][args.split]

  # 1. Caricamento Modello TFLite
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
  out_scale, out_zp = out_detail["quantization"]

  print(
      f"[I/Q] Pre-calcolo heatmap TFLite INT8 per {len(files)} file in memoria RAM..."
  )
  cached_frames = []
  for fname in files:
    path = os.path.join(args.data_dir, fname)
    d = np.load(path)
    cir, xy, mask = d["radar_cir_iq"], d["people_xy"], d["people_mask"]

    # UNICA differenza rispetto all'originale: pipeline I/Q separata
    # (4 varianti) invece di magnitudine (2 varianti).
    channels = dual_ema_declutter_iq(cir, radar_idx, alpha_slow, alpha_fast, ema_mode, use_log1p)
    windows, frame_idx = build_windows_multi(channels, k)

    for i, t in enumerate(frame_idx):
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

      gt_pts = xy[t][mask[t].astype(bool)].tolist()
      cached_frames.append((heat_prob, gt_pts))

  print(
      f"Caching completato: {len(cached_frames)} frame pronti. Avvio Optuna"
      " Search..."
  )

  def objective(trial):
    t_low = trial.suggest_float("t_low", 0.15, 0.35, step=0.01)
    t_high = trial.suggest_float("t_high", t_low + 0.02, 0.45, step=0.01)
    nms_rad = trial.suggest_float("nms_rad", 0.40, 0.75, step=0.05)
    max_dist = trial.suggest_float("max_dist", 0.8, 2.0, step=0.10)
    max_missed = trial.suggest_int("max_missed", 2, 9)
    min_hits = trial.suggest_int("min_hits", 4, 9)

    proc_pos = trial.suggest_float("proc_pos", 1e-3, 1e-1, log=True)
    proc_vel = trial.suggest_float("proc_vel", 1e-3, 1e-1, log=True)
    meas_noise = trial.suggest_float("meas_noise", 0.1, 0.8, step=0.05)

    # Parametri della Maschera Spaziale Anti-Fantasma
    center_radius = trial.suggest_float("center_radius", 0.30, 1.20, step=0.10)
    center_thresh_boost = trial.suggest_float(
        "center_boost", 0.02, 0.25, step=0.02
    )

    tracker = HeatmapTracker(
        max_tracks=4,
        max_match_dist=max_dist,
        max_missed=max_missed,
        min_hits_to_report=min_hits,
        thresh_high=t_high,
        process_noise_pos=proc_pos,
        process_noise_vel=proc_vel,
        meas_noise=meas_noise,
    )

    tot_tp = tot_fp = tot_fn = 0

    for heat_prob, gt_pts in cached_frames:
      raw_pts = decode_heatmap(
          heat_prob,
          grid_w,
          grid_h,
          thresh=t_low,
          max_people=4,
          nms_radius=nms_rad,
          refine=True,
          center_radius=center_radius,
          center_thresh_boost=center_thresh_boost,
      )

      tracked_pts = tracker.update(raw_pts)
      tp, fp, fn = match_frame(tracked_pts, gt_pts, thresh=1.0)
      tot_tp += tp
      tot_fp += fp
      tot_fn += fn

    prec = tot_tp / (tot_tp + tot_fp + 1e-9)
    rec = tot_tp / (tot_tp + tot_fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)

  study = optuna.create_study(
      study_name="tflite_spatial_tracking_opt_iq", direction="maximize"
  )
  study.optimize(objective, n_trials=args.n_trials)

  report_path = os.path.join(
      args.out_dir, "best_tflite_spatial_tracking_params_iq.json"
  )
  with open(report_path, "w") as f:
    json.dump(
        {"best_f1": study.best_value, "params": study.best_params}, f, indent=4
    )

  print("\n" + "=" * 65)
  print("[I/Q] OTTIMIZZAZIONE TFLite INT8 + MASCHERA SPAZIALE COMPLETATA")
  print("=" * 65)
  print(f"Miglior F1-Score ottenuto: {study.best_value:.4f}")
  print("\nParametri ottimali trovati:")
  for k, v in study.best_params.items():
    print(f"  {k:<22}: {v}")
  print("=" * 65)


if __name__ == "__main__":
  main()