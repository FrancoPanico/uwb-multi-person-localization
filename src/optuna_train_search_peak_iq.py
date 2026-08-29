"""optuna_train_search_peak_iq.py
==================================
Uso:
    python optuna_train_search_peak_iq.py --dataset-dir ./dataset_k5_m2_iq --geometry-dir ./geometry_out_cleansplit --out-dir ./optuna_peak_f1_trials_iq --epochs 30 --n-trials 20

"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import optuna
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Errore configurazione GPU: {e}")

from evaluate_v2 import decode_heatmap, match_frame
from generate_dataset_iq_experiment import build_windows_multi, dual_ema_declutter_iq
from tracking import HeatmapTracker

FIXED_TRACKING_PARAMS = dict(
    t_low=0.29,
    t_high=0.43,
    nms_rad=0.70,
    max_dist=1.8,
    max_missed=9,
    min_hits=8,
    proc_pos=0.020973223078315652,
    proc_vel=0.0016672601244298502,
    meas_noise=0.3,
    center_radius=0.6,
    center_boost=0.04,
)


def apply_norm_v2(window, mean, std, k):
    """Identica all'originale -- gia' generica in n_var = mean.shape[0]."""
    n_var = mean.shape[0]
    out = window.copy()
    for v in range(n_var):
        for i in range(k):
            out[..., v * k + i] = (window[..., v * k + i] - mean[v]) / std[v]
    return out.astype(np.float16)


def run_file_inference_iq(
    model, cir, radar_idx, radars_1based, k, mean, std,
    ema_mode="v2", alpha_slow=0.02, alpha_fast=0.15, use_log1p=True,
    batch_size=256,
):
    channels = dual_ema_declutter_iq(cir, radar_idx, alpha_slow, alpha_fast, ema_mode, use_log1p)
    windows, frame_idx = build_windows_multi(channels, k)
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


def evaluate_model_fixed_tracking(model_path, dataset_meta_path, split_json_path, data_dir, split="val"):
    """Valuta il modello IQ con i parametri di tracking/decodifica fissi
    (FIXED_TRACKING_PARAMS in cima al file)."""
    with open(dataset_meta_path) as f:
        meta = json.load(f)
    with open(split_json_path) as f:
        split_data = json.load(f)

    files = split_data["split"][split]
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

    model = tf.keras.models.load_model(model_path, compile=False)

    tot_tp = tot_fp = tot_fn = 0

    p = FIXED_TRACKING_PARAMS
    tracker = HeatmapTracker(
        max_tracks=4,
        max_match_dist=p["max_dist"],
        max_missed=p["max_missed"],
        min_hits_to_report=p["min_hits"],
        thresh_high=p["t_high"],
        process_noise_pos=p["proc_pos"],
        process_noise_vel=p["proc_vel"],
        meas_noise=p["meas_noise"],
    )

    try:
        for fname in files:
            path = os.path.join(data_dir, fname)
            d = np.load(path)
            cir, xy, mask = d["radar_cir_iq"], d["people_xy"], d["people_mask"]

            frame_idx, heat_all = run_file_inference_iq(
                model, cir, radar_idx, radars_1based, k, mean, std,
                ema_mode, alpha_slow, alpha_fast, use_log1p, batch_size=256,
            )

            for i, t in enumerate(frame_idx):
                gt_pts = xy[t][mask[t].astype(bool)].tolist()
                raw_pts = decode_heatmap(
                    heat_all[i], grid_w, grid_h, thresh=p["t_low"],
                    max_people=4, nms_radius=p["nms_rad"], refine=True,
                    center_radius=p["center_radius"], center_thresh_boost=p["center_boost"]
                )
                tracked_pts = tracker.update(raw_pts)
                tp, fp, fn = match_frame(tracked_pts, gt_pts, thresh=1.0)
                tot_tp += tp
                tot_fp += fp
                tot_fn += fn

        prec = tot_tp / (tot_tp + tot_fp + 1e-9)
        rec = tot_tp / (tot_tp + tot_fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
    finally:
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return f1


def plot_and_save_curves(study, out_dir):
    fig_path = os.path.join(out_dir, "training_curves.png")
    completed_trials = [t for t in study.trials if t.value is not None]
    if not completed_trials:
        return

    trial_numbers = [t.number for t in completed_trials]
    values = [t.value for t in completed_trials]

    plt.figure(figsize=(10, 5))
    plt.plot(trial_numbers, values, marker='o', linestyle='-', color='b', label='Peak F1-Score (IQ)')
    best_values = np.maximum.accumulate(values)
    plt.plot(trial_numbers, best_values, linestyle='--', color='r', label='Miglior F1-Score')

    plt.title("Ottimizzazione Iperparametri Optuna -- Esperimento I/Q (Peak F1)")
    plt.xlabel("Numero Trial")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


BEST_REPORT_FILENAME = "best_peak_f1_params_iq.json"


def load_best_so_far(out_dir):
    report_path = os.path.join(out_dir, BEST_REPORT_FILENAME)
    if os.path.exists(report_path):
        with open(report_path) as f:
            data = json.load(f)
        return {"f1": data["best_f1"], "trial_number": data["best_trial_number"], "params": data["params"]}
    return {"f1": -1.0, "trial_number": None, "params": None}


def save_if_better(f1, trial_number, params, trial_dir, out_dir, best_so_far):
    if f1 <= best_so_far["f1"]:
        return
    best_so_far["f1"] = f1
    best_so_far["trial_number"] = trial_number
    best_so_far["params"] = params

    winning_model_src = os.path.join(trial_dir, "best_model.keras")
    winning_model_dst = os.path.join(out_dir, "best_model_overall.keras")
    if os.path.exists(winning_model_src):
        shutil.copy(winning_model_src, winning_model_dst)

    report_path = os.path.join(out_dir, BEST_REPORT_FILENAME)
    with open(report_path, "w") as f:
        json.dump({
            "best_f1": f1,
            "best_trial_number": trial_number,
            "params": params,
        }, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./dataset_k5_m2_iq")
    parser.add_argument("--data-dir", default="./dataset_uwb/data")
    parser.add_argument("--split-json", default="./eda_out/split.json")
    parser.add_argument("--geometry-dir", default="./geometry_out_cleansplit")
    parser.add_argument("--out-dir", default="./optuna_peak_f1_trials_iq")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.dataset_dir, "dataset_meta_v2.json")

    best_so_far = load_best_so_far(args.out_dir)
    if best_so_far["trial_number"] is not None:
        print(
            f"[RESUME] Trovato un run precedente in questo out-dir: miglior F1 "
            f"finora = {best_so_far['f1']:.4f} (trial {best_so_far['trial_number']})"
        )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 8e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.20, step=0.05)
        focal_beta = trial.suggest_float("focal_beta", 2.0, 4.0, step=0.5)
        width = trial.suggest_categorical("width", [1.0, 1.25, 1.5])
        refine_width = trial.suggest_categorical("refine_width", [16, 24])

        trial_dir = os.path.join(args.out_dir, f"trial_{trial.number:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        cmd = [
            sys.executable, "train_v2.py",
            "--dataset-dir", args.dataset_dir,
            "--geometry-dir", args.geometry_dir,
            "--out-dir", trial_dir,
            "--epochs", str(args.epochs),
            "--batch-size", "64",
            "--lr", f"{lr:.6f}",
            "--dropout", f"{dropout:.2f}",
            "--focal-beta", str(focal_beta),
            "--width", str(width),
            "--refine-width", str(refine_width),
        ]

        print(f"\n================ AVVIO TRIAL {trial.number:03d} (IQ) ================")
        print("Parametri Addestramento:", trial.params)

        out_log_path = os.path.join(trial_dir, "stdout.log")
        err_log_path = os.path.join(trial_dir, "stderr.log")

        out_f = open(out_log_path, "w")
        err_f = open(err_log_path, "w")

        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
        log_file = os.path.join(trial_dir, "training_log.csv")
        reported_epochs = set()

        try:
            while proc.poll() is None:
                time.sleep(3)
                if os.path.exists(log_file):
                    try:
                        df = pd.read_csv(log_file)
                        for _, row in df.iterrows():
                            ep = int(row["epoch"])
                            if ep not in reported_epochs:
                                reported_epochs.add(ep)
                                val_loss = float(row["val_loss"])

                                trial.report(-val_loss, step=ep)
                                if trial.should_prune():
                                    proc.kill()
                                    proc.wait()
                                    print(f"--> TRIAL {trial.number:03d} PRUNATO all'epoca {ep} (Val Loss: {val_loss:.4f})")
                                    raise optuna.TrialPruned(f"Prunato all'epoca {ep}")
                    except optuna.TrialPruned:
                        raise
                    except Exception:
                        continue
        finally:
            out_f.close()
            err_f.close()

        if proc.returncode != 0 and not trial.should_prune():
            raise optuna.TrialPruned("Addestramento fallito. Riferirsi a stderr.log.")

        best_model_path = os.path.join(trial_dir, "best_model.keras")
        f1_score = evaluate_model_fixed_tracking(
            best_model_path, meta_path, args.split_json, args.data_dir, split="val"
        )

        print(f"--> Trial {trial.number:03d} Completato | F1-Score (IQ, tracker fisso): {f1_score:.4f}")

        plot_and_save_curves(trial.study, args.out_dir)
        save_if_better(f1_score, trial.number, trial.params, trial_dir, args.out_dir, best_so_far)

        return f1_score

    storage_path = os.path.join(args.out_dir, "optuna_study_iq.db")
    storage = f"sqlite:///{storage_path}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name="model_search_iq", direction="maximize", pruner=pruner,
        storage=storage, load_if_exists=True,
    )

    n_already_done = len([t for t in study.trials if t.value is not None])
    if n_already_done > 0:
        print(f"[RESUME] {n_already_done} trial gia' completati in precedenza, ripresi da {storage_path}")

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\nRicerca interrotta manualmente dall'utente.")

    if best_so_far["trial_number"] is not None:
        print("\n" + "=" * 65)
        print("SEARCH OPTUNA MODEL PEAK F1 (ESPERIMENTO IQ) COMPLETATO")
        print("=" * 65)
        print(f"Miglior F1-Score ottenuto: {best_so_far['f1']:.4f} (Trial {best_so_far['trial_number']:03d})")
        print(f"Miglior modello salvato in: {os.path.join(args.out_dir, 'best_model_overall.keras')}")
        print(f"Report parametri salvato in: {os.path.join(args.out_dir, 'best_peak_f1_params_iq.json')}")
        print("\nMigliori iperparametri:")
        for k, v in best_so_far["params"].items():
            print(f"  {k:<20}: {v}")
        print("=" * 65)
    else:
        print("\nNessun trial completato con successo -- nessun best-so-far da riportare.")

    plot_and_save_curves(study, args.out_dir)


if __name__ == "__main__":
    main()