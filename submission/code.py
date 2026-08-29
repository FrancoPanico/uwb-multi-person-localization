"""
code.py

VARIANTE "I/Q SEPARATI" (trial 015)
Carica il modello INT8 quantizzato 'model.tflite' (deve trovarsi nella
STESSA cartella di questo script) ed esegue l'intera pipeline end-to-end
su un file .npy grezzo di shape (T, 6, 3, 120, 2) CIR complesso [I, Q]
per T frame, 6 radar, 3 antenne, 120 bin di range — producendo un file
.jsonl con una riga per frame:
    {"frame": N, "localizations": [[x, y], ...]}

CONFIGURAZIONE:
  - Preprocessing I/Q separati: a differenza del candidato a magnitudine
    precedente (che collassava subito I e Q in sqrt(I^2+Q^2), perdendo
    l'informazione di fase), questa variante mantiene 4 canali separati
    per radar: I_slow, Q_slow, I_fast, Q_fast (compressione "signed
    log1p" f(x)=sign(x)*log1p(|x|), definita anche per valori negativi).
    Dataset: dataset_k5_m2_iq/ (stesso split, sigma_m=0.2, alpha_slow/fast
    identici al candidato a magnitudine).
  - Iperparametri di training: trial 015 della ricerca Optuna dedicata
    (lr=0.000245, dropout=0.10, focal_beta=4.0, width=1.5, refine_width=24
    — Aggiornamento 44), DIVERSI da quelli del candidato a magnitudine.
  - Parametri di decodifica/tracking: ricerca Optuna dedicata per
    trial 015. F1=0.8778 su test a soglia ufficiale 1.0m, +0.0354 rispetto al
    candidato a magnitudine, +0.0241 rispetto al primo candidato IQ.

CLI fissa:
    python code.py --input-path <input.npy> --output-path <output.jsonl>
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Costanti geometria stanza
# --------------------------------------------------------------------------- #
ROOM_W, ROOM_H = 4.8, 7.2

# --------------------------------------------------------------------------- #
# Configurazione dataset/preprocessing FINALE (variante I/Q)
# --------------------------------------------------------------------------- #
RADARS_1BASED = [1, 2, 3, 4, 5, 6]
RADAR_IDX = [r - 1 for r in RADARS_1BASED]
K = 5
GRID_H, GRID_W = 18, 12
EMA_MODE = "v2"
USE_LOG1P = True
ALPHA_SLOW = 0.02
ALPHA_FAST = 0.15

# norm_mean / norm_std: shape (n_var=4, 3 antenne) = (I_slow, Q_slow,
# I_fast, Q_fast) x (ant1, ant2, ant3)
NORM_MEAN = np.array(
    [
        [0.004057994578033686, -0.00018941561575047672, 0.00032585737062618136],
        [-0.003183339023962617, 0.0008569739875383675, 0.0008086703019216657],
        [0.0031685647554695606, 3.869431384373456e-05, 0.0002835431369021535],
        [-0.0025223593693226576, 0.0005853529437445104, 0.0004033131990581751],
    ],
    dtype=np.float32,
)
NORM_STD = np.array(
    [
        [2.698016405105591, 1.993811845779419, 1.5406538248062134],
        [2.6907715797424316, 2.00046968460083, 1.5752630233764648],
        [2.541093349456787, 1.858491063117981, 1.4269143342971802],
        [2.534311056137085, 1.864850640296936, 1.4618616104125977],
    ],
    dtype=np.float32,
)

MAX_PEOPLE = 4

# --------------------------------------------------------------------------- #
# Parametri di decodifica/tracking FINALI (variante I/Q, trial 015)
# (Optuna dedicato per trial 015, F1=0.8778 su test)
# --------------------------------------------------------------------------- #
T_LOW = 0.27
T_HIGH = 0.43
NMS_RAD = 0.70
MAX_DIST = 2.0
MAX_MISSED = 9
MIN_HITS = 9
PROC_POS = 0.028308131373182843
PROC_VEL = 0.0012012262662818589
MEAS_NOISE = 0.70
CENTER_RADIUS = 1.1
CENTER_BOOST = 0.02


# =========================================================================== #
# Preprocessing proveniente da generate_dataset_iq_experiment.py
# =========================================================================== #

def _signed_log1p(x):
    """Compressione log1p 'con segno', definita anche per x negativi:
    f(x) = sign(x) * log1p(|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def dual_ema_declutter_iq(cir, radar_idx, alpha_slow, alpha_fast, ema_mode, use_log1p):
    sel = cir[:, radar_idx].astype(np.float32)  # (T, R, 3, 120, 2)
    T = sel.shape[0]

    decl_slow = np.empty_like(sel)
    decl_fast = np.empty_like(sel)

    if ema_mode != "v2":
        raise ValueError(f"ema_mode non supportato in questa build")

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

    # Differenza rispetto al modello a magnitudine: NON calcoliamo
    # sqrt(I^2+Q^2). Teniamo I (indice 0) e Q (indice 1) separati.
    i_slow = decl_slow[..., 0]  # (T, R, 3, 120)
    q_slow = decl_slow[..., 1]
    i_fast = decl_fast[..., 0]
    q_fast = decl_fast[..., 1]

    if use_log1p:
        i_slow = _signed_log1p(i_slow)
        q_slow = _signed_log1p(q_slow)
        i_fast = _signed_log1p(i_fast)
        q_fast = _signed_log1p(q_fast)

    # Trasposizione per convenzione RangeNet: (T, R, 120, 3)
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
        return (
            np.empty((0, R, B, A, n_var * k), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    out = np.empty((n_win, R, B, A, n_var * k), dtype=np.float32)
    for v, mag in enumerate(channels):
        for i in range(k):
            out[..., v * k + i] = mag[i : i + n_win]
    frame_idx = np.arange(k - 1, T)
    return out, frame_idx


def apply_norm_v2(window, mean, std, k):
    n_var = mean.shape[0]
    out = window.copy()
    for v in range(n_var):
        for i in range(k):
            out[..., v * k + i] = (window[..., v * k + i] - mean[v]) / std[v]
    return out.astype(np.float16)


# =========================================================================== #
# Decodifica heatmap -> punti [x, y, score] proveniente da evaluate_v2.py
# (INVARIATA rispetto al modello a magnitudine: dipende solo dalla
# heatmap di uscita del modello, non dalla rappresentazione di input)
# =========================================================================== #

def decode_heatmap(
    heat,
    grid_w,
    grid_h,
    thresh=0.20,
    max_people=4,
    nms_radius=0.50,
    refine=True,
    center_radius=0.80,
    center_thresh_boost=0.10,
):
    cell_w, cell_h = ROOM_W / grid_w, ROOM_H / grid_h

    center_x = ROOM_W / 2.0
    center_y = ROOM_H / 2.0

    gy_idx, gx_idx = np.indices(heat.shape)
    x_coords = (gx_idx + 0.5) * cell_w
    y_coords = (gy_idx + 0.5) * cell_h

    dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    dynamic_thresh = np.where(
        dist_from_center <= center_radius, thresh + center_thresh_boost, thresh
    )

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


# =========================================================================== #
# Tracking proveniente da tracking.py 
# =========================================================================== #

class KalmanTrack2D:
    def __init__(self, x0, y0, process_noise_pos=0.02, process_noise_vel=0.05, meas_noise=0.35):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([meas_noise**2, meas_noise**2, 1.0, 1.0])
        self.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag(
            [process_noise_pos**2, process_noise_pos**2, process_noise_vel**2, process_noise_vel**2]
        )
        self.R = np.diag([meas_noise**2, meas_noise**2])
        self.missed = 0
        self.hits = 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = np.array(z, dtype=np.float64) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(4) - Kg @ self.H) @ self.P
        self.missed = 0
        self.hits += 1

    @property
    def position(self):
        return [float(self.x[0]), float(self.x[1])]


def greedy_match(track_positions, detections, max_dist):
    candidates = []
    for ti, tp in enumerate(track_positions):
        for di, d in enumerate(detections):
            dist = float(np.hypot(tp[0] - d[0], tp[1] - d[1]))
            if dist <= max_dist:
                candidates.append((dist, ti, di))
    candidates.sort(key=lambda c: c[0])

    matched_t, matched_d = set(), set()
    assignment = []
    for dist, ti, di in candidates:
        if ti in matched_t or di in matched_d:
            continue
        matched_t.add(ti)
        matched_d.add(di)
        assignment.append((ti, di))
    return assignment


class HeatmapTracker:
    def __init__(
        self,
        max_tracks=4,
        max_match_dist=1.40,
        max_missed=4,
        min_hits_to_report=3,
        thresh_high=0.33,
        process_noise_pos=0.0935,
        process_noise_vel=0.0158,
        meas_noise=0.4000,
    ):
        self.max_tracks = max_tracks
        self.max_match_dist = max_match_dist
        self.max_missed = max_missed
        self.min_hits_to_report = min_hits_to_report
        self.thresh_high = thresh_high
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.meas_noise = meas_noise
        self.tracks = []

    def update(self, detections):
        for tr in self.tracks:
            tr.predict()

        track_positions = [tr.position for tr in self.tracks]
        det_coords = [[d[0], d[1]] for d in detections]
        assignment = greedy_match(track_positions, det_coords, self.max_match_dist)

        matched_t = {ti for ti, _ in assignment}
        matched_d = {di for _, di in assignment}

        for ti, di in assignment:
            self.tracks[ti].update(det_coords[di])

        for ti, tr in enumerate(self.tracks):
            if ti not in matched_t:
                tr.missed += 1

        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

        for di, d in enumerate(detections):
            if di in matched_d:
                continue
            score = d[2]
            if score < self.thresh_high:
                continue
            if len(self.tracks) >= self.max_tracks:
                continue
            self.tracks.append(
                KalmanTrack2D(d[0], d[1], self.process_noise_pos, self.process_noise_vel, self.meas_noise)
            )

        reported = [tr.position for tr in self.tracks if tr.hits >= self.min_hits_to_report]
        return reported[: self.max_tracks]


# =========================================================================== #
# Inferenza TFLite INT8 (quantizzazione/dequantizzazione manuale, come in
# evaluate_best_params_iq.py / optuna_tflite_tracking_search_iq.py)
# =========================================================================== #

def quantize_data(x_float, scale, zero_point):
    if scale == 0:
        return x_float.astype(np.int8)
    q = np.round(x_float / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_data(x_int8, scale, zero_point):
    return (x_int8.astype(np.float32) - zero_point) * scale


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True, help="File .npy grezzo (T,6,3,120,2)")
    ap.add_argument("--output-path", required=True, help="File .jsonl di output")
    args = ap.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.tflite")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_map = {}
    for detail in input_details:
        tensor_name = detail["name"]
        for r in RADARS_1BASED:
            if f"radar_{r}" in tensor_name:
                input_map[r] = detail
                break

    out_detail = output_details[0]
    out_scale, out_zp = out_detail["quantization"]

    cir = np.load(args.input_path)  # (T, 6, 3, 120, 2)

    channels = dual_ema_declutter_iq(cir, RADAR_IDX, ALPHA_SLOW, ALPHA_FAST, EMA_MODE, USE_LOG1P)
    windows, frame_idx = build_windows_multi(channels, K)

    tracker = HeatmapTracker(
        max_tracks=MAX_PEOPLE,
        max_match_dist=MAX_DIST,
        max_missed=MAX_MISSED,
        min_hits_to_report=MIN_HITS,
        thresh_high=T_HIGH,
        process_noise_pos=PROC_POS,
        process_noise_vel=PROC_VEL,
        meas_noise=MEAS_NOISE,
    )

    out_lines = []

    for i, t in enumerate(frame_idx):
        window = windows[i]
        normed = apply_norm_v2(window, NORM_MEAN, NORM_STD, K).astype(np.float32)

        for ri, r in enumerate(RADARS_1BASED):
            inp_det = input_map[r]
            inp_scale, inp_zp = inp_det["quantization"]
            radar_feat = normed[ri][None, ...]
            quant_feat = quantize_data(radar_feat, inp_scale, inp_zp)
            interpreter.set_tensor(inp_det["index"], quant_feat)

        interpreter.invoke()
        raw_out = interpreter.get_tensor(out_detail["index"])
        heat_prob = dequantize_data(raw_out, out_scale, out_zp)[0, ..., 0]

        raw_pts = decode_heatmap(
            heat_prob,
            GRID_W,
            GRID_H,
            thresh=T_LOW,
            max_people=MAX_PEOPLE,
            nms_radius=NMS_RAD,
            refine=True,
            center_radius=CENTER_RADIUS,
            center_thresh_boost=CENTER_BOOST,
        )
        tracked_pts = tracker.update(raw_pts)

        localizations = [
            [round(float(np.clip(x, 0.0, ROOM_W)), 4), round(float(np.clip(y, 0.0, ROOM_H)), 4)]
            for x, y in tracked_pts
        ]
        out_lines.append(json.dumps({"frame": int(t), "localizations": localizations}))

    with open(args.output_path, "w") as f:
        for line in out_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()