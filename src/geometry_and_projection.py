"""
geometry_and_projection.py
===========================
Passo 1 della pipeline "backprojection": usa i frame con una sola persona
(distanza radar-persona nota e non ambigua) per calibrare empiricamente la
relazione bin -> distanza per ciascuno dei 6 radar, poi costruisce le
matrici di backprojection fisse (una per radar) che verranno usate come
pesi NON allenabili di un layer Dense nel modello.

Geometria nota dalla scheda del corso (Room Setup & Sensor Configuration):
  Radar 1: (2.4, 0.0)  boresight verso +y (90 deg)
  Radar 2: (4.8, 2.4)  boresight verso -x (180 deg)
  Radar 3: (4.8, 4.8)  boresight verso -x (180 deg)
  Radar 4: (2.4, 7.2)  boresight verso -y (270 deg)
  Radar 5: (0.0, 4.8)  boresight verso +x (0 deg)
  Radar 6: (0.0, 2.4)  boresight verso +x (0 deg)

Uso: python geometry_and_projection.py --data-dir ./dataset_uwb/data --split-json ./output_split/split.json --out-dir ./geometry_out_1218 --grid-w 12 --grid-h 18 --n-bins 120 --fov-deg 120 --sigma-bins 1.5
    python geometry_and_projection.py \
        --data-dir /percorso/ai/file/npz \
        --split-json ./eda_out/split.json \
        --out-dir ./geometry_out \
        --grid-w 12 --grid-h 8 \
        --n-bins 120 \
        --fov-deg 120 --sigma-bins 1.5

Output in --out-dir:
    range_calibration.json     (a,b fisso E stimato per radar, + confronto residui)
    calibration_scatter.png    (bin vs distanza osservata, con entrambe le rette)
    projection_matrices.npz    (una matrice (grid_h*grid_w, n_bins) per radar)
"""

import argparse
import json
import os

import numpy as np

ROOM_W, ROOM_H = 4.8, 7.2

RADAR_GEOMETRY = {
    1: {"pos": (2.4, 0.0), "boresight_deg": 90},
    2: {"pos": (4.8, 2.4), "boresight_deg": 180},
    3: {"pos": (4.8, 4.8), "boresight_deg": 180},
    4: {"pos": (2.4, 7.2), "boresight_deg": 270},
    5: {"pos": (0.0, 4.8), "boresight_deg": 0},
    6: {"pos": (0.0, 2.4), "boresight_deg": 0},
}


# --------------------------------------------------------------------------- #
# Declutter leggero 
# --------------------------------------------------------------------------- #

def ema_declutter_magnitude(cir, radar_idx_0based, alpha=0.05):
    sel = cir[:, radar_idx_0based]
    y = np.empty_like(sel)
    y[0] = sel[0]
    for t in range(1, sel.shape[0]):
        y[t] = alpha * sel[t] + (1 - alpha) * y[t - 1]
    decl = sel - y
    return np.sqrt(decl[..., 0] ** 2 + decl[..., 1] ** 2)   # (T,R,3,120)


# --------------------------------------------------------------------------- #
# Calibrazione bin -> distanza
# --------------------------------------------------------------------------- #

def calibrate_range(data_dir, files, radars_1based, n_bins=120,
                     min_samples=200, outlier_std=2.5,
                     guard_bins=12, max_bins=50, smooth_window=5, snr_thresh=6.0):
    """
    Per ciascun radar: colleziona coppie (bin_di_picco, distanza_vera) dai
    frame con ESATTAMENTE una persona. Fit lineare
    distanza = a*bin + b, con un giro di rimozione outlier.

    Accorgimenti anti-rumore (importanti: l'argmax su un singolo frame e'
    troppo sensibile a rumore/multipath/clutter residuo):
      - guard_bins: esclude i primi bin (self-interference/accoppiamento
        vicino all'antenna, tipico delle UWB).
      - smooth_window: media la magnitudine su una piccola finestra
        temporale centrata sul frame (la persona si muove poco in 200ms),
        aumenta il rapporto segnale/rumore prima di cercare il picco.
      - snr_thresh: scarta il campione se il picco non emerge chiaramente
        dal rumore di fondo (picco < snr_thresh * mediana del profilo).
      - raffinamento sub-bin per interpolazione parabolica attorno al
        picco, per un target di regressione piu' preciso del solo intero.
    """
    samples = {r: {"bins": [], "dists": []} for r in radars_1based}
    radar_idx_0based = [r - 1 for r in radars_1based]
    half_w = smooth_window // 2

    for fname in files:
        path = os.path.join(data_dir, fname)
        d = np.load(path)
        cir = d["radar_cir_iq"]
        xy = d["people_xy"]
        mask = d["people_mask"]

        n_people = mask.sum(axis=1)
        single_frames = np.where(n_people == 1)[0]
        if len(single_frames) == 0:
            continue

        mag = ema_declutter_magnitude(cir, radar_idx_0based)   # (T,R,3,120)
        mag_avg_ant = mag.mean(axis=2)                          # (T,R,120)
        T = mag_avg_ant.shape[0]

        for t in single_frames:
            t0, t1 = max(0, t - half_w), min(T, t + half_w + 1)
            if t1 - t0 < 1:
                continue
            profile_smooth = mag_avg_ant[t0:t1].mean(axis=0)   # (R,120), media temporale

            p_idx = np.argmax(mask[t])
            px, py = xy[t, p_idx]

            for ri, r in enumerate(radars_1based):
                prof = profile_smooth[ri].copy()
                prof[:guard_bins] = -np.inf     # self-interference vicino al radar
                prof[max_bins:] = -np.inf       # clutter/muri lontani (oltre max_bins)

                peak_bin = int(np.argmax(prof))
                peak_val = prof[peak_bin]
                noise_floor = np.median(mag_avg_ant[t0:t1, ri][:, guard_bins:max_bins])
                if noise_floor <= 1e-9 or peak_val < snr_thresh * noise_floor:
                    continue   # picco non abbastanza sopra il rumore, scarta il campione

                # raffinamento sub-bin: interpolazione parabolica sui 3 bin attorno al picco.
                # Guardia isfinite: se il picco e' al bordo di guard_bins/max_bins, uno dei tre
                # valori vale -inf (mascherato sopra) e va evitato il calcolo (altrimenti NaN
                # manda in crash il fit successivo).
                if guard_bins < peak_bin < max_bins - 1:
                    y0, y1, y2 = prof[peak_bin - 1], prof[peak_bin], prof[peak_bin + 1]
                    if np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2):
                        denom = (y0 - 2 * y1 + y2)
                        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                        delta = float(np.clip(delta, -0.5, 0.5))
                    else:
                        delta = 0.0
                else:
                    delta = 0.0
                refined_bin = peak_bin + delta

                rx, ry = RADAR_GEOMETRY[r]["pos"]
                dist = float(np.hypot(px - rx, py - ry))
                samples[r]["bins"].append(refined_bin)
                samples[r]["dists"].append(dist)

    results = {}
    for r in radars_1based:
        bins = np.array(samples[r]["bins"], dtype=np.float64)
        dists = np.array(samples[r]["dists"], dtype=np.float64)
        if len(bins) < min_samples:
            raise RuntimeError(
                f"Radar {r}: solo {len(bins)} campioni di calibrazione validi (< {min_samples}) "
                "dopo il filtro SNR. Prova ad abbassare --snr-thresh o aumentare i file di calibrazione.")

        a, b = np.polyfit(bins, dists, 1)
        resid = dists - (a * bins + b)
        keep = np.abs(resid) <= outlier_std * resid.std()
        n_outliers = int((~keep).sum())
        a2, b2 = np.polyfit(bins[keep], dists[keep], 1)
        resid2 = dists[keep] - (a2 * bins[keep] + b2)

        results[r] = {
            "a": float(a2), "b": float(b2),
            "n_samples": int(len(bins)), "n_outliers_removed": n_outliers,
            "resid_std_m": float(resid2.std()),
            "bins_raw": bins.tolist(), "dists_raw": dists.tolist(),
        }
        print(f"Radar {r}: distanza = {a2:.4f}*bin + {b2:.4f}  "
              f"(n={len(bins)} validi su filtro SNR, outlier rimossi={n_outliers}, resid_std={resid2.std():.3f}m)")

    return results


def refine_shared_slope(results, radars_1based, outlier_std=2.5):
    a_values = np.array([results[r]["a"] for r in radars_1based])
    resid_stds = np.array([results[r]["resid_std_m"] for r in radars_1based])
    weights = 1.0 / (resid_stds ** 2 + 1e-6)
    a_shared = float(np.average(a_values, weights=weights))
    print(f"\nPendenza condivisa stimata (media pesata per qualita' fit): a_shared={a_shared:.4f}")

    for r in radars_1based:
        bins = np.array(results[r]["bins_raw"])
        dists = np.array(results[r]["dists_raw"])

        resid0 = dists - a_shared * bins
        b0 = float(np.median(resid0))
        resid1 = dists - (a_shared * bins + b0)
        keep = np.abs(resid1) <= outlier_std * resid1.std()
        b_shared = float(np.median(dists[keep] - a_shared * bins[keep]))
        resid_final = dists[keep] - (a_shared * bins[keep] + b_shared)

        results[r]["a_shared"] = a_shared
        results[r]["b_shared"] = b_shared
        results[r]["resid_std_shared_m"] = float(resid_final.std())
        print(f"Radar {r}: (pendenza condivisa) distanza = {a_shared:.4f}*bin + {b_shared:.4f}  "
              f"resid_std={resid_final.std():.3f}m  (era {results[r]['resid_std_m']:.3f}m con fit individuale)")

    return results, a_shared


def validate_fixed_calibration(results, radars_1based, bin_a, bin_b):
    """
    Confronta il valore FISSO/documentato (bin_a, bin_b, es. 0.15 m/bin dal
    README ufficiale) contro i campioni (bin,distanza) raccolti dalla
    calibrazione empirica: quanto bene il valore ufficiale spiega i dati
    osservati, sullo stesso residuo usato per giudicare il fit empirico.
    E' solo un confronto/validazione per il report.
    """
    print(f"\nValidazione calibrazione fissa (documentata): distanza = {bin_a}*bin + {bin_b}")
    for r in radars_1based:
        bins = np.array(results[r]["bins_raw"])
        dists = np.array(results[r]["dists_raw"])
        resid_fixed = dists - (bin_a * bins + bin_b)
        resid_std_fixed = float(resid_fixed.std())
        results[r]["resid_std_fixed_m"] = resid_std_fixed
        print(f"  Radar {r}: resid_std con valore fisso={resid_std_fixed:.3f}m  "
              f"(vs empirico individuale={results[r]['resid_std_m']:.3f}m)")
    return results


def plot_calibration(results, out_path, bin_a=None, bin_b=None):
    import matplotlib.pyplot as plt
    radars = sorted(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, r in zip(axes.flat, radars):
        bins = np.array(results[r]["bins_raw"])
        dists = np.array(results[r]["dists_raw"])
        ax.scatter(bins, dists, s=3, alpha=0.3)
        a, b = results[r]["a"], results[r]["b"]
        xs = np.linspace(bins.min(), bins.max(), 50)
        ax.plot(xs, a * xs + b, color="red", label=f"empirico: d={a:.3f}b+{b:.3f}")
        if bin_a is not None:
            ax.plot(xs, bin_a * xs + bin_b, color="green", linestyle="--",
                    label=f"documentato: d={bin_a}b+{bin_b}")
        title = f"Radar {r} (resid_std={results[r]['resid_std_m']:.3f}m"
        if "resid_std_fixed_m" in results[r]:
            title += f", fisso={results[r]['resid_std_fixed_m']:.3f}m"
        title += ")"
        ax.set_title(title)
        ax.set_xlabel("bin index"); ax.set_ylabel("distanza vera (m)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
# Costruzione matrici di backprojection
# --------------------------------------------------------------------------- #

def build_projection_matrix(grid_h, grid_w, radar_pos, boresight_deg,
                             a, b, n_bins, fov_deg=150, sigma_bins=1.5):
    """
    Ritorna P: shape (grid_h*grid_w, n_bins).
    Riga = una cella della griglia; colonna = un bin di range.
    Il peso e' una gaussiana centrata sul bin atteso per quella cella dato
    quel radar, azzerata fuori dal campo visivo (FOV) del radar.
    """
    cell_w, cell_h = ROOM_W / grid_w, ROOM_H / grid_h
    bx, by = radar_pos
    boresight_rad = np.deg2rad(boresight_deg)
    half_fov = np.deg2rad(fov_deg) / 2.0
    bins_axis = np.arange(n_bins)

    P = np.zeros((grid_h * grid_w, n_bins), dtype=np.float32)
    idx = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            cx = (gx + 0.5) * cell_w
            cy = (gy + 0.5) * cell_h
            dx, dy = cx - bx, cy - by
            dist = float(np.hypot(dx, dy))
            angle = np.arctan2(dy, dx)
            ang_diff = np.arctan2(np.sin(angle - boresight_rad), np.cos(angle - boresight_rad))

            if abs(ang_diff) <= half_fov and dist > 0:
                bin_center = (dist - b) / a
                w = np.exp(-0.5 * ((bins_axis - bin_center) / sigma_bins) ** 2)
                s = w.sum()
                if s > 1e-9:
                    P[idx] = (w / s).astype(np.float32)
            idx += 1
    return P


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--radars", default="1,2,3,4,5,6")
    ap.add_argument("--grid-w", type=int, default=12)
    ap.add_argument("--grid-h", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=120)
    ap.add_argument("--fov-deg", type=float, default=120)
    ap.add_argument("--bin-a", type=float, default=0.15,
                     help="Risoluzione documentata: 1 bin = 0.15 m")
    ap.add_argument("--bin-b", type=float, default=0.0,
                     help="Offset documentato: distanza = bin_a*bin + bin_b (0.15*bin, nessun offset)")
    ap.add_argument("--calibration-mode", choices=["fixed", "empirical"], default="empirical",
                     help="'empirical' (default, ripristinato): usa la regressione per-radar, "
                          "validata su dati reali con F1 nettamente migliore. 'fixed': usa "
                          "--bin-a/--bin-b (documentati) -- VERIFICATO PEGGIORE su questo dataset "
                          "(resid_std 26-76%% piu' alto su tutti e 6 i radar), non consigliato "
                          "come default se non per confronto esplicito nel report.")
    ap.add_argument("--sigma-bins", type=float, default=1.5,
                     help="Larghezza (in bin) del kernel gaussiano di backprojection: "
                          "assorbe l'incertezza della calibrazione")
    ap.add_argument("--calib-split", default="train",
                     help="Split da usare per la calibrazione (train, per non fare leakage su val/test)")
    ap.add_argument("--guard-bins", type=int, default=12,
                     help="Bin iniziali esclusi dalla ricerca del picco (self-interference vicino al radar)")
    ap.add_argument("--max-bins", type=int, default=40,
                     help="Bin oltre il quale ignorare i picchi (clutter/muri lontani). "
                          "CONFERMATO EMPIRICAMENTE (confronto reale 40 vs 50 vs 100 su dati veri): "
                          "40 e' il migliore. Un valore piu' alto (100, provato) peggiora "
                          "drasticamente tutti i radar (es. Radar1 resid_std 0.13m->1.03m) perche' "
                          "oltre bin ~50 c'e' vero rumore/multipath dai muri, non solo segnale "
                          "censurato")
    ap.add_argument("--smooth-window", type=int, default=5,
                     help="Finestra temporale (in frame) su cui mediare il profilo prima del picco")
    ap.add_argument("--snr-thresh", type=float, default=6.0,
                     help="Rapporto minimo picco/rumore per accettare un campione di calibrazione")
    ap.add_argument("--shared-slope", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    radars_1based = [int(r) for r in args.radars.split(",")]

    with open(args.split_json) as f:
        split_data = json.load(f)
    split = split_data["split"]
    if "folds" in split:
        raise ValueError("split.json in modalita' k-fold: passare uno split 'single_split'.")
    calib_files = split[args.calib_split]

    print(f"Calibrazione su {len(calib_files)} file ({args.calib_split}), radar {radars_1based}...")

    results = calibrate_range(args.data_dir, calib_files, radars_1based, args.n_bins,
                               guard_bins=args.guard_bins, max_bins=args.max_bins,
                               smooth_window=args.smooth_window,
                               snr_thresh=args.snr_thresh)

    a_shared = None
    if args.shared_slope:
        results, a_shared = refine_shared_slope(results, radars_1based)

    # Validazione incrociata: quanto bene il valore FISSO documentato (0.15 m/bin
    # di default) spiega i dati osservati -- indipendente da quale modalita' si
    # user poi per costruire le matrici, e' un buon numero da riportare nel report.
    results = validate_fixed_calibration(results, radars_1based, args.bin_a, args.bin_b)

    plot_path = os.path.join(args.out_dir, "calibration_scatter.png")
    plot_calibration(results, plot_path, args.bin_a, args.bin_b)
    print(f"Grafico di verifica salvato in {plot_path}")

    print(f"\n>>> Modalita' di calibrazione per le matrici di backprojection: {args.calibration_mode}")
    if args.calibration_mode == "fixed":
        print(f">>> Uso il valore documentato per TUTTI i radar: a={args.bin_a}, b={args.bin_b}")

    calib_out = {str(r): {"a_empirical": results[r]["a"], "b_empirical": results[r]["b"],
                           "resid_std_empirical_m": results[r]["resid_std_m"],
                           "resid_std_fixed_m": results[r]["resid_std_fixed_m"],
                           "n_samples": results[r]["n_samples"],
                           "n_outliers_removed": results[r]["n_outliers_removed"],
                           **({"a_shared": results[r]["a_shared"], "b_shared": results[r]["b_shared"],
                               "resid_std_shared_m": results[r]["resid_std_shared_m"]}
                              if args.shared_slope else {})}
                 for r in radars_1based}
    with open(os.path.join(args.out_dir, "range_calibration.json"), "w") as f:
        json.dump({
            "radars": radars_1based,
            "n_bins": args.n_bins,
            "calibration_mode_used_for_projection": args.calibration_mode,
            "bin_a_fixed": args.bin_a, "bin_b_fixed": args.bin_b,
            "shared_slope_mode": args.shared_slope,
            "calibration": calib_out,
            "note": "distanza_metri = a * bin_index + b. 'fixed' = valore documentato README "
                    "(usato per le matrici se calibration_mode=fixed); 'empirical' = regressione "
                    "per-radar (usata se calibration_mode=empirical, altrimenti solo diagnostica).",
        }, f, indent=2)

    print("\nCostruzione matrici di backprojection...")
    proj_matrices = {}
    for r in radars_1based:
        if args.calibration_mode == "fixed":
            a, b = args.bin_a, args.bin_b
        elif args.shared_slope:
            a, b = results[r]["a_shared"], results[r]["b_shared"]
        else:
            a, b = results[r]["a"], results[r]["b"]
        P = build_projection_matrix(
            args.grid_h, args.grid_w, RADAR_GEOMETRY[r]["pos"], RADAR_GEOMETRY[r]["boresight_deg"],
            a, b, args.n_bins, args.fov_deg, args.sigma_bins)
        proj_matrices[f"radar_{r}"] = P
        n_nonzero_rows = int((P.sum(axis=1) > 0).sum())
        print(f"  Radar {r}: matrice {P.shape}, {n_nonzero_rows}/{P.shape[0]} celle nel FOV")

    proj_path = os.path.join(args.out_dir, "projection_matrices.npz")
    np.savez(proj_path, **proj_matrices)
    total_weights = sum(p.size for p in proj_matrices.values())
    print(f"\nMatrici salvate in {proj_path}")
    print(f"Totale pesi di backprojection: {total_weights} "
          f"({total_weights * 4 / 1024:.1f} KB float32, ~{total_weights / 1024:.1f} KB INT8)")

    meta = {
        "radars": radars_1based,
        "grid_w": args.grid_w, "grid_h": args.grid_h,
        "room_w": ROOM_W, "room_h": ROOM_H,
        "n_bins": args.n_bins, "fov_deg": args.fov_deg, "sigma_bins": args.sigma_bins,
        "calibration_mode": args.calibration_mode,
        "bin_a_fixed": args.bin_a, "bin_b_fixed": args.bin_b,
        "radar_geometry": {str(r): RADAR_GEOMETRY[r] for r in radars_1based},
    }
    with open(os.path.join(args.out_dir, "projection_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFatto. Output in {args.out_dir}")


if __name__ == "__main__":
    main()