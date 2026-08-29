"""train_v2.py

Uso:
    python train_v2.py \
        --dataset-dir ./dataset_v2_tfrecord \
        --geometry-dir ./geometry_out \
        --out-dir ./train_v2_out \
        --epochs 40 --batch-size 64 --lr 3e-4 --augment-symmetry --center-boost 2.0
"""

import argparse
import csv
import json
import os

from model_v2 import build_model_v2
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Data augmentation per simmetria geometrica della stanza (opzionale)
# --------------------------------------------------------------------------- #
_SYMMETRY_MAPS = [
    {r: r for r in range(1, 7)},  # identita'
    {1: 1, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2},  # specchiatura sinistra-destra
    {1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 5},  # specchiatura alto-basso
    {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3},  # rotazione 180 gradi
]

_ANTENNA_FLIP = [False, True, True, False]


def compute_symmetry_permutations(radars_1based):
  missing = [r for r in range(1, 7) if r not in radars_1based]
  if missing:
    raise ValueError(
        f"--augment-symmetry richiede tutti e 6 i radar; mancanti: {missing}"
    )

  perms = []
  for mapping in _SYMMETRY_MAPS:
    perm = [radars_1based.index(mapping[r]) for r in radars_1based]
    perms.append(perm)
  return perms


def apply_symmetry_augmentation(feat, heat, perms_tensor):
  choice = tf.random.uniform([], maxval=4, dtype=tf.int32)
  perm = perms_tensor[choice]
  feat_aug = tf.gather(feat, perm, axis=0)  # permuta i blocchi radar

  # feat ha shape (R, n_bins, n_antennas, ch) -- asse antenne = axis 2.
  # Una singola specchiatura (scelte 1 e 2) inverte la chiralita'
  # dell'array di antenne perpendicolare al boresight; la rotazione di 180
  # gradi (scelta 3) e composizione di due specchiature, quindi la
  # preserva. Vedi docstring in cima al file per la derivazione.
  feat_aug = tf.switch_case(
      choice,
      branch_fns={
          0: lambda: feat_aug,
          1: lambda: tf.reverse(feat_aug, axis=[2]),
          2: lambda: tf.reverse(feat_aug, axis=[2]),
          3: lambda: feat_aug,
      },
  )

  heat_aug = tf.switch_case(
      choice,
      branch_fns={
          0: lambda: heat,
          1: lambda: tf.reverse(
              heat, axis=[1]
          ),  # specchiatura sinistra-destra: flip X
          2: lambda: tf.reverse(
              heat, axis=[0]
          ),  # specchiatura alto-basso: flip Y
          3: lambda: tf.reverse(heat, axis=[0, 1]),  # rotazione 180
      },
  )
  return feat_aug, heat_aug


# --------------------------------------------------------------------------- #
# Dataset: parsing + split in R input separati
# --------------------------------------------------------------------------- #


def make_parse_fn(feature_shape, grid_shape):
  spec = {
      "feature": tf.io.FixedLenFeature([], tf.string),
      "heatmap": tf.io.FixedLenFeature([], tf.string),
      "file_id": tf.io.FixedLenFeature([], tf.string),
      "frame_idx": tf.io.FixedLenFeature([], tf.int64),
  }

  def _parse(example_proto):
    p = tf.io.parse_single_example(example_proto, spec)
    feat = tf.io.decode_raw(p["feature"], tf.float16)
    feat = tf.reshape(tf.cast(feat, tf.float32), feature_shape)
    heat = tf.io.decode_raw(p["heatmap"], tf.float16)
    heat = tf.reshape(tf.cast(heat, tf.float32), grid_shape)
    return feat, heat

  return _parse


def build_dataset(
    shard_glob,
    feature_shape,
    grid_shape,
    radars_1based,
    batch_size,
    shuffle=False,
    shuffle_buffer=8192,
    augment_symmetry=False,
):
  files = tf.io.gfile.glob(shard_glob)
  if not files:
    raise FileNotFoundError(
        f"Nessuno shard trovato per il pattern: {shard_glob}"
    )
  ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
  if shuffle:
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
  ds = ds.map(
      make_parse_fn(feature_shape, grid_shape),
      num_parallel_calls=tf.data.AUTOTUNE,
  )

  if augment_symmetry:
    perms = compute_symmetry_permutations(radars_1based)
    perms_tensor = tf.constant(perms, dtype=tf.int32)

    def _augment(feat, heat):
      return apply_symmetry_augmentation(feat, heat, perms_tensor)

    ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

  def to_multi_input(feat, heat):
    radar_tensors = tf.unstack(feat, axis=0)
    inputs = {f"radar_{r}": t for r, t in zip(radars_1based, radar_tensors)}
    return inputs, heat

  ds = ds.map(to_multi_input, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


# --------------------------------------------------------------------------- #
# Loss: Penalty-Reduced Focal Loss con Spatially-Weighted Center Penalty
# --------------------------------------------------------------------------- #


def centernet_focal_loss_from_logits(
    y_true, logits, alpha=2.0, beta=4.0, center_boost=2.0
):
  """Penalty-reduced focal loss con penalizzazione maggiorata per Falsi Positivi
  nella zona centrale della stanza (soppressione del bias spaziale da
  backprojection).
  """
  p = tf.sigmoid(logits)
  log_p = tf.math.log_sigmoid(logits)
  log_1_minus_p = tf.math.log_sigmoid(-logits)

  pos_mask = tf.cast(y_true >= 0.999, tf.float32)
  neg_mask = 1.0 - pos_mask

  # Matrice di peso spaziale: picco di penalita' al centro della stanza
  grid_h = tf.shape(y_true)[1]
  grid_w = tf.shape(y_true)[2]
  ys = tf.cast(tf.range(0, grid_h), tf.float32) + 0.5
  xs = tf.cast(tf.range(0, grid_w), tf.float32) + 0.5
  gy, gx = tf.meshgrid(ys, xs, indexing="ij")

  cy = tf.cast(grid_h, tf.float32) / 2.0
  cx = tf.cast(grid_w, tf.float32) / 2.0
  dist_sq = ((gx - cx) / cx) ** 2 + ((gy - cy) / cy) ** 2

  # Incrementa il peso del background fino a (1 + center_boost) al centro
  spatial_weights = 1.0 + center_boost * tf.exp(-dist_sq / 0.5)

  pos_loss = -tf.pow(1.0 - p, alpha) * log_p * pos_mask
  neg_loss = (
      -tf.pow(1.0 - y_true, beta)
      * tf.pow(p, alpha)
      * log_1_minus_p
      * neg_mask
      * spatial_weights
  )

  num_pos = tf.reduce_sum(pos_mask)
  total_loss = tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)
  loss = total_loss / tf.maximum(num_pos, 1.0)
  return loss


def heatmap_metrics(y_true, y_pred, thresh=0.5):
  pred_bin = tf.cast(y_pred > thresh, tf.float32)
  true_bin = tf.cast(y_true > thresh, tf.float32)
  tp = tf.reduce_sum(pred_bin * true_bin)
  fp = tf.reduce_sum(pred_bin * (1 - true_bin))
  fn = tf.reduce_sum((1 - pred_bin) * true_bin)
  precision = tp / (tp + fp + 1e-6)
  recall = tp / (tp + fn + 1e-6)
  return precision, recall


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset-dir", required=True)
  ap.add_argument("--geometry-dir", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--epochs", type=int, default=40)
  ap.add_argument("--batch-size", type=int, default=64)
  ap.add_argument("--lr", type=float, default=3e-4)
  ap.add_argument("--clipnorm", type=float, default=1.0)
  ap.add_argument("--focal-alpha", type=float, default=2.0)
  ap.add_argument("--focal-beta", type=float, default=4.0)
  ap.add_argument(
      "--center-boost",
      type=float,
      default=2.0,
      help=(
          "Moltiplicatore di penalizzazione per i falsi positivi al centro della"
          " stanza"
      ),
  )
  ap.add_argument("--width", type=float, default=1.0)
  ap.add_argument("--refine-width", type=int, default=16)
  ap.add_argument("--dilation-rate", type=int, default=1)
  ap.add_argument("--dropout", type=float, default=0.0)
  ap.add_argument("--bn-momentum", type=float, default=0.99)
  ap.add_argument("--patience", type=int, default=6)
  ap.add_argument(
      "--augment-symmetry",
      action="store_true",
      help="Attiva la data augmentation per simmetria geometrica dei 6 radar",
  )
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)

  with open(os.path.join(args.dataset_dir, "dataset_meta_v2.json")) as f:
    meta = json.load(f)
  radars_1based = meta["radars_1based"]
  R, n_bins, n_ant, ch = meta["feature_shape"]
  feature_shape_full = (R, n_bins, n_ant, ch)
  feature_shape_per_radar = (n_bins, n_ant, ch)
  grid_shape = tuple(meta["grid_shape"])
  print(
      f"radars={radars_1based},"
      f" feature_shape_per_radar={feature_shape_per_radar},"
      f" grid_shape={grid_shape}"
  )

  proj_npz = np.load(
      os.path.join(args.geometry_dir, "projection_matrices.npz")
  )
  projection_matrices = {k: proj_npz[k] for k in proj_npz.files}
  missing = [
      f"radar_{r}" for r in radars_1based if f"radar_{r}" not in projection_matrices
  ]
  if missing:
    raise ValueError(
        f"Matrici di backprojection mancanti per: {missing}."
    )

  train_ds = build_dataset(
      os.path.join(args.dataset_dir, "train_shard_*.tfrecord"),
      feature_shape_full,
      grid_shape,
      radars_1based,
      args.batch_size,
      shuffle=True,
      augment_symmetry=args.augment_symmetry,
  )
  val_ds = build_dataset(
      os.path.join(args.dataset_dir, "val_shard_*.tfrecord"),
      feature_shape_full,
      grid_shape,
      radars_1based,
      args.batch_size,
      shuffle=False,
      augment_symmetry=False,
  )

  model = build_model_v2(
      feature_shape_per_radar,
      grid_shape,
      radars_1based,
      projection_matrices,
      width=args.width,
      refine_width=args.refine_width,
      dropout=args.dropout,
      bn_momentum=args.bn_momentum,
      dilation_rate=args.dilation_rate,
  )
  model.summary()

  optimizer = tf.keras.optimizers.Adam(
      learning_rate=args.lr, clipnorm=args.clipnorm
  )

  @tf.function
  def train_step(inputs, heat_true):
    with tf.GradientTape() as tape:
      heat_logits, heat_prob = model(inputs, training=True)
      heat_logits = tf.squeeze(heat_logits, axis=-1)
      heat_prob = tf.squeeze(heat_prob, axis=-1)
      loss = centernet_focal_loss_from_logits(
          heat_true,
          heat_logits,
          args.focal_alpha,
          args.focal_beta,
          args.center_boost,
      )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    precision, recall = heatmap_metrics(heat_true, heat_prob)
    return loss, precision, recall

  @tf.function
  def val_step(inputs, heat_true):
    heat_logits, heat_prob = model(inputs, training=False)
    heat_logits = tf.squeeze(heat_logits, axis=-1)
    heat_prob = tf.squeeze(heat_prob, axis=-1)
    loss = centernet_focal_loss_from_logits(
        heat_true,
        heat_logits,
        args.focal_alpha,
        args.focal_beta,
        args.center_boost,
    )
    precision, recall = heatmap_metrics(heat_true, heat_prob)
    return loss, precision, recall

  log_path = os.path.join(args.out_dir, "training_log.csv")
  with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow([
        "epoch",
        "train_loss",
        "train_prec",
        "train_rec",
        "val_loss",
        "val_prec",
        "val_rec",
    ])

  best_val_loss = float("inf")
  epochs_no_improve = 0
  history = {"train_loss": [], "val_loss": []}

  for epoch in range(1, args.epochs + 1):
    tr = np.zeros(3)
    n_batches = 0
    for inputs, heat_true in train_ds:
      loss, prec, rec = train_step(inputs, heat_true)
      tr += np.array([loss.numpy(), prec.numpy(), rec.numpy()])
      n_batches += 1
    tr /= max(n_batches, 1)

    va = np.zeros(3)
    n_val_batches = 0
    for inputs, heat_true in val_ds:
      loss, prec, rec = val_step(inputs, heat_true)
      va += np.array([loss.numpy(), prec.numpy(), rec.numpy()])
      n_val_batches += 1
    va /= max(n_val_batches, 1)

    print(
        f"Epoch {epoch:03d} | train_loss={tr[0]:.4f} P={tr[1]:.3f}"
        f" R={tr[2]:.3f} | val_loss={va[0]:.4f} P={va[1]:.3f} R={va[2]:.3f}"
    )

    with open(log_path, "a", newline="") as f:
      csv.writer(f).writerow([epoch, *tr, *va])

    history["train_loss"].append(tr[0])
    history["val_loss"].append(va[0])

    if va[0] < best_val_loss:
      best_val_loss = va[0]
      epochs_no_improve = 0
      model.save(os.path.join(args.out_dir, "best_model.keras"))
    else:
      epochs_no_improve += 1
      if epochs_no_improve >= args.patience:
        print(
            f"Early stopping all'epoca {epoch} (nessun miglioramento da"
            f" {args.patience} epoche)."
        )
        break

  model.save(os.path.join(args.out_dir, "final_model.keras"))

  try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=120)
  except Exception as e:
    print(f"Impossibile salvare training_curves.png: {e}")

  history_path = os.path.join(args.out_dir, "history.json")
  with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

  print(f"\nMigliore val_loss: {best_val_loss:.4f}")
  print(f"Modelli salvati in: {args.out_dir}")


if __name__ == "__main__":
  main()