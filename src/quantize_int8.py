"""quantize_int8.py
=====================

Uso:
    python quantize_int8.py --model-path ./train_v2_out_cleansplit_confirm/best_model.keras --dataset-dir ./dataset_k5_m2_cleansplit --out ./model_cleansplit_int8.tflite

Poi:
    python evaluate_constraint.py --model-path ./model_cleansplit_int8.tflite
"""

import argparse
import json
import os

import tensorflow as tf

from train_v2 import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", default="./model_int8.tflite")
    ap.add_argument("--n-calib-samples", type=int, default=200,
                     help="Numero di sample reali (batch_size=1) usati per calibrare i range INT8")
    args = ap.parse_args()

    with open(os.path.join(args.dataset_dir, "dataset_meta_v2.json")) as f:
        meta = json.load(f)
    radars_1based = meta["radars_1based"]
    R, n_bins, n_ant, ch = meta["feature_shape"]
    feature_shape_full = (R, n_bins, n_ant, ch)
    grid_shape = tuple(meta["grid_shape"])

    print(f"Carico modello da {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    calib_ds = build_dataset(
        os.path.join(args.dataset_dir, "train_shard_*.tfrecord"),
        feature_shape_full,
        grid_shape,
        radars_1based,
        batch_size=1,
        shuffle=True,
        augment_symmetry=False,
    )

    def representative_dataset():
        count = 0
        for inputs, _ in calib_ds:
            yield [inputs[f"radar_{r}"].numpy() for r in radars_1based]
            count += 1
            if count >= args.n_calib_samples:
                break

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"Calibrazione INT8 su {args.n_calib_samples} sample reali dal train set...")
    tflite_model = converter.convert()

    with open(args.out, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nModello INT8 salvato in: {args.out}")
    print(f"Dimensione file: {size_kb:.1f} KB  (limite Flash: 800 KB)")
    print("\nOra esegui:")
    print(f"    python evaluate_constraint.py --model-path {args.out}")


if __name__ == "__main__":
    main()