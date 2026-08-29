"""model_v2_coordconv.py
Variante di model_v2.py: inietta due canali di coordinate
costanti (x, y normalizzate della griglia stanza, range [-1, 1]) in
input al blocco di refine, subito dopo la fusione dei radar -- tecnica
"CoordConv" (Liu, Lehman, Molino, Such, Frank, Sung, Ho, "An Intriguing
Failing of Convolutional Neural Networks and the CoordConv Solution",
NeurIPS 2018).

Motivazione per QUESTO progetto: la fusione radar produce una mappa 2D
su una griglia di stanza FISSA e nota (la stessa geometria usata per la
matrice di backprojection, gia' precalcolata e iniettata come pesi non
allenabili). Il blocco di refine convoluzionale, pero', non "sa"
esplicitamente in che punto della griglia si trova ciascuna cella se non
tramite il campo recettivo appreso CoordConv gli da' questa
informazione, cosa che nel paper originale si dimostra
aiutare proprio i task in cui la posizione assoluta conta.

Uso in train_v2.py: cambiare SOLO la riga di import
    from model_v2 import build_model_v2
in
    from model_v2_coordconv import build_model_v2_coordconv as build_model_v2
"""

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L

from model_v2 import build_rangenet_layers


@keras.saving.register_keras_serializable(package="model_v2_coordconv")
def _tile_to_batch(args):
    fused_t, coord_t = args
    batch = tf.shape(fused_t)[0]
    return tf.tile(coord_t, [batch, 1, 1, 1])


def _make_coord_grid(grid_h, grid_w):
    """Griglia di coordinate (x, y) normalizzate in [-1, 1], shape
    (grid_h, grid_w, 2) -- canale 0 = x (colonna), canale 1 = y (riga),
    stessa convenzione riga/colonna usata ovunque nel progetto per
    grid_shape=(grid_h, grid_w)."""
    yy = np.linspace(-1.0, 1.0, grid_h, dtype=np.float32).reshape(grid_h, 1)
    xx = np.linspace(-1.0, 1.0, grid_w, dtype=np.float32).reshape(1, grid_w)
    yy_grid = np.tile(yy, (1, grid_w))
    xx_grid = np.tile(xx, (grid_h, 1))
    return np.stack([xx_grid, yy_grid], axis=-1).astype(np.float32)


def build_model_v2_coordconv(feature_shape, grid_shape, radars_1based, projection_matrices,
                              width=1.0, refine_width=16, dropout=0.0, bn_momentum=0.99,
                              dilation_rate=1, batch_size=None,
                              name="uwb_backprojection_detector_coordconv"):
    """Identico a build_model_v2 (model_v2.py) fino alla fusione dei
    radar inclusa; da li' in poi concatena 2 canali costanti di
    coordinate prima del blocco di refine."""
    grid_h, grid_w = grid_shape
    n_bins = feature_shape[0]

    apply_rangenet = build_rangenet_layers(feature_shape, width=width, bn_momentum=bn_momentum,
                                            dilation_rate=dilation_rate)

    radar_inputs = []
    projected_maps = []
    for r in radars_1based:
        inp_r = L.Input(shape=feature_shape, batch_size=batch_size, name=f"radar_{r}")
        radar_inputs.append(inp_r)

        range_resp = apply_rangenet(inp_r)

        proj_layer = L.Dense(grid_h * grid_w, use_bias=False, trainable=False,
                              name=f"backproj_radar_{r}")
        proj_flat = proj_layer(range_resp)
        proj_map = L.Reshape((grid_h, grid_w, 1), name=f"reshape_radar_{r}")(proj_flat)
        projected_maps.append(proj_map)

    fused = L.Add(name="fuse_radars")(projected_maps) if len(projected_maps) > 1 else projected_maps[0]

    coord_np = _make_coord_grid(grid_h, grid_w)              # (grid_h, grid_w, 2)
    coord_const = tf.constant(coord_np[np.newaxis, ...])      # (1, grid_h, grid_w, 2)

    coord_channels = L.Lambda(_tile_to_batch, name="coord_tile_to_batch")([fused, coord_const])
    fused_coord = L.Concatenate(name="fuse_coords", axis=-1)([fused, coord_channels])

    # Refinement Spaziale 2D sulla griglia della stanza (input a 3
    # canali: 1 fusione radar + 2 coordinate, invece di 1 come in
    # model_v2.py -- unica differenza nel blocco di refine)
    x = L.Conv2D(refine_width, 3, padding="same", use_bias=False, name="refine_conv1")(fused_coord)
    x = L.BatchNormalization(momentum=bn_momentum, name="refine_bn1")(x)
    x = L.ReLU(name="refine_relu1")(x)
    if dropout > 0:
        x = L.SpatialDropout2D(dropout, name="refine_dropout1")(x)

    x = L.Conv2D(refine_width, 3, padding="same", use_bias=False, name="refine_conv2")(x)
    x = L.BatchNormalization(momentum=bn_momentum, name="refine_bn2")(x)
    x = L.ReLU(name="refine_relu2")(x)
    if dropout > 0:
        x = L.SpatialDropout2D(dropout, name="refine_dropout2")(x)

    heat_logits = L.Conv2D(1, 1, padding="same", name="heatmap_logits")(x)
    heat_prob = L.Activation("sigmoid", name="heatmap_prob")(heat_logits)

    model = tf.keras.Model(radar_inputs, [heat_logits, heat_prob], name=name)

    for r in radars_1based:
        layer = model.get_layer(f"backproj_radar_{r}")
        P_r = projection_matrices[f"radar_{r}"].astype(np.float32)
        assert P_r.shape == (grid_h * grid_w, n_bins), \
            f"radar_{r}: attesa matrice {(grid_h*grid_w, n_bins)}, trovata {P_r.shape}"
        layer.set_weights([P_r.T])

    return model