"""
model_v2.py
===========
Architettura "Backprojection Detector" avanzata (v2-LateCollapse) per UWB.

Aggiornamento:
- Preserva l'asse delle antenne nei primi due layer convoluzionali 2D 
  per estrarre correlazioni spaziali di fase/ampiezza (Angle-of-Arrival / AoA)
  e sopprimere i riflessi multipath dei muri.
- Collasso dell'asse antenne ritardato (Late Collapse) prima della Backprojection.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L


def build_rangenet_layers(input_shape, width=1.0, bn_momentum=0.99, dilation_rate=1, name="range_net_2d"):
    """
    Crea i layer di RangeNet 2D con Collasso Ritardato (Late Collapse).
    
    input_shape: (n_bins, n_antennas, 2K) per un singolo radar.
    dilation_rate: applica dilatazione lungo l'asse range (distanza).
    """
    n_bins = input_shape[0]
    n_antennas = input_shape[1]
    f = lambda c: max(8, int(c * width))

    # 1. Blocco Convoluzionale 2D (Range x Antenne): Estrazione feature di distanza e angolo (AoA)
    conv_joint1 = L.Conv2D(f(16), (5, n_antennas), padding="same", use_bias=False,
                           dilation_rate=(dilation_rate, 1), name=f"{name}_conv_joint1")
    bn1 = L.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn1")
    relu1 = L.ReLU(name=f"{name}_relu1")

    # 2. Blocco Convoluzionale 2D Profondo: Modellazione avanzata dei rimbalzi e sfasamenti
    conv_joint2 = L.Conv2D(f(32), (5, n_antennas), padding="same", use_bias=False,
                           dilation_rate=(dilation_rate, 1), name=f"{name}_conv_joint2")
    bn2 = L.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn2")
    relu2 = L.ReLU(name=f"{name}_relu2")

    # 3. LATE COLLAPSE: Riduzione dell'asse antenne da N a 1 solo dopo l'estrazione delle feature spaziali
    collapse_ant = L.Conv2D(f(32), (1, n_antennas), padding="valid", use_bias=False,
                            name=f"{name}_collapse_ant")
    bn3 = L.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn3")
    relu3 = L.ReLU(name=f"{name}_relu3")

    # 4. Rifinitura 1D lungo i bin di distanza
    conv_range = L.Conv2D(f(16), (5, 1), padding="same", use_bias=False,
                          name=f"{name}_conv_range")
    bn4 = L.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn4")
    relu4 = L.ReLU(name=f"{name}_relu4")

    # 5. Mappa di evidenza 1D (probabilità per bin) e Flatten per Backprojection
    evidence = L.Conv2D(1, (1, 1), padding="same", activation="sigmoid", name=f"{name}_evidence")
    flatten = L.Reshape((n_bins,), name=f"{name}_flatten")

    def apply_rangenet(x):
        # Feature Extraction 2D congiunta (Bin + Antenne)
        x = conv_joint1(x); x = bn1(x); x = relu1(x)
        x = conv_joint2(x); x = bn2(x); x = relu2(x)

        # Collasso spaziale ritardato
        x = collapse_ant(x); x = bn3(x); x = relu3(x)

        # Rifinitura Range ed Evidenza 1D
        x = conv_range(x); x = bn4(x); x = relu4(x)
        x = evidence(x)
        return flatten(x)

    return apply_rangenet


def build_model_v2(feature_shape, grid_shape, radars_1based, projection_matrices,
                    width=1.0, refine_width=16, dropout=0.0, bn_momentum=0.99,
                    dilation_rate=1, batch_size=None, name="uwb_backprojection_detector"):
    """
    Costruisce l'intero modello Detector Backprojection v2.
    """
    grid_h, grid_w = grid_shape
    n_bins = feature_shape[0]

    # Inizializza RangeNet 2D (condivisa tra tutti i radar per riuso pesi)
    apply_rangenet = build_rangenet_layers(feature_shape, width=width, bn_momentum=bn_momentum,
                                            dilation_rate=dilation_rate)

    radar_inputs = []
    projected_maps = []
    for r in radars_1based:
        inp_r = L.Input(shape=feature_shape, batch_size=batch_size, name=f"radar_{r}")
        radar_inputs.append(inp_r)

        # Estrazione evidenza 1D con pesi condivisi (batch, n_bins)
        range_resp = apply_rangenet(inp_r)

        # Backprojection FISSA 1D -> 2D (moltiplicazione matriciale con P_r^T precalcolata)
        proj_layer = L.Dense(grid_h * grid_w, use_bias=False, trainable=False,
                              name=f"backproj_radar_{r}")
        proj_flat = proj_layer(range_resp)
        proj_map = L.Reshape((grid_h, grid_w, 1), name=f"reshape_radar_{r}")(proj_flat)
        projected_maps.append(proj_map)

    # Fusione punto a punto delle mappe proiettate dai radar
    fused = L.Add(name="fuse_radars")(projected_maps) if len(projected_maps) > 1 else projected_maps[0]

    # Refinement Spaziale 2D sulla griglia della stanza
    x = L.Conv2D(refine_width, 3, padding="same", use_bias=False, name="refine_conv1")(fused)
    x = L.BatchNormalization(momentum=bn_momentum, name="refine_bn1")(x)
    x = L.ReLU(name="refine_relu1")(x)
    if dropout > 0:
        x = L.SpatialDropout2D(dropout, name="refine_dropout1")(x)

    x = L.Conv2D(refine_width, 3, padding="same", use_bias=False, name="refine_conv2")(x)
    x = L.BatchNormalization(momentum=bn_momentum, name="refine_bn2")(x)
    x = L.ReLU(name="refine_relu2")(x)
    if dropout > 0:
        x = L.SpatialDropout2D(dropout, name="refine_dropout2")(x)

    # Layer finale di output Heatmap
    heat_logits = L.Conv2D(1, 1, padding="same", name="heatmap_logits")(x)
    heat_prob = L.Activation("sigmoid", name="heatmap_prob")(heat_logits)

    model = tf.keras.Model(radar_inputs, [heat_logits, heat_prob], name=name)

    # Iniezione dei pesi congelati di Backprojection geometrica (P_r.T)
    for r in radars_1based:
        layer = model.get_layer(f"backproj_radar_{r}")
        P_r = projection_matrices[f"radar_{r}"].astype(np.float32)
        assert P_r.shape == (grid_h * grid_w, n_bins), \
            f"radar_{r}: attesa matrice {(grid_h*grid_w, n_bins)}, trovata {P_r.shape}"
        layer.set_weights([P_r.T])

    return model