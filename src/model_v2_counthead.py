"""model_v2_counthead.py
=========================
Variante model_v2.py: aggiunge una testa ausiliaria di
classificazione del conteggio persone (0..max_people), allenata insieme
all'heatmap con una loss combinata. Ispirata a Lin, Hu, Xie, Zhang, Huang,
Chen, "A Multitask Network for People Counting, Motion Recognition, and
Localization Using Through-Wall Radar," 

la testa di conteggio vive SOLO in un secondo modello Keras (multitask_model) usato
UNICAMENTE per il training; il modello da salvare/quantizzare/valutare
resta sempre base_model, che e' l'esatto, identico modello a 2 output
di sempre (stessi layer, stessi pesi, stessi nomi) -- zero modifiche
richieste a quantize_int8.py, evaluate_constraint.py o code.py.

Uso in train_v2_counthead.py:
    base_model, multitask_model = build_model_v2_multitask(...)
    # allenare multitask_model (aggiorna in-place gli stessi Layer di
    # base_model, essendo condivisi -- non copie)
    ...
    base_model.save(...)
"""

import tensorflow as tf
from tensorflow.keras import layers as L

from model_v2 import build_model_v2


def build_model_v2_multitask(feature_shape, grid_shape, radars_1based, projection_matrices,
                              width=1.0, refine_width=16, dropout=0.0, bn_momentum=0.99,
                              dilation_rate=1, batch_size=None, max_people=4,
                              name="uwb_backprojection_detector"):
    base_model = build_model_v2(
        feature_shape, grid_shape, radars_1based, projection_matrices,
        width=width, refine_width=refine_width, dropout=dropout,
        bn_momentum=bn_momentum, dilation_rate=dilation_rate,
        batch_size=batch_size, name=name,
    )

    count_x = L.GlobalAveragePooling2D(name="count_gap")(shared_feat)
    count_x = L.Dense(32, activation="relu", name="count_dense1")(count_x)
    count_logits = L.Dense(max_people + 1, name="count_logits")(count_x)

    multitask_model = tf.keras.Model(
        base_model.inputs,
        [base_model.outputs[0], base_model.outputs[1], count_logits],
        name=name + "_multitask",
    )
    return base_model, multitask_model