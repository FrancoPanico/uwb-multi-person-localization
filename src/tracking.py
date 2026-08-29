"""
tracking.py
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Filtro di Kalman 2D, modello a velocita' costante
# --------------------------------------------------------------------------- #

class KalmanTrack2D:

    def __init__(self, x0, y0, process_noise_pos=0.02, process_noise_vel=0.05,
                 meas_noise=0.35):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([meas_noise**2, meas_noise**2, 1.0, 1.0])
        self.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag([process_noise_pos**2, process_noise_pos**2,
                           process_noise_vel**2, process_noise_vel**2])
        self.R = np.diag([meas_noise**2, meas_noise**2])

        self.missed = 0
        self.hits = 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = np.array(z, dtype=np.float64) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0
        self.hits += 1

    @property
    def position(self):
        return [float(self.x[0]), float(self.x[1])]


# --------------------------------------------------------------------------- #
# Associazione 
# --------------------------------------------------------------------------- #

def greedy_match(track_positions, detections, max_dist):
    """
    track_positions: lista di [x,y] (posizioni PREDETTE delle tracce vive)
    detections: lista di [x,y] (rilevazioni del frame corrente)
    Ritorna: lista di (track_idx, det_idx) associati, in ordine di distanza
    crescente, rispettando la soglia max_dist e l'unicita' (1-a-1).
    """
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


# --------------------------------------------------------------------------- #
# Tracker multi-bersaglio
# --------------------------------------------------------------------------- #

class HeatmapTracker:

  def __init__(
      self,
      max_tracks=4,
      max_match_dist=1.40,
      max_missed=4,
      min_hits_to_report=3,
      thresh_high=0.33,  # Soglia alta per NUOVE tracce
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
    """detections: lista di [x, y, score] estratte con thresh_low (0.20)"""
    for tr in self.tracks:
      tr.predict()

    track_positions = [tr.position for tr in self.tracks]

    # Estrai solo x, y per il matching geometrico
    det_coords = [[d[0], d[1]] for d in detections]
    assignment = greedy_match(
        track_positions, det_coords, self.max_match_dist
    )

    matched_t = {ti for ti, _ in assignment}
    matched_d = {di for _, di in assignment}

    # Aggiorna le tracce esistenti (anche con score bassi tra 0.20 e 0.33)
    for ti, di in assignment:
      self.tracks[ti].update(det_coords[di])

    for ti, tr in enumerate(self.tracks):
      if ti not in matched_t:
        tr.missed += 1

    self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

    # CREAZIONE NUOVE TRACCE: Solo se lo score supera thresh_high (0.33)
    for di, d in enumerate(detections):
      if di in matched_d:
        continue
      score = d[2]
      if score < self.thresh_high:
        continue  # Scarta i picchi deboli per nuove tracce

      if len(self.tracks) >= self.max_tracks:
        continue

      self.tracks.append(
          KalmanTrack2D(
              d[0],
              d[1],
              self.process_noise_pos,
              self.process_noise_vel,
              self.meas_noise,
          )
      )

    reported = [
        tr.position for tr in self.tracks if tr.hits >= self.min_hits_to_report
    ]
    return reported[: self.max_tracks]