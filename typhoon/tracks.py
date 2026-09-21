"""Great-circle distance in kilometers."""

import numpy as np


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad((np.asarray(lon2) - lon1 + 180) % 360 - 180)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
