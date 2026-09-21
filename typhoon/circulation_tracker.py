"""Causal association of already detected circulation maxima, including short gaps."""

from typhoon.tracks import great_circle_km


class CirculationTracker:
    def __init__(self, seed, search_km=450, max_gap_steps=1):
        self.last = tuple(seed)
        self.last_lead = 0
        self.older = None
        self.search_km = search_km
        self.max_gap_steps = max_gap_steps
        self.missed = 0
        self.terminated = False

    def locate(self, candidates, lead):
        if self.terminated:
            return None, "terminated_after_gap"
        predicted = self.last
        if self.older is not None:
            p, t = self.older
            dt = self.last_lead - t
            if dt > 0:
                predicted = tuple(
                    a + (a - b) * (lead - self.last_lead) / dt
                    for a, b in zip(self.last, p)
                )
        # Keep the same 450 km gate even across a missing step. Do not enlarge it
        # to acquire an unrelated vortex. No future reference positions enter.
        choices = [
            c
            for c in candidates
            if great_circle_km(*self.last, c["lat"], c["lon"]) <= self.search_km
        ]
        if not choices:
            self.missed += 1
            self.terminated = self.missed > self.max_gap_steps
            return None, "terminated_after_gap" if self.terminated else "temporary_gap"
        p = min(choices, key=lambda c: great_circle_km(*predicted, c["lat"], c["lon"]))
        p = {
            **p,
            "step_distance_km": float(great_circle_km(*self.last, p["lat"], p["lon"])),
            "elapsed_hours_from_last_center": lead - self.last_lead,
            "recovered_after_missing_steps": self.missed,
            "candidate_count_within_search_radius": len(choices),
        }
        if lead > self.last_lead:
            self.older = (self.last, self.last_lead)
        self.last = (p["lat"], p["lon"])
        self.last_lead = lead
        status = "recovered_center" if self.missed else "diagnostic_center"
        self.missed = 0
        return p, status
