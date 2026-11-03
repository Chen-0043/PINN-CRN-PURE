#!/usr/bin/env python3
"""Run MetRS enzyme-pool-only perturbations with sparse full-network snapshots.

This wrapper intentionally preserves the earlier MetRS perturbation semantics
used by ``metrs_light_refine_20260712``:

    aaRS = total enzyme pool synchronized across enzyme-containing species;
           aggregate AA / bg_feed_AA is NOT co-scaled.

The main rerun script was later changed for total-aaRS supply sensitivity by
co-scaling the aggregate AA limiter.  That is a different perturbation axis, so
for the MetRS-specific peak at 0.01x we patch only the aaRS synchronization
function before delegating to the main runner.
"""

from __future__ import annotations

from typing import Dict, List

import rerun_synced_ntp_aars_perturbations_20260711 as base


def synchronize_aars_enzyme_only(model: "base.libsbml.Model", component: str, scale: float) -> Dict[str, object]:
    species_scaled: List[str] = []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sid = sp.getId()
        if component in sid and base.scale_species_initial(model, sid, scale):
            species_scaled.append(sid)
    return {
        "kind": "aars_total_enzyme_pool",
        "species_scaled": species_scaled,
        "params_scaled": [],
        "params_set": {},
    }


base.synchronize_aars_perturbation = synchronize_aars_enzyme_only


if __name__ == "__main__":
    base.main()
