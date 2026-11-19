"""Freeze one pretrained UAX candidate using MFO validation results only."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    stages = ("pretrained", "soft_confidence", "global_skip", "gearnet")
    candidates = []
    for stage in stages:
        path = HERE / f"mf_uax_residue_{stage}_validation.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["settings"]["evaluate_test"] or any("test_fmax" in row for row in run["metrics"]):
            raise ValueError("Candidate selection requires validation-only metrics")
        by_variant = {row["variant"]: row for row in run["metrics"]}
        if stage == "pretrained":
            reference_neutral = by_variant["neutral"]["validation_fmax"]
        neutral = (reference_neutral if stage == "soft_confidence" else
                   by_variant["neutral"]["validation_fmax"])
        aligned = by_variant["aligned"]["validation_fmax"]
        shuffled = by_variant["shuffled"]["validation_fmax"]
        candidates.append({"stage": stage, "aligned": aligned, "neutral": neutral, "shuffled": shuffled,
                           "margin": aligned - max(neutral, shuffled), "strength": by_variant["aligned"]["strength"],
                           "backbone": run["settings"].get("structure_backbone", "saprot"),
                           "global_skip": run["settings"].get("global_skip", False), "source": path.name})
    selected = max(candidates, key=lambda c: (c["margin"], c["aligned"]))
    result = {"method": "Largest MFO validation Fmax gain over the stronger of neutral and shuffled controls",
              "selected": selected, "candidates": candidates, "temporal_seed": 1}
    with (HERE / "uax_residue_temporal_selection.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
