import json
from pathlib import Path

from experiments.m1_adamw_tuning import GRID, _rank


def _metrics(path: Path, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tuning_accuracy": accuracy}) + "\n", encoding="utf-8")


def test_grid_is_exact_and_ranking_enforces_clean_constraint(tmp_path: Path) -> None:
    assert len(GRID) == 12
    assert {(item.learning_rate, item.weight_decay) for item in GRID} == {
        (lr, wd) for lr in (3e-5, 1e-4, 3e-4, 1e-3) for wd in (0.0, 0.01, 0.1)
    }
    configs = list(GRID[:2])
    for config, clean, noisy in ((configs[0], 0.90, 0.80), (configs[1], 0.89, 0.95)):
        _metrics(tmp_path / config.config_id / "clean-seed201" / "epoch_metrics.jsonl", clean)
        _metrics(tmp_path / config.config_id / "noisy-seed201" / "epoch_metrics.jsonl", noisy)
    ranking = _rank(configs, tmp_path, (201,))
    assert ranking[0]["config_id"] == configs[0].config_id
    assert ranking[1]["clean_eligible"] is False
