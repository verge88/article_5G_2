import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_ablation",
    ROOT / "scripts" / "run_ablation.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args():
    return argparse.Namespace(
        epochs=1,
        batch_size=8,
        neighbors=4,
        lr=1e-3,
        target_fpr=1e-3,
        test_memory="safe",
        allow_debug_subset=False,
    )


def test_temporal_heat_variants_have_distinct_result_labels():
    labels = {}
    for model in (
        "temporal_heat",
        "temporal_heat_aux",
        "temporal_heat_continuation",
    ):
        command = MODULE.model_command(
            model,
            data=Path("data"),
            output=Path(f"{model}.json"),
            seed=7,
            args=args(),
        )
        index = command.index("--result-model-name")
        labels[model] = command[index + 1]

    assert labels == {
        "temporal_heat": "temporal_heat",
        "temporal_heat_aux": "temporal_heat_aux",
        "temporal_heat_continuation": "temporal_heat_continuation",
    }
