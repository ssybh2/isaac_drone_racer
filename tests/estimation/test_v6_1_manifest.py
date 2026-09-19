import json
from pathlib import Path


MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "learned_inertial"
    / "imo_v6_1_manifest.json"
)


def _load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_v6_1_manifest_has_explicit_unique_splits():
    data = _load_manifest()

    assert data["schema"] == "isaac_drone_racer.imo_dataset_manifest.v2"
    traces = data["traces"]
    assert traces

    names = [item["name"] for item in traces]
    paths = [item["path"] for item in traces]

    assert len(names) == len(set(names))
    assert len(paths) == len(set(paths))
    assert {item["split"] for item in traces} == {"train", "val", "test"}


def test_v6_1_manifest_covers_coupled_motion_in_every_split():
    data = _load_manifest()
    required_profiles = {"circle", "lissajous", "racing_like", "translate_yaw"}

    for split in ("train", "val", "test"):
        profiles = {
            item["collector"]["profile"]
            for item in data["traces"]
            if item["split"] == split
        }
        assert required_profiles <= profiles


def test_v6_1_training_split_contains_long_lissajous_and_racing_traces():
    data = _load_manifest()
    long_profiles = {"lissajous", "racing_like"}

    train_long = [
        item
        for item in data["traces"]
        if item["split"] == "train"
        and item["collector"]["profile"] in long_profiles
        and int(item["collector"].get("steps", 0)) >= 3000
    ]

    assert len(train_long) >= 10
    assert any(int(item["collector"]["steps"]) >= 4500 for item in train_long)


def test_v6_1_validation_and_test_use_distinct_seeds():
    data = _load_manifest()

    val_seeds = {
        int(item["collector"]["seed"])
        for item in data["traces"]
        if item["split"] == "val"
    }
    test_seeds = {
        int(item["collector"]["seed"])
        for item in data["traces"]
        if item["split"] == "test"
    }

    assert val_seeds
    assert test_seeds
    assert val_seeds.isdisjoint(test_seeds)
