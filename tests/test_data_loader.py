import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import data_loader


def make_fake_dataset(train_count=3):
    train = [{"text": f"train_{i}", "label": i % 3} for i in range(train_count)]
    validation = [{"text": "val_0", "label": 1}]
    test = [{"text": "test_0", "label": 2}]
    return {"train": train, "validation": validation, "test": test}


def test_load_sentiment_dataset_success(monkeypatch):
    fake = make_fake_dataset(5)

    def fake_load_dataset(name, config, cache_dir=None):
        return fake

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)

    ds = data_loader.load_sentiment_dataset(cache_dir=None)
    assert isinstance(ds, dict)
    assert set(ds.keys()) == {"train", "validation", "test"}
    assert len(ds["train"]) == 5


def test_load_sentiment_dataset_empty_split_raises(monkeypatch):
    fake = {"train": [], "validation": [{"text": "v", "label": 0}], "test": [{"text": "t", "label": 1}]}

    def fake_load_dataset(name, config, cache_dir=None):
        return fake

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError):
        data_loader.load_sentiment_dataset()


def test_get_splits_and_sample_examples(monkeypatch):
    fake = make_fake_dataset(4)

    def fake_load_dataset(name, config, cache_dir=None):
        return fake

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)

    train, val, test = data_loader.get_splits()
    assert len(train) == 4
    assert len(val) == 1
    assert len(test) == 1

    samples = data_loader.sample_examples(n=2)
    assert len(samples) == 2
    for s in samples:
        assert "text" in s and "label" in s
        assert s["label"] in data_loader.get_label_mapping().values()


def test_get_label_mapping():
    mapping = data_loader.get_label_mapping()
    assert mapping == {0: "negative", 1: "neutral", 2: "positive"}
