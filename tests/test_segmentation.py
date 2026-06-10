from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def test_sam3_detect_forwards_to_comfy_node(monkeypatch: Any) -> None:
    import comfy_diffusion.segmentation as segmentation

    calls: dict[str, Any] = {}

    class FakeSAM3Detect:
        @classmethod
        def execute(cls, **kwargs: Any) -> Any:
            calls.update(kwargs)
            return SimpleNamespace(result=("masks", "bboxes"))

    monkeypatch.setattr(segmentation, "_get_sam3_detect_type", lambda: FakeSAM3Detect)

    result = segmentation.sam3_detect(
        "model",
        "image",
        conditioning="conditioning",
        threshold=0.7,
        refine_iterations=3,
    )

    assert result == ("masks", "bboxes")
    assert calls == {
        "model": "model",
        "image": "image",
        "conditioning": "conditioning",
        "bboxes": None,
        "positive_coords": None,
        "negative_coords": None,
        "threshold": 0.7,
        "refine_iterations": 3,
        "individual_masks": False,
    }
