from datetime import datetime

from strand.manifests import Manifest


def test_manifest_save_and_load(tmp_path):
    manifest = Manifest(
        run_id="run-1",
        timestamp=datetime.utcnow(),
        experiment="exp",
        inputs={"sequences": []},
        optimizer={"method": "cem"},
        reward_blocks=[],
        results={"scores": []},
    )
    path = manifest.save(tmp_path / "manifest.json")
    loaded = Manifest.load(path)
    assert loaded.run_id == manifest.run_id
    assert loaded.experiment == "exp"
