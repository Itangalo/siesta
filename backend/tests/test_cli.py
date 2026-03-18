import backend.siesta.cli as cli_module


def test_strip_episodes_removes_nested_episode_lists():
    payload = {
        "policy": "model",
        "episodes": [{"seed": 1}],
        "nested": {
            "episodes": [{"seed": 2}],
            "value": 3,
        },
        "items": [
            {"episodes": [{"seed": 4}], "value": 5},
        ],
    }

    stripped = cli_module.strip_episodes(payload)
    assert "episodes" not in stripped
    assert "episodes" not in stripped["nested"]
    assert "episodes" not in stripped["items"][0]
    assert stripped["nested"]["value"] == 3
