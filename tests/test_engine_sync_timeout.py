import re
from pathlib import Path


def test_controlled_network_sync_timeout_does_not_advance_missing_tic():
    source = (
        Path(__file__).parents[1] / "src" / "vizdoom" / "src" / "d_net.cpp"
    ).read_text()
    timeout_guard = re.search(
        r"if\(\(unsigned int\)\*viz_sync_timeout > 0 &&\s*"
        r"\(unsigned int\)\*viz_sync_timeout <= I_MSTime\(\) - waitEnterTime\)"
        r"\s*\{(?P<body>.*?)\n\s*\}",
        source,
        flags=re.DOTALL,
    )

    assert timeout_guard is not None
    body = timeout_guard.group("body")
    assert "return;" in body
    assert "break;" not in body
