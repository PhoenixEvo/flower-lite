import os
import sys

sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path

import pytest

from tools.benchmark import FINGERPRINT, parse_registry


def test_fingerprint_matches_context():
    context_path = Path("Context.md")
    content = context_path.read_text(encoding="utf-8")
    
    # Extract the exact string from Context.md Section 5.2
    # The line is usually like `FL-L:S3x3-64...`
    # Let's search for it.
    found = False
    for line in content.splitlines():
        if FINGERPRINT in line:
            found = True
            break
    assert found, f"Fingerprint {FINGERPRINT} not found exactly in Context.md"

def test_hardware_json_schema(tmp_path):
    import os

    from tools.benchmark import write_hardware_json
    
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_hardware_json()
        with open("hardware.json") as f:
            hw = json.load(f)
        
        assert "os" in hw
        assert "python" in hw
        assert "pytorch" in hw
        assert "cuda_available" in hw
        assert "cuda_version" in hw
        assert "gpu_count" in hw
        assert "gpus" in hw
        
        if hw["gpu_count"] > 0:
            for g in hw["gpus"]:
                assert "name" in g
                assert "vram_total_gb" in g
    finally:
        os.chdir(orig_dir)

def test_registry_parser():
    # Valid
    content_valid = """
| Fingerprint | Date | Status | Decided-by | Evidence |
|---|---|---|---|---|
| FL-L | 2026-10-07 | pending | [HUMAN] | N/A |
"""
    res = parse_registry(content_valid)
    assert len(res) == 1
    assert res[0]["status"] == "pending"
    
    # Invalid empty status
    content_invalid_empty = """
| Fingerprint | Date | Status | Decided-by | Evidence |
|---|---|---|---|---|
| FL-L | 2026-10-07 |   | [HUMAN] | N/A |
"""
    with pytest.raises(ValueError, match="Unknown status"):
        parse_registry(content_invalid_empty)
        
    # Invalid unknown status
    content_invalid_unknown = """
| Fingerprint | Date | Status | Decided-by | Evidence |
|---|---|---|---|---|
| FL-L | 2026-10-07 | unapproved | [HUMAN] | N/A |
"""
    with pytest.raises(ValueError, match="Unknown status"):
        parse_registry(content_invalid_unknown)
