"""Subprocess test: verify streamlit_app loads when torch is unavailable."""

import os
import subprocess
import sys
from pathlib import Path


def test_streamlit_app_loads_without_torch():
    """Verify streamlit_app.py imports successfully when torch is blocked."""
    worktree = Path(__file__).parent.parent

    # Script that blocks torch then tries to import streamlit_app
    test_script = """
import sys

# Block torch from being imported
class BlockTorch:
    def find_module(self, fullname, path=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError("torch is blocked for this test (simulating Railway without torch)")
        return None

sys.meta_path.insert(0, BlockTorch())

# Try to import streamlit_app
try:
    from hackathon.opencv_aws import streamlit_app
    print("SUCCESS: streamlit_app imported successfully without torch")
    sys.exit(0)
except ModuleNotFoundError as e:
    if 'torch' in str(e):
        print(f"FAILURE: Import failed due to torch: {e}")
        sys.exit(1)
    else:
        # Some other module missing (e.g., streamlit, pydicom) - that's OK for this test
        print(f"EXPECTED: Different module missing (not torch): {e}")
        sys.exit(0)
except Exception as e:
    print(f"FAILURE: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    # Run in subprocess with worktree on PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree)

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        cwd=str(worktree),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    print("=== Subprocess Output ===")
    print(result.stdout)
    if result.stderr:
        print("=== Subprocess Stderr ===")
        print(result.stderr)

    # Success if:
    # - Exit code 0 (import succeeded or torch error was the only problem)
    # - stdout contains "SUCCESS" or "EXPECTED"
    assert result.returncode == 0, f"Subprocess failed with code {result.returncode}"
    assert "SUCCESS" in result.stdout or "EXPECTED" in result.stdout, \
        f"Unexpected result. Expected SUCCESS or EXPECTED in output, got:\n{result.stdout}"
