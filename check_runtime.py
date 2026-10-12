"""Run in a child process before training or starting Streamlit.

Windows native failures produce a nonzero exit code and faulthandler output,
without opening a recurring application-error dialog. No network is used.
"""
import faulthandler
import os
import sys


def main():
    faulthandler.enable()
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0002)
    import numpy as np
    import cv2
    import ezdxf
    import streamlit
    import torch
    from scipy.linalg import inv

    for dtype in (np.float32, np.float64):
        matrix = np.arange(16, dtype=dtype).reshape(4, 4) + np.eye(4, dtype=dtype)
        assert np.isfinite(matrix @ matrix).all()
        # Non-diagonal input actually exercises BLAS/LAPACK; identity can use a shortcut.
        dense = np.array([[2,1,0],[1,3,1],[0,1,4]], dtype=dtype)
        assert np.allclose(dense @ inv(dense), np.eye(3), atol=1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conv = torch.nn.Conv2d(3, 4, 3).to(device)
    conv(torch.randn(2, 3, 16, 16, device=device)).square().mean().backward()
    assert cv2.resize(np.zeros((20, 20), np.uint8), (10, 10)).shape == (10, 10)
    print(
        f"PASS: Python {sys.version.split()[0]}, NumPy {np.__version__}, "
        f"PyTorch {torch.__version__}, ezdxf {ezdxf.__version__}, "
        f"Streamlit {streamlit.__version__}, {device}"
    )
    print(f"NumPy location: {np.__file__}")


if __name__ == "__main__":
    main()
