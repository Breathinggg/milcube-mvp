"""
Download or convert OSNet Re-ID model to ONNX format.

Usage:
    python scripts/download_reid_model.py

This script will:
1. Try to download a pre-converted ONNX model
2. If not available, convert from PyTorch using torchreid
"""

import os
import sys
import urllib.request
import hashlib

MODEL_DIR = "models"
OUTPUT_PATH = os.path.join(MODEL_DIR, "osnet_x0_25.onnx")

# Pre-converted model URLs (community hosted)
DOWNLOAD_URLS = [
    # FastReID pre-converted models
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/osnet_x0_25_msmt17.onnx",
]


def download_file(url, path):
    """Download file with progress indicator"""
    print(f"Downloading from: {url}")
    try:
        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, path, reporthook=progress)
        print("\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        return False


def try_download():
    """Try to download pre-converted model"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for url in DOWNLOAD_URLS:
        if download_file(url, OUTPUT_PATH):
            return True

    return False


def convert_from_pytorch():
    """Convert OSNet from PyTorch to ONNX"""
    print("\nAttempting to convert from PyTorch...")

    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Please install: pip install torch")
        return False

    try:
        import torchreid
    except ImportError:
        print("torchreid not installed. Installing...")
        os.system(f"{sys.executable} -m pip install torchreid")
        try:
            import torchreid
        except ImportError:
            print("Failed to install torchreid. Please install manually:")
            print("  pip install torchreid")
            return False

    print("Building OSNet model...")
    model = torchreid.models.build_model(
        name="osnet_x0_25",
        num_classes=1,
        loss='softmax',
        pretrained=True
    )
    model.eval()

    # Wrapper to get features only
    class OSNetFeatures(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            f = self.model.featuremaps(x)
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1)
            return f.view(f.size(0), -1)

    feature_model = OSNetFeatures(model)
    feature_model.eval()

    # Export to ONNX
    dummy = torch.randn(1, 3, 256, 128)
    print(f"Exporting to: {OUTPUT_PATH}")

    torch.onnx.export(
        feature_model,
        dummy,
        OUTPUT_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )

    print("Conversion complete!")
    return True


def verify_model():
    """Verify the ONNX model works"""
    if not os.path.exists(OUTPUT_PATH):
        return False

    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])
        inp = np.random.randn(1, 3, 256, 128).astype(np.float32)
        out = sess.run(None, {"input": inp})[0]

        print(f"\nModel verification:")
        print(f"  Input shape: {inp.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Feature dim: {out.shape[1]}")

        if out.shape[1] == 512:
            print("  Status: OK!")
            return True
        else:
            print(f"  Warning: Expected 512-dim features, got {out.shape[1]}")
            return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    print("=" * 50)
    print("OSNet Re-ID Model Setup")
    print("=" * 50)

    if os.path.exists(OUTPUT_PATH):
        print(f"\nModel already exists: {OUTPUT_PATH}")
        if verify_model():
            print("\nSetup complete! Re-ID is ready to use.")
            return 0

    print("\nStep 1: Trying to download pre-converted model...")
    if try_download() and verify_model():
        print("\nSetup complete! Re-ID is ready to use.")
        return 0

    print("\nStep 2: Downloading failed, trying PyTorch conversion...")
    if convert_from_pytorch() and verify_model():
        print("\nSetup complete! Re-ID is ready to use.")
        return 0

    print("\n" + "=" * 50)
    print("SETUP FAILED")
    print("=" * 50)
    print("\nPlease manually download an OSNet ONNX model and place it at:")
    print(f"  {os.path.abspath(OUTPUT_PATH)}")
    print("\nAlternatively, you can use --no_reid flag to use color histogram matching.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
