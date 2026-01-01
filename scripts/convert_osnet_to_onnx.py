"""
Convert OSNet from PyTorch (torchreid) to ONNX format.

Usage:
    pip install torchreid
    python scripts/convert_osnet_to_onnx.py

This will download pretrained OSNet x0.25 and convert to ONNX.
"""

import os
import sys
import torch

def convert_osnet_to_onnx(variant="x0_25", output_path="models/osnet_x0_25.onnx"):
    """
    Convert OSNet to ONNX format.

    Args:
        variant: OSNet variant ("x0_25", "x0_5", "x0_75", "x1_0")
        output_path: Output ONNX file path
    """
    try:
        import torchreid
    except ImportError:
        print("torchreid not installed. Installing...")
        os.system("pip install torchreid")
        import torchreid

    print(f"Loading OSNet {variant}...")

    # Build model
    model_name = f"osnet_{variant}"
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1,  # We only need features, not classification
        loss='softmax',
        pretrained=True
    )

    model.eval()

    # Remove classifier layer - we only want features
    # OSNet outputs features from the last conv layer
    # The default torchreid model includes classifier, we need to extract backbone

    print(f"Model loaded. Feature dim: {model.feature_dim}")

    # Create dummy input (batch=1, channels=3, height=256, width=128)
    dummy_input = torch.randn(1, 3, 256, 128)

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")

    # We need a wrapper to get only features (not classifier output)
    class OSNetFeatureExtractor(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # torchreid models have a forward() that returns (features, logits) or just features
            # We use featuremaps() method to get features before GAP
            f = self.model.featuremaps(x)  # (B, C, H, W)
            # Global average pooling
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1)
            f = f.view(f.size(0), -1)  # (B, C)
            return f

    feature_model = OSNetFeatureExtractor(model)
    feature_model.eval()

    # Test output shape
    with torch.no_grad():
        test_out = feature_model(dummy_input)
        print(f"Output shape: {test_out.shape}")

    torch.onnx.export(
        feature_model,
        dummy_input,
        output_path,
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

    print(f"ONNX model saved to: {output_path}")

    # Verify the model
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        inp = np.random.randn(1, 3, 256, 128).astype(np.float32)
        out = sess.run(None, {"input": inp})[0]
        print(f"ONNX verification: input {inp.shape} -> output {out.shape}")
        print("Conversion successful!")
    except Exception as e:
        print(f"ONNX verification failed: {e}")

    return output_path


def download_pretrained_onnx(output_path="models/osnet_x0_25.onnx"):
    """
    Try to download pre-converted ONNX model.
    If not available, fall back to conversion.
    """
    import urllib.request

    urls = [
        # Community pre-converted models
        "https://huggingface.co/spaces/kadirnar/torchyolo/resolve/main/osnet_x0_25.onnx",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for url in urls:
        try:
            print(f"Trying to download from: {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Failed: {e}")
            continue

    print("No pre-converted model available. Converting from PyTorch...")
    return convert_osnet_to_onnx(output_path=output_path)


if __name__ == "__main__":
    output = "models/osnet_x0_25.onnx"

    if len(sys.argv) > 1:
        output = sys.argv[1]

    if os.path.exists(output):
        print(f"Model already exists: {output}")
        sys.exit(0)

    # Try download first, then convert
    try:
        download_pretrained_onnx(output)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Falling back to conversion...")
        convert_osnet_to_onnx(output_path=output)
