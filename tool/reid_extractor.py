"""
Re-ID Feature Extractor using OSNet (ONNX)

OSNet is a lightweight person re-identification model that outputs 512-dim feature vectors.
Use cosine similarity to match features across cameras.

Performance on Orin Nano (8GB): ~10-15ms per person
"""

import cv2
import numpy as np
import os
import urllib.request
import sys

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def auto_download_osnet(save_path: str) -> bool:
    """Auto-download OSNet ONNX model if not exists"""
    if os.path.exists(save_path):
        return True

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Try multiple sources (verified working URLs)
    urls = [
        # PINTO model zoo - verified working
        "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/115_ReID/osnet_x0_25/saved_model/opset11/osnet_x0_25_msmt17_256x128.onnx",
        # Backup from HuggingFace
        "https://huggingface.co/pinto0309/PINTO_model_zoo/resolve/main/115_ReID/osnet_x0_25/saved_model/opset11/osnet_x0_25_msmt17_256x128.onnx",
    ]

    print(f"[ReID] Model not found, downloading OSNet...")

    for url in urls:
        try:
            print(f"[ReID] Trying: {url}")

            def progress(count, block_size, total_size):
                pct = int(count * block_size * 100 / max(1, total_size))
                sys.stdout.write(f"\r[ReID] Downloading: {pct}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, save_path, reporthook=progress)
            print(f"\n[ReID] Downloaded to: {save_path}")
            return True
        except Exception as e:
            print(f"\n[ReID] Failed: {e}")
            continue

    print("[ReID] ERROR: Could not download model from any source")
    return False


class ReIDExtractor:
    """
    OSNet-based Re-ID feature extractor.

    Input: BGR image crop of a person
    Output: L2-normalized feature vector (128 or 512 dim depending on model)
    """

    def __init__(self, model_path: str, providers: list = None):
        """
        Args:
            model_path: Path to OSNet ONNX model
            providers: ONNX Runtime providers (default: CPU)
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")

        # Auto-download if model doesn't exist
        if not os.path.exists(model_path):
            if not auto_download_osnet(model_path):
                raise FileNotFoundError(f"ReID model not found and download failed: {model_path}")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # OSNet input size: 256x128 (HxW)
        self.input_h = 256
        self.input_w = 128

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Detect feature dimension from model
        self.feature_dim = self._warm_up()

    def _warm_up(self):
        """Warm up the model and detect feature dimension"""
        dummy = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        out = self.session.run([self.output_name], {self.input_name: dummy})[0]
        feat_dim = out.shape[1]
        print(f"[ReID] Feature dimension: {feat_dim}")
        return feat_dim

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image crop.

        Args:
            img_bgr: BGR image (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, 256, 128)
        """
        # Resize to model input size
        img = cv2.resize(img_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        img = (img - self.mean) / self.std

        # HWC -> CHW, add batch dim
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def preprocess_batch(self, crops: list) -> np.ndarray:
        """
        Preprocess multiple crops into a batch.

        Args:
            crops: List of BGR images

        Returns:
            Batched tensor (N, 3, 256, 128)
        """
        if not crops:
            return np.zeros((0, 3, self.input_h, self.input_w), dtype=np.float32)

        batch = []
        for img in crops:
            # Resize
            resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            # BGR -> RGB, normalize
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - self.mean) / self.std
            # HWC -> CHW
            batch.append(rgb.transpose(2, 0, 1))

        return np.stack(batch, axis=0).astype(np.float32)

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extract feature from a single image crop.

        Args:
            img_bgr: BGR image (H, W, 3)

        Returns:
            L2-normalized feature vector
        """
        inp = self.preprocess(img_bgr)
        feat = self.session.run([self.output_name], {self.input_name: inp})[0]
        feat = feat.flatten()

        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm

        return feat

    def extract_batch(self, crops: list) -> np.ndarray:
        """
        Extract features from multiple crops (batched for efficiency).

        Args:
            crops: List of BGR images

        Returns:
            L2-normalized features (N, feature_dim)
        """
        if not crops:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # Batch inference
        inp = self.preprocess_batch(crops)
        feats = self.session.run([self.output_name], {self.input_name: inp})[0]

        # L2 normalize each feature
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        feats = feats / norms

        return feats


def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized feature vectors.

    Since features are L2-normalized, cosine similarity = dot product.
    Returns value in [-1, 1], higher = more similar.
    """
    return float(np.dot(feat1, feat2))


def cosine_distance_matrix(feats1: np.ndarray, feats2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix between two sets of features.

    Args:
        feats1: (M, D) normalized features
        feats2: (N, D) normalized features

    Returns:
        (M, N) distance matrix, lower = more similar
    """
    # Cosine similarity = dot product for normalized vectors
    sim = np.dot(feats1, feats2.T)
    # Convert to distance
    return 1.0 - sim


# ========== Model Download Helper ==========
def download_osnet_onnx(save_path: str = "models/osnet_x0_25.onnx"):
    """
    Download pre-converted OSNet ONNX model from GitHub releases.

    OSNet x0.25 is the smallest variant:
    - Parameters: 0.5M
    - Input: 256x128
    - Output: 512-dim
    - Inference: ~5-10ms on GPU
    """
    import urllib.request

    url = "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x0_25_msmt17.onnx"

    # Alternative URLs if official doesn't have ONNX
    alt_urls = [
        "https://huggingface.co/datasets/milcube/reid-models/resolve/main/osnet_x0_25.onnx",
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading OSNet model to {save_path}...")

    try:
        urllib.request.urlretrieve(url, save_path)
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Download failed from primary URL: {e}")
        for alt in alt_urls:
            try:
                urllib.request.urlretrieve(alt, save_path)
                print("Download complete from alternative URL!")
                return True
            except:
                continue

    print("ERROR: Could not download OSNet model.")
    print("Please manually download an OSNet ONNX model and place it at:", save_path)
    return False


if __name__ == "__main__":
    # Test the extractor
    import sys

    model_path = "models/osnet_x0_25.onnx"

    if not os.path.exists(model_path):
        print("Model not found. Attempting download...")
        download_osnet_onnx(model_path)

    if os.path.exists(model_path):
        print("Testing ReID extractor...")
        extractor = ReIDExtractor(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Test with dummy image
        dummy = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        feat = extractor.extract(dummy)
        print(f"Feature shape: {feat.shape}")
        print(f"Feature norm: {np.linalg.norm(feat):.4f}")
        print(f"First 10 dims: {feat[:10]}")
    else:
        print("Model not available. Please provide osnet_x0_25.onnx")
