"""
Download OSNet Re-ID model (Market1501 trained) and convert to ONNX.
"""
import os
import sys

os.makedirs("models", exist_ok=True)
pt_path = "models/osnet_x0_25_market.pth"
onnx_path = "models/osnet_x0_25.onnx"

# Skip if already converted
if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 1000000:
    print(f"Model already exists: {onnx_path}")
    sys.exit(0)

# Download if needed
if not os.path.exists(pt_path):
    import gdown
    file_id = "1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj"
    print("Downloading OSNet (Market1501 trained)...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", pt_path, quiet=False)

print(f"PyTorch model: {pt_path} ({os.path.getsize(pt_path)} bytes)")

# Convert to ONNX
print("\nConverting to ONNX...")
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))

class ChannelGate(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        mid = max(in_c // reduction, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, mid, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_c, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.global_avgpool(x)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y

class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction=4):
        super().__init__()
        mid = out_c // reduction
        self.conv1 = ConvBlock(in_c, mid, 1)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = nn.Sequential(nn.Conv2d(mid, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.gate(self.conv2a(x)) + self.gate(self.conv2b(x)) + self.gate(self.conv2c(x)) + self.gate(self.conv2d(x))
        x = self.conv3(x)
        if self.downsample is not None:
            res = self.downsample(res)
        return self.relu(x + res)

class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class OSNet(nn.Module):
    def __init__(self, num_classes=751, width_mult=0.25):
        super().__init__()
        # channels: [16, 64, 96, 128] for x0.25
        channels = [int(c * width_mult) for c in [64, 256, 384, 512]]

        self.conv1 = ConvBlock(3, channels[0], 7, s=2, p=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # conv2 includes transition layer
        self.conv2 = nn.Sequential(
            self._make_layer(channels[0], channels[1], 2),
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, 2)
        )

        # conv3 includes transition layer
        self.conv3 = nn.Sequential(
            self._make_layer(channels[1], channels[2], 2),
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, 2)
        )

        self.conv4 = self._make_layer(channels[2], channels[3], 2)
        self.conv5 = Conv1x1(channels[3], channels[3])  # 128 -> 128

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Feature embedding: 128 -> 512
        self.fc = nn.Sequential(
            nn.Linear(channels[3], 512),
            nn.BatchNorm1d(512)
        )

        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, in_c, out_c, blocks):
        layers = [OSBlock(in_c, out_c)]
        for _ in range(1, blocks):
            layers.append(OSBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # 512-dim features
        return x

# Load weights
print("Loading Market1501 trained weights...")
model = OSNet(num_classes=751, width_mult=0.25)
state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)

# Map old keys to new structure
new_state = {}
for k, v in state_dict.items():
    if k.startswith('classifier'):
        continue  # Skip classifier
    # Handle conv2, conv3 structure differences
    new_k = k
    # conv2.2.0 -> conv2.1 (transition)
    if 'conv2.2.' in k:
        new_k = k.replace('conv2.2.', 'conv2.1.')
    elif 'conv3.2.' in k:
        new_k = k.replace('conv3.2.', 'conv3.1.')
    new_state[new_k] = v

model.load_state_dict(new_state, strict=False)
model.eval()

# Export to ONNX
print("Exporting to ONNX (512-dim features)...")
dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(
    model, dummy, onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['features'],
    dynamic_axes={'input': {0: 'batch'}, 'features': {0: 'batch'}}
)

print(f"\nSUCCESS! Model saved to: {onnx_path}")
print(f"File size: {os.path.getsize(onnx_path)} bytes")

# Verify
print("\nVerifying...")
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
out = sess.run(None, {'input': np.random.randn(1, 3, 256, 128).astype(np.float32)})[0]
print(f"Output shape: {out.shape} (should be [1, 512])")
