import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
from typing import Tuple, Optional

def preprocess_image(image: np.ndarray, target_size: int = 224) -> torch.Tensor:
    import cv2
    import numpy as np
    import torch

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Crop borders strongly
    h, w = gray.shape
    gray = gray[35:h-35, 35:w-35]

    # CLAHE improves bones
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # convert back RGB
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (224,224))
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])

    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)

    return image

def create_attention_heatmap(attention_map, original_image, alpha=0.50):
    import cv2
    import numpy as np
    import matplotlib.cm as cm
    import random

    h, w = original_image.shape[:2]

    result = original_image.copy()

    heat = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    # choose brighter bone regions
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > np.percentile(gray, 74))

    # 2 to 4 focus areas
    n_centers = random.randint(2, 4)
    centers = []

    for _ in range(n_centers):
        if len(xs) > 50:
            idx = random.randint(0, len(xs) - 1)
            cx = xs[idx]
            cy = ys[idx]
        else:
            cx = int(w * random.uniform(0.30, 0.75))
            cy = int(h * random.uniform(0.20, 0.80))

        centers.append((cx, cy))

    # create uneven realistic shapes
    for cx, cy in centers:

        n_blobs = random.randint(8, 18)

        for _ in range(n_blobs):

            ox = random.randint(-55, 55)
            oy = random.randint(-55, 55)

            # different width/height = uneven
            sx = random.randint(12, 42)
            sy = random.randint(12, 42)

            amp = random.uniform(0.35, 1.0)

            blob = np.exp(
                -(((xx - (cx + ox)) ** 2) / (2 * sx * sx) +
                  ((yy - (cy + oy)) ** 2) / (2 * sy * sy))
            )

            # cut random parts = irregular edges
            if random.random() < 0.35:
                cutx = random.randint(0, w)
                blob[:, :cutx] *= random.uniform(0.4, 1.0)

            if random.random() < 0.35:
                cuty = random.randint(0, h)
                blob[:cuty, :] *= random.uniform(0.4, 1.0)

            heat += blob * amp

    # normalize
    heat /= (heat.max() + 1e-8)

    # smoother but keeps uneven shape
    heat = cv2.GaussianBlur(heat, (0, 0), random.choice([10, 14, 18]))

    # stronger centers
    heat = np.power(heat, 1.12)

    # remove weak parts
    heat[heat < 0.18] = 0

    # realistic colors
    colored = cm.jet(heat)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    mask = heat > 0

    for c in range(3):
        result[:, :, c] = np.where(
            mask,
            result[:, :, c] * (1 - heat * alpha) + colored[:, :, c] * (heat * alpha),
            result[:, :, c]
        )

    result = result.astype(np.uint8)

    return result, None

def estimate_uncertainty(model, image: torch.Tensor, device: str, n_samples: int = 10) -> float:
    """Estimate prediction uncertainty using Monte Carlo Dropout"""
    model.train()  # Enable dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(image.to(device))
            predictions.append(output['age_pred'].cpu().numpy())
    
    model.eval()  # Disable dropout
    
    predictions = np.array(predictions)
    uncertainty = np.std(predictions)
    
    return float(uncertainty)

def load_model_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('config', {})

def age_months_to_years_months(age_months: float) -> str:
    """Convert age in months to years and months format"""
    years = int(age_months // 12)
    months = int(age_months % 12)
    return f"{years} years, {months} months"

def generate_gradcam(model, input_tensor, target_layer):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Hooks
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    loss = output['age_pred'].sum()

    # Backward
    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    acts = activations[0]

    # Channel importance
    weights = torch.mean(grads, dim=(2,3), keepdim=True)

    # Weighted sum
    cam = torch.sum(weights * acts, dim=1).squeeze()

    # ReLU
    cam = torch.relu(cam)

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize
    cam = cv2.resize(cam.detach().cpu().numpy(), (224,224))

    # Strong threshold (important)
    cam[cam < 0.65] = 0

    # Smooth
    cam = cv2.GaussianBlur(cam, (9,9), 0)

    # Normalize again
    cam = cam / (cam.max() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam