import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models


# ============================================================
# ResNet-50 Signature Verification
# Based directly on the uploaded training/evaluation notebook.
# ============================================================

st.set_page_config(
    page_title="ResNet-50 Signature Verification",
    page_icon="✍️",
    layout="wide",
)

IMAGE_SIZE = 224
EMBED_DIM = 256
MODEL_PATH = Path(__file__).resolve().parent / "Model_weights" / "best.pt"

# Thresholds reported by the uploaded notebook.
# Thresholds were selected from the Akash+Robin validation protocol
# and evaluated on CEDAR in the notebook.
THRESHOLDS = {
    2: 0.343098,
    3: 0.311824,
    6: 0.309600,
    10: 0.351160,
}


# ------------------------------------------------------------
# Exact model architecture from the notebook
# ------------------------------------------------------------
class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # The checkpoint contains the trained weights, so pretrained
        # ImageNet weights are not required at deployment time.
        b = models.resnet50(weights=None)

        old = b.conv1
        c = nn.Conv2d(
            1,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=False,
        )

        # Same construction used in the notebook:
        # initialize the 1-channel convolution from the RGB structure.
        with torch.no_grad():
            # This initialization is irrelevant after loading best.pt,
            # but keeps the architecture identical.
            rgb_model = models.resnet50(weights=None)
            c.weight.copy_(rgb_model.conv1.weight.mean(1, keepdim=True))

        b.conv1 = c
        inf = b.fc.in_features
        b.fc = nn.Identity()

        self.backbone = b
        self.proj = nn.Sequential(
            nn.Linear(inf, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, EMBED_DIM),
        )

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), p=2, dim=1)


# ------------------------------------------------------------
# Exact preprocessing from the notebook
# ------------------------------------------------------------
def preprocess_signature(pil_image):
    """
    Same preprocessing logic as load_signature() + eval_tf
    in the uploaded notebook:

      grayscale
      Otsu threshold
      crop to ink bounding box + 5% padding
      resize while preserving aspect ratio
      center on a 224x224 white canvas
      convert white background / black ink to float tensor [0,1]
    """
    rgb = np.array(pil_image.convert("RGB"))
    x = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    _, b = cv2.threshold(
        x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    ink = cv2.findNonZero(255 - b)

    if ink is not None:
        xx, yy, w, h = cv2.boundingRect(ink)
        pad = max(3, int(0.05 * max(w, h)))

        x = x[
            max(0, yy - pad): min(x.shape[0], yy + h + pad),
            max(0, xx - pad): min(x.shape[1], xx + w + pad),
        ]

    h, w = x.shape

    scale = min(
        IMAGE_SIZE / max(w, 1),
        IMAGE_SIZE / max(h, 1),
    )

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    r = cv2.resize(
        x,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE),
        255,
        np.uint8,
    )

    y0 = (IMAGE_SIZE - nh) // 2
    x0 = (IMAGE_SIZE - nw) // 2

    canvas[
        y0:y0 + nh,
        x0:x0 + nw
    ] = r

    # Same representation as:
    # (1.0-canvas.astype(np.float32)/255.0).astype(np.float32)
    tensor = 1.0 - canvas.astype(np.float32) / 255.0

    return torch.from_numpy(tensor).unsqueeze(0)


# ------------------------------------------------------------
# Load exact best.pt state_dict
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = ResNet50Encoder()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    # The notebook saves:
    # best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # torch.save(best_state, outdir/"best.pt")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    # Handle checkpoints saved with an optional "module." prefix.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    return model


@torch.no_grad()
def get_embedding(model, image):
    x = preprocess_signature(image).unsqueeze(0)
    return model(x).cpu()


@torch.no_grad()
def verify_signatures(model, reference_images, query_image):
    """
    Same distance protocol as notebook:

    - Generate normalized embeddings.
    - Average the reference embeddings for the writer.
    - Calculate Euclidean distance from query embedding to reference center.
    """
    reference_embeddings = torch.cat(
        [get_embedding(model, img) for img in reference_images],
        dim=0,
    )

    query_embedding = get_embedding(model, query_image)

    reference_center = reference_embeddings.mean(dim=0, keepdim=True)

    distance = torch.linalg.vector_norm(
        query_embedding - reference_center,
        dim=1,
    ).item()

    return distance


# ============================================================
# UI
# ============================================================

st.title("✍️ ResNet-50 Signature Verification")
st.write(
    "Upload 2, 3, 6, or 10 genuine reference signatures "
    "from the same writer and one query signature."
)

reference_count = st.selectbox(
    "Number of reference signatures",
    options=[2, 3, 6, 10],
    index=0,
)

st.info(
    f"Using the notebook's {reference_count}-reference threshold: "
    f"{THRESHOLDS[reference_count]:.6f}"
)

reference_files = st.file_uploader(
    f"Upload exactly {reference_count} reference signatures",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    key="references",
)

query_file = st.file_uploader(
    "Upload query signature",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=False,
    key="query",
)

if reference_files:
    if len(reference_files) > reference_count:
        st.warning(
            f"You uploaded {len(reference_files)} reference images. "
            f"Please keep exactly {reference_count}."
        )

    cols = st.columns(min(len(reference_files), 5))

    for i, file in enumerate(reference_files):
        with cols[i % len(cols)]:
            st.image(
                Image.open(file),
                caption=f"Reference {i + 1}",
                use_container_width=True,
            )

if query_file:
    st.image(
        Image.open(query_file),
        caption="Query signature",
        width=350,
    )

st.divider()

if st.button(
    "🔍 Verify Signature",
    type="primary",
    use_container_width=True,
):
    if len(reference_files or []) != reference_count:
        st.error(
            f"Please upload exactly {reference_count} reference signatures."
        )
        st.stop()

    if query_file is None:
        st.error("Please upload a query signature.")
        st.stop()

    try:
        with st.spinner("Loading ResNet-50 and comparing signatures..."):
            model = load_model()

            reference_images = [
                Image.open(f).convert("RGB")
                for f in reference_files
            ]

            query_image = Image.open(query_file).convert("RGB")

            distance = verify_signatures(
                model,
                reference_images,
                query_image,
            )

        threshold = THRESHOLDS[reference_count]
        genuine = distance <= threshold

        st.subheader("Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Distance",
                f"{distance:.6f}",
            )

        with col2:
            st.metric(
                "Threshold",
                f"{threshold:.6f}",
            )

        with col3:
            st.metric(
                "References",
                str(reference_count),
            )

        if genuine:
            st.success(
                "✅ GENUINE — distance is at or below the threshold."
            )
        else:
            st.error(
                "❌ FORGED — distance is above the threshold."
            )

    except Exception as e:
        st.error(f"Verification error: {e}")
        st.exception(e)

st.divider()

st.caption(
    "Model: ResNet-50 encoder • Embedding dimension: 256 • "
    "Preprocessing and distance protocol match the uploaded evaluation notebook."
)
