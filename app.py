import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ============================================================
# Signature Verification - Streamlit App
# ============================================================
# Put your trained model at:
# Model_weights/best.pt
#
# IMPORTANT:
# This file is a deployment template. If best.pt contains only
# a state_dict from a custom Siamese architecture, the model
# architecture must match the architecture used during training.
# ============================================================

st.set_page_config(
    page_title="Signature Verification",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Signature Verification")
st.write("Upload a reference signature and a query signature to compare them.")

MODEL_PATH = os.path.join("Model_weights", "best.pt")

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():
    """
    Attempts to load a complete PyTorch model saved with torch.save(model, ...).

    If your best.pt contains a state_dict instead, this function needs to be
    changed to recreate the exact model architecture used during training.
    """
    if not os.path.exists(MODEL_PATH):
        return None, f"Model not found: {MODEL_PATH}"

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
            weights_only=False
        )

        # Case 1: complete nn.Module
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.eval()
            return model, None

        # Case 2: checkpoint containing a complete model
        if isinstance(checkpoint, dict) and isinstance(
            checkpoint.get("model"), nn.Module
        ):
            model = checkpoint["model"]
            model.eval()
            return model, None

        # Case 3: state_dict/checkpoint
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("weights")
            )

            if state_dict is not None:
                return None, (
                    "best.pt contains a state_dict/checkpoint rather than a "
                    "complete PyTorch model. The exact ResNet-50/Siamese "
                    "architecture used during training must be recreated "
                    "before deployment."
                )

        return None, "Unsupported best.pt format."

    except Exception as e:
        return None, f"Could not load model: {e}"


def get_embedding(model, image):
    """Generate an embedding from a single signature image."""
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    # Handle common model output formats
    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        for key in ["embedding", "embeddings", "features", "output"]:
            if key in output:
                output = output[key]
                break
        else:
            output = next(iter(output.values()))

    return output.flatten(1)


def cosine_similarity(a, b):
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return torch.sum(a * b, dim=1).item()


# -----------------------------
# Upload section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Signature")
    reference_file = st.file_uploader(
        "Upload reference",
        type=["png", "jpg", "jpeg"],
        key="reference"
    )

with col2:
    st.subheader("Query Signature")
    query_file = st.file_uploader(
        "Upload query",
        type=["png", "jpg", "jpeg"],
        key="query"
    )

if reference_file:
    st.image(
        Image.open(reference_file),
        caption="Reference",
        use_container_width=True
    )

if query_file:
    st.image(
        Image.open(query_file),
        caption="Query",
        use_container_width=True
    )

# -----------------------------
# Verification
# -----------------------------
if st.button("🔍 Verify Signature", use_container_width=True):

    if reference_file is None or query_file is None:
        st.warning("Please upload both reference and query signatures.")
        st.stop()

    with st.spinner("Loading model and comparing signatures..."):
        model, error = load_model()

        if error:
            st.error(error)
            st.info(
                "If your best.pt is a Siamese/ResNet-50 state_dict, "
                "send me the training/model definition and I will adapt "
                "this app to your exact checkpoint."
            )
            st.stop()

        try:
            reference_image = Image.open(reference_file).convert("RGB")
            query_image = Image.open(query_file).convert("RGB")

            reference_embedding = get_embedding(model, reference_image)
            query_embedding = get_embedding(model, query_image)

            score = cosine_similarity(
                reference_embedding,
                query_embedding
            )

            st.subheader("Result")
            st.metric("Similarity Score", f"{score:.4f}")

            # Default demo threshold.
            # Replace this with your validated threshold from your experiment.
            threshold = 0.443431

            st.write(f"Threshold: `{threshold:.6f}`")

            if score >= threshold:
                st.success("✅ GENUINE")
            else:
                st.error("❌ FORGED")

        except Exception as e:
            st.error(f"Inference error: {e}")
            st.info(
                "The preprocessing/model output may differ from the training "
                "code. Use the exact preprocessing and inference architecture "
                "from your trained ResNet-50 project."
            )

st.divider()
st.caption("ResNet-50 Signature Verification • CPU inference")
