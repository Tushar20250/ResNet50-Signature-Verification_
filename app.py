
import re
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

st.set_page_config(
    page_title="SignatureVerify AI",
    page_icon="✍️",
    layout="wide",
)

# ---------- Modern responsive UI ----------
st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 8% 0%, rgba(45,212,191,.10), transparent 28%),
      radial-gradient(circle at 92% 0%, rgba(99,102,241,.14), transparent 30%),
      #07111f;
    color:#eef6ff;
}
.block-container {max-width:1180px;padding:2rem 1rem 3rem;}
[data-testid="stSidebar"] {background:#081525;border-right:1px solid #1d3550;}
.hero {
    padding:30px;
    border-radius:24px;
    border:1px solid #23425f;
    background:linear-gradient(135deg,#10263d,#0b192b);
    box-shadow:0 20px 60px rgba(0,0,0,.25);
    margin-bottom:24px;
}
.hero h1 {margin:0;font-size:clamp(2rem,5vw,3.25rem);letter-spacing:-.04em;}
.hero p {color:#9fb4ca;margin:.6rem 0 0;font-size:1.05rem;}
.badge {
    display:inline-block;padding:5px 11px;border-radius:999px;
    color:#8deaff;background:rgba(45,212,191,.09);
    border:1px solid rgba(45,212,191,.25);
    font-size:.76rem;font-weight:800;
}
.card {
    padding:20px;border-radius:18px;background:#0d1d30;
    border:1px solid #1d3853;margin:12px 0;
}
.muted {color:#9fb4ca;}
.good,.bad {
    padding:22px;border-radius:18px;margin:16px 0;border:1px solid;
}
.good {background:rgba(52,211,153,.08);border-color:rgba(52,211,153,.35);}
.bad {background:rgba(248,113,113,.08);border-color:rgba(248,113,113,.35);}
.result {font-size:1.7rem;font-weight:850;}
[data-testid="stMetric"] {
    background:#0d1d30;border:1px solid #1d3853;
    border-radius:14px;padding:12px;
}
.stButton>button,.stDownloadButton>button {
    min-height:44px;border-radius:12px;
    border:1px solid #315574;
    background:linear-gradient(135deg,#16415b,#303064);
    color:white;font-weight:800;
}
.stButton>button:hover,.stDownloadButton>button:hover {border-color:#45d9ff;}
[data-testid="stFileUploader"] section {
    background:#0a192a;border-color:#294863;border-radius:15px;
}
@media(max-width:700px) {
    .block-container {padding:.9rem .7rem 2rem;}
    .hero {padding:21px 17px;border-radius:18px;}
    .card {padding:15px;border-radius:15px;}
}
</style>
""", unsafe_allow_html=True)

# ---------- Model / notebook constants ----------
IMAGE_SIZE = 224
EMBED_DIM = 256
SEED = 42
BATCH_SIZE = 16
MODEL_PATH = Path(__file__).resolve().parent / "Model_weights" / "best.pt"

# Thresholds selected on Akash + Robin validation in the uploaded notebook.
THRESHOLDS = {
    2: 0.343098,
    3: 0.311824,
    6: 0.309600,
    10: 0.351160,
}
REF_COUNTS = [2, 3, 6, 10]
EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

# ---------- Exact ResNet-50 architecture ----------
class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.resnet50(weights=None)
        old = b.conv1
        b.conv1 = nn.Conv2d(
            1, old.out_channels, old.kernel_size,
            old.stride, old.padding, bias=False
        )
        b.fc = nn.Identity()
        self.backbone = b
        self.proj = nn.Sequential(
            nn.Linear(old.out_channels * 0 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, EMBED_DIM),
        )

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), p=2, dim=1)

# ---------- Exact preprocessing from notebook ----------
def preprocess_array(gray):
    _, b = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    ink = cv2.findNonZero(255 - b)
    if ink is not None:
        xx, yy, w, h = cv2.boundingRect(ink)
        pad = max(3, int(.05 * max(w, h)))
        gray = gray[
            max(0, yy-pad):min(gray.shape[0], yy+h+pad),
            max(0, xx-pad):min(gray.shape[1], xx+w+pad)
        ]

    h, w = gray.shape
    scale = min(IMAGE_SIZE/max(w,1), IMAGE_SIZE/max(h,1))
    nw = max(1, int(round(w*scale)))
    nh = max(1, int(round(h*scale)))
    r = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, np.uint8)
    x0 = (IMAGE_SIZE-nw)//2
    y0 = (IMAGE_SIZE-nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = r
    return (1.0 - canvas.astype(np.float32)/255.0).astype(np.float32)

def preprocess_pil(img):
    return preprocess_array(np.array(img.convert("L")))

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Encoder().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint: {type(state)}")
    if isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    elif isinstance(state.get("model_state_dict"), dict):
        state = state["model_state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device

@torch.no_grad()
def one_embedding(model, device, img):
    x = torch.from_numpy(preprocess_pil(img)).unsqueeze(0).unsqueeze(0).to(device)
    return model(x).cpu().numpy()[0]

def verify(model, device, refs, query):
    rz = np.stack([one_embedding(model, device, x) for x in refs])
    qz = one_embedding(model, device, query)
    return float(np.linalg.norm(qz - rz.mean(axis=0)))

# ---------- CEDAR dataset evaluation ----------
class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return torch.from_numpy(preprocess_array(
            cv2.imread(str(r.path), cv2.IMREAD_GRAYSCALE)
        )), int(r.label)

@torch.no_grad()
def embed_dataset(model, device, df):
    dl = DataLoader(
        ImageDataset(df), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    out = []
    bar = st.progress(0, "Extracting embeddings…")
    total = max(1, len(dl))
    for i, (x, _) in enumerate(dl, 1):
        out.append(model(x.to(device)).cpu().numpy())
        bar.progress(i/total, f"Extracting embeddings… {i}/{total}")
    bar.empty()
    return np.concatenate(out)

def cedar_manifest(root):
    root = Path(root)
    orgs = list(root.rglob("full_org"))
    forgs = list(root.rglob("full_forg"))
    if not orgs or not forgs:
        raise ValueError("Expected CEDAR folders full_org and full_forg were not found.")
    rows = []
    for p in sorted(orgs[0].iterdir()):
        m = re.match(r"original_(\d+)_(\d+)", p.stem)
        if p.suffix.lower() in EXTS and m:
            rows.append({"writer_id":f"cedar_{m.group(1)}","path":str(p),"label":1})
    for p in sorted(forgs[0].iterdir()):
        m = re.match(r"forgeries_(\d+)_(\d+)", p.stem)
        if p.suffix.lower() in EXTS and m:
            rows.append({"writer_id":f"cedar_{m.group(1)}","path":str(p),"label":0})
    if not rows:
        raise ValueError("No CEDAR signature images found.")
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["_orig_idx"] = np.arange(len(df))
    return df

def make_ref_query(df, nref):
    rng = np.random.default_rng(SEED)
    refs, queries = [], []
    for writer, g in df.groupby("writer_id"):
        gen = g[g.label == 1].copy()
        forg = g[g.label == 0].copy()
        if len(gen) <= nref:
            continue
        idx = np.array(gen.index)
        rng.shuffle(idx)
        refs.append(gen.loc[idx[:nref]])
        queries.append(gen.loc[idx[nref:]])
        if len(forg):
            queries.append(forg)
    return pd.concat(refs, ignore_index=True), pd.concat(queries, ignore_index=True)

def dataset_distances(ref_df, ref_z, query_df, query_z):
    centers = {}
    for writer, g in ref_df.groupby("writer_id"):
        centers[writer] = ref_z[g.index.to_numpy()].mean(axis=0)
    d = np.array([
        np.linalg.norm(z - centers[w])
        for z, w in zip(query_z, query_df.writer_id)
    ])
    return d, query_df.label.to_numpy()

def metric_row(d, y, threshold, nref):
    pred = (d <= threshold).astype(int)
    genuine = y == 1
    forged = y == 0
    return {
        "References": nref,
        "Accuracy": accuracy_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "FAR": ((pred == 1) & forged).sum()/max(1, forged.sum()),
        "FRR": ((pred == 0) & genuine).sum()/max(1, genuine.sum()),
        "AUC": roc_auc_score(y, -d),
        "Threshold": threshold,
        "Samples": len(y),
    }

@st.cache_data(show_spinner=False)
def extract_zip(data):
    work = Path(tempfile.mkdtemp(prefix="cedar_"))
    zpath = work/"dataset.zip"
    zpath.write_bytes(data)
    with zipfile.ZipFile(zpath) as z:
        base = work.resolve()
        for m in z.infolist():
            target = (work/m.filename).resolve()
            if not str(target).startswith(str(base)):
                raise ValueError("Unsafe ZIP path.")
        z.extractall(work/"extracted")
    return str(work/"extracted")

def repo_cedar():
    candidates = [
        Path(__file__).resolve().parent/"data"/"CEDAR"/"CEDAR.zip",
        Path(__file__).resolve().parent/"data"/"CEDAR.zip",
    ]
    return next((p for p in candidates if p.exists()), None)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <span class="badge">AI SIGNATURE VERIFICATION • RESNET-50</span>
  <h1>SignatureVerify AI</h1>
  <p>Interactive signature verification and zero-shot dataset evaluation in one responsive interface.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ✍️ SignatureVerify AI")
    st.caption("ResNet-50 • 256-D normalized embeddings")
    st.divider()
    mode = st.radio(
        "Mode",
        ["🔎 Signature Verification", "📊 Dataset Evaluation"]
    )
    st.divider()
    st.caption("224 × 224 grayscale preprocessing")
    st.caption("Reference protocols: 2 / 3 / 6 / 10")
    st.caption("CPU/GPU auto-detection")

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ============================================================
# OPTION 1 — LIVE SIGNATURE VERIFICATION
# ============================================================
if mode == "🔎 Signature Verification":
    st.markdown("## Verify a signature")
    st.markdown("""
    <div class="card">
      <h3>Compare a query against genuine references</h3>
      <div class="muted">
      Upload genuine signatures from the same writer as references,
      then upload one query signature. The query is compared with the
      mean reference embedding using Euclidean distance.
      </div>
    </div>
    """, unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        nref = st.selectbox("Number of reference signatures", REF_COUNTS)
    with b:
        st.metric("Threshold", f"{THRESHOLDS[nref]:.6f}")

    refs = st.file_uploader(
        f"Upload exactly {nref} genuine reference signatures",
        type=["png","jpg","jpeg","bmp","tif","tiff"],
        accept_multiple_files=True,
        key="refs"
    )
    query = st.file_uploader(
        "Upload query signature",
        type=["png","jpg","jpeg","bmp","tif","tiff"],
        key="query"
    )

    if refs:
        st.markdown("### References")
        cols = st.columns(min(5, len(refs)))
        for i, f in enumerate(refs):
            with cols[i % len(cols)]:
                st.image(Image.open(f), caption=f"Reference {i+1}", use_container_width=True)

    if query:
        st.markdown("### Query")
        st.image(Image.open(query), caption="Query signature", width=320)

    if st.button("🔍 Verify Signature", type="primary", use_container_width=True):
        if len(refs or []) != nref:
            st.error(f"Please upload exactly {nref} reference signatures.")
            st.stop()
        if query is None:
            st.error("Please upload a query signature.")
            st.stop()

        with st.spinner("Running ResNet-50 verification…"):
            ref_images = [Image.open(x).convert("RGB") for x in refs]
            query_image = Image.open(query).convert("RGB")
            distance = verify(model, device, ref_images, query_image)

        threshold = THRESHOLDS[nref]
        genuine = distance <= threshold

        if genuine:
            st.markdown(
                '<div class="good"><div class="result">✅ GENUINE</div>'
                '<div class="muted">Distance is at or below the configured threshold.</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="bad"><div class="result">❌ FORGED</div>'
                '<div class="muted">Distance is above the configured threshold.</div></div>',
                unsafe_allow_html=True
            )

        c1,c2,c3 = st.columns(3)
        c1.metric("Distance", f"{distance:.6f}")
        c2.metric("Threshold", f"{threshold:.6f}")
        c3.metric("References", nref)

# ============================================================
# OPTION 2 — DATASET EVALUATION
# ============================================================
else:
    st.markdown("## Dataset evaluation")
    st.markdown("""
    <div class="card">
      <h3>Zero-shot CEDAR benchmark</h3>
      <div class="muted">
      This follows the evaluation protocol in the uploaded ResNet-50 notebook:
      thresholds are taken from Akash + Robin validation and CEDAR is used
      for the zero-shot evaluation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Full dataset evaluation is much heavier than single-signature testing. "
        "For a full benchmark, local/Colab execution is recommended over a free web CPU."
    )

    source = st.radio(
        "Dataset source",
        ["Repository CEDAR.zip", "Upload CEDAR.zip"],
        horizontal=True
    )

    uploaded = None
    if source == "Upload CEDAR.zip":
        uploaded = st.file_uploader(
            "Upload CEDAR.zip",
            type=["zip"],
            key="cedar"
        )
    else:
        p = repo_cedar()
        if p:
            st.success(f"Found: {p.as_posix()}")
        else:
            st.warning("data/CEDAR/CEDAR.zip was not found. Upload it instead.")

    counts = st.multiselect(
        "Reference counts",
        REF_COUNTS,
        default=REF_COUNTS
    )

    if st.button("▶ Run Dataset Evaluation", type="primary", use_container_width=True):
        if not counts:
            st.error("Select at least one reference count.")
            st.stop()

        try:
            if source == "Upload CEDAR.zip":
                if uploaded is None:
                    st.error("Upload CEDAR.zip first.")
                    st.stop()
                root = extract_zip(uploaded.getvalue())
            else:
                p = repo_cedar()
                if p is None:
                    st.error("Repository CEDAR.zip not found.")
                    st.stop()
                root = extract_zip(p.read_bytes())

            df = cedar_manifest(root)
            st.success(
                f"C E D A R loaded — {len(df):,} images / "
                f"{df.writer_id.nunique()} writers"
            )

            # Embed each CEDAR image only once.
            z = embed_dataset(model, device, df)

            rows = []
            for nref in counts:
                with st.spinner(f"Evaluating {nref}-reference protocol…"):
                    ref_df, query_df = make_ref_query(df, nref)
                    ref_z = z[ref_df["_orig_idx"].to_numpy()]
                    query_z = z[query_df["_orig_idx"].to_numpy()]
                    d, y = dataset_distances(ref_df, ref_z, query_df, query_z)
                    rows.append(metric_row(d, y, THRESHOLDS[nref], nref))

            result = pd.DataFrame(rows)

            st.markdown("### Results")
            cols = st.columns(len(rows))
            for col, row in zip(cols, rows):
                with col:
                    st.markdown(f"**{row['References']} references**")
                    st.metric("Accuracy", f"{row['Accuracy']:.2%}")
                    st.metric("Balanced Accuracy", f"{row['Balanced Accuracy']:.2%}")

            display_df = result.copy()
            for c in ["Accuracy","Balanced Accuracy","FAR","FRR"]:
                display_df[c] = display_df[c].map(lambda x:f"{x:.2%}")
            display_df["AUC"] = display_df["AUC"].map(lambda x:f"{x:.4f}")
            display_df["Threshold"] = display_df["Threshold"].map(lambda x:f"{x:.6f}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇ Download evaluation CSV",
                result.to_csv(index=False).encode("utf-8"),
                "resnet50_signature_evaluation.csv",
                "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Dataset evaluation failed: {e}")
            st.exception(e)

st.divider()
st.caption(
    "SignatureVerify AI • ResNet-50 • 224×224 preprocessing • "
    "256-D normalized embeddings"
)
