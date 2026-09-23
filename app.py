import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from matplotlib import cm
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")


# ---------- Definizioni del modello ----------

def build_image_backbone(backbone_name):
    backbone = models.efficientnet_b3(weights=None)
    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return backbone, feature_dim


def fastai_emb_dim(n_categories):
    return min(600, round(1.6 * n_categories ** 0.56))


class ClinicalEncoder(nn.Module):
    def __init__(self, n_sex, n_site, output_dim=64):
        super().__init__()
        emb_dim_sex = fastai_emb_dim(n_sex)
        emb_dim_site = fastai_emb_dim(n_site)
        self.sex_embedding = nn.Embedding(n_sex, emb_dim_sex)
        self.site_embedding = nn.Embedding(n_site, emb_dim_site)
        self.age_bn = nn.BatchNorm1d(1)
        concat_dim = emb_dim_sex + emb_dim_site + 1
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

    def forward(self, age, sex_idx, site_idx):
        combined = torch.cat([self.sex_embedding(sex_idx), self.site_embedding(site_idx), self.age_bn(age)], dim=1)
        return self.fc(combined)


class ConcatFusion(nn.Module):
    def __init__(self, image_dim, clinical_dim):
        super().__init__()
        self.output_dim = image_dim + clinical_dim

    def forward(self, image_features, clinical_features):
        return torch.cat([image_features, clinical_features], dim=1)


class MultimodalSkinLesionModel(nn.Module):
    def __init__(self, n_sex, n_site, n_classes, dropout=0.5):
        super().__init__()
        self.image_backbone, image_dim = build_image_backbone('efficientnet_b3')
        self.clinical_encoder = ClinicalEncoder(n_sex=n_sex, n_site=n_site)
        self.fusion = ConcatFusion(image_dim, self.clinical_encoder.output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, n_classes),
        )

    def forward(self, image, age, sex_idx, site_idx):
        image_features = self.image_backbone(image)
        clinical_features = self.clinical_encoder(age, sex_idx, site_idx)
        return self.classifier(self.fusion(image_features, clinical_features))


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, age, sex_idx, site_idx):
        output = self.model(image, age, sex_idx, site_idx)
        target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ---------- Artefatti di preprocessing e modello ----------

with open('preprocessing_artifacts_final.json') as f:
    artefatti = json.load(f)

class_to_idx = artefatti['class_to_idx']
sex_to_idx = artefatti['sex_to_idx']
site_to_idx = artefatti['site_to_idx']
age_mean = artefatti['age_mean']
age_std = artefatti['age_std']
IMG_SIZE = artefatti['img_size']
class_names = [c for c, i in sorted(class_to_idx.items(), key=lambda x: x[1])]

device = torch.device('cpu')

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def carica_modello():
    percorso = hf_hub_download(
        repo_id='valeriamartinisi/skin-lesion-classification',
        filename='best_model_concat_final.pt',
    )
    m = MultimodalSkinLesionModel(
        n_sex=len(sex_to_idx),
        n_site=len(site_to_idx),
        n_classes=len(class_to_idx),
    )
    m.load_state_dict(torch.load(percorso, map_location=device))
    m.eval()
    return m


model = carica_modello()


# ---------- Preprocessing e predizione ----------

def prepara_dati_clinici(age, sex, site):
    age_norm = 0.0 if age is None else (age - age_mean) / age_std
    sex_i = sex_to_idx.get(sex, sex_to_idx['__unseen__'])
    site_i = site_to_idx.get(site, site_to_idx['__unseen__'])
    age_t = torch.tensor([[age_norm]], dtype=torch.float32)
    sex_t = torch.tensor([sex_i], dtype=torch.long)
    site_t = torch.tensor([site_i], dtype=torch.long)
    return age_t, sex_t, site_t


def predict(pil_image, age, sex, site):
    pil_image = pil_image.convert('RGB')
    image_t = eval_transform(pil_image).unsqueeze(0)
    age_t, sex_t, site_t = prepara_dati_clinici(age, sex, site)

    with torch.no_grad():
        probs = F.softmax(model(image_t, age_t, sex_t, site_t), dim=1)[0].numpy()
    probabilita = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    gradcam = GradCAM(model, model.image_backbone.features)
    cam, pred_class = gradcam.generate(image_t, age_t, sex_t, site_t)
    gradcam.remove_hooks()

    img_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap = cm.jet(cv2.resize(cam, (IMG_SIZE, IMG_SIZE)))[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap

    return class_names[pred_class], probabilita, overlay


# ---------- Interfaccia ----------

st.title("Classificazione multimodale di lesioni cutanee")
st.warning(
    "Prototipo di ricerca sviluppato per un project work universitario. "
    "NON è un dispositivo medico e non fornisce diagnosi: per qualsiasi dubbio "
    "su una lesione della pelle rivolgersi a un dermatologo."
)

opzioni_sesso = [s for s in sex_to_idx if s != '__unseen__']
opzioni_sede = [s for s in site_to_idx if s != '__unseen__']

def etichetta(valore):
    return 'Non specificato' if valore == 'NaN' else valore

col_input, col_output = st.columns(2)

with col_input:
    foto = st.file_uploader("Carica un'immagine dermoscopica", type=['jpg', 'jpeg', 'png'])
    eta = st.number_input("Età (lascia vuoto se non nota)", min_value=0, max_value=100, value=None, step=5)
    sesso = st.selectbox("Sesso", opzioni_sesso, format_func=etichetta)
    sede = st.selectbox("Sede anatomica", opzioni_sede, format_func=etichetta)
    avvia = st.button("Analizza", disabled=foto is None)

with col_output:
    if avvia:
        immagine = Image.open(foto)
        with st.spinner("Analisi in corso..."):
            classe, probabilita, overlay = predict(immagine, eta, sesso, sede)

        st.subheader(f"Classe predetta: {classe}")
        c1, c2 = st.columns(2)
        c1.image(immagine.resize((IMG_SIZE, IMG_SIZE)), caption="Immagine caricata")
        c2.image(overlay, caption="Grad-CAM: zone più rilevanti per la predizione", clamp=True)

        st.markdown("**Probabilità per classe**")
        for nome, p in sorted(probabilita.items(), key=lambda x: -x[1]):
            st.progress(p, text=f"{nome}: {p:.1%}")
