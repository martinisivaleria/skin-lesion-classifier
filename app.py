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
ood = np.load('stats_final.npz')
medie_classi = ood['medie_classi']
precisione = ood['precisione']
soglia_ood = float(ood['soglia'])

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

# ----------------- Calcolo della distanza tra le feature dell'immagine input e le medie salvate -----------------
def distanza_ood(image_t):
    with torch.no_grad():
        feat = model.image_backbone(image_t).numpy()     # (1, 1536)
    diff = feat - medie_classi                           # (8, 1536): differenza da ciascun profilo
    distanze = ((diff @ precisione) * diff).sum(axis=1)  # distanza di Mahalanobis da ciascuna classe
    return distanze.min()

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


## ---------- Interfaccia ----------

# ---------- Dizionari -----------

NOMI_DIAGNOSI = {
    'Actinic Keratosis': 'Cheratosi attinica',
    'Basal Cell Carcinoma': 'Carcinoma basocellulare',
    'Benign Keratosis': 'Cheratosi benigna',
    'Dermatofibroma': 'Dermatofibroma',
    'Melanocytic Nevus': 'Nevo melanocitico (neo)',
    'Melanoma': 'Melanoma',
    'Squamous Cell Carcinoma': 'Carcinoma squamocellulare',
    'Vascular Lesion': 'Lesione vascolare',
}

NOMI_SEDE = {
    'Head and neck': 'Testa e collo',
    'Trunk': 'Tronco (petto, addome, schiena)',
    'Upper extremity': 'Braccia e mani',
    'Lower extremity': 'Gambe e piedi',
    'Anogenital region': 'Regione anogenitale',
    'NaN': 'Non specificato',
}

NOMI_SESSO = {
    'female': 'Donna',
    'male': 'Uomo',
    'NaN': 'Non specificato',
}

INFO_PROGETTO = """
#### 🎯 Obiettivo 
Classificare le lesioni cutanee in 8 categorie con un modello **multimodale**, che combina
l'immagine dermoscopica con i dati clinici del paziente (età, sesso e sede anatomica), come
avviene nella valutazione medica. Diversi studi mostrano che integrare i dati clinici migliora
la classificazione rispetto all'uso della sola immagine.

#### 🗂️ Dati 
- Per l'allenamento del modello sono state utilizzate **11.720 immagini dermoscopiche** della collezione pubblica HAM10000 (ISIC Archive),
  relative a **8 lesioni** distinte, complete di metadati clinici dei pazienti.
- **Split raggruppato per lesione** (70% training, 15% validation, 15% test): nel dataset la stessa lesione compare spesso in più foto, in questo studio
  tutte le foto della stessa lesione finiscono nello stesso insieme, ciò ha consentito di valutare il modello solo su
  lesioni mai viste. Molti studi basati sullo stesso dataset dividono per singola immagine, con il rischio che
  la stessa lesione compaia sia in training sia in test (*data leakage*).
- **Dati mancanti**: età imputata con la mediana del training; per sesso e sede è stata definita una categoria
  esplicita "non specificato", distinta da quella riservata ai valori mai visti in training.
- **Classi fortemente sbilanciate**: i nevi sono circa due terzi delle immagini, mentre
  classi come dermatofibroma e lesioni vascolari ne hanno poche centinaia.

#### 🧪 Metodo
- **Ramo immagine**: è stata utilizzata una rete EfficientNet-B3 pre-addestrata su ImageNet e ri-addestrata sulle
  immagini del dataset (transfer learning), con immagini a 300×300 pixel.
- **Ramo clinico**: sesso e sede sono stati codificati con *embedding* appresi, l'età è stata normalizzata (Z-score);
  una piccola rete li trasforma in un vettore di 64 valori.
- **Fusione e classificazione**: le due rappresentazioni vengono concatenate e passate a
  una rete di classificazione a tre strati.
- **Gestione dello sbilanciamento**: in addestramento le classi rare vengono campionate più
  spesso (*weighted sampling*); il modello migliore non è stato scelto tramite l'accuratezza, ma in base all'**F1 macro**, che dà lo
  stesso peso a tutte le classi.
- **Esperimenti controllati**: è stato fatto un confronto sequenziale tra i vari parametri che influenzavano le performance del modello: il modo
  di combinare immagine e dati clinici, la risoluzione delle immagini, la funzione di errore e
  l'intensità della data augmentation, al fine di scegliere la combinazione più performante.

#### 🔍 Trasparenza
- **Grad-CAM** mostra le zone dell'immagine che hanno influenzato maggiormente la predizione.
- Per rendere l'applicazione più efficiente è stato implementato un **filtro di coerenza** basato sulla distanza di Mahalanobis nello spazio delle feature che
  rifiuta le immagini troppo diverse da quelle di training. Sul test set accetta il 93,8%
  delle dermoscopie reali; resta però un filtro di base, che non intercetta tutte le immagini estranee.
"""


INFO_LESIONI = """
| Lesione | Natura | In breve |
|---|---|---|
| Nevo melanocitico (neo) | Benigna | Il comune neo: un accumulo di cellule che producono pigmento. |
| Cheratosi benigna | Benigna | Macchie o rilievi della pelle frequenti con l'età. |
| Dermatofibroma | Benigna | Piccolo nodulo duro della pelle. |
| Lesione vascolare | Benigna | Lesioni formate da piccoli vasi sanguigni. |
| Cheratosi attinica | Precancerosa | Dovuta all'esposizione al sole; se non trattata può evolvere in carcinoma squamocellulare. |
| Carcinoma basocellulare | Maligna | Il tumore della pelle più frequente; cresce lentamente e raramente si diffonde, ma va trattato. |
| Carcinoma squamocellulare | Maligna | Tumore della pelle che in alcuni casi può diffondersi ad altri organi. |
| Melanoma | Maligna | Il più aggressivo tra questi tumori; una diagnosi precoce è fondamentale. |

*Descrizioni generali a scopo informativo: non sostituiscono il parere di un medico.*
"""

INFO_GRADCAM = """
**Grad-CAM** è una tecnica che mostra *dove ha guardato* il modello per prendere la sua decisione.

I colori sovrapposti all'immagine indicano quanto ogni zona ha influito sulla predizione:
**rosso** molto, **giallo e verde** in modo moderato, **blu** poco o nulla.

Serve a capire se il modello si è concentrato sulla lesione o su dettagli irrilevanti
(peli, bordi della foto, riflessi). Non indica dove si trova un eventuale tumore:
mostra solo il ragionamento del modello.
"""

st.title("Classificatore multimodale di lesioni cutanee")
st.info(
    "Carica un'immagine dermoscopica di una lesione della pelle e, se li conosci, inserisci età, "
    "sesso e sede della lesione. Premendo **Analizza**, il modello stima a quale tipo di lesione "
    "appartiene.\n\n"
    "⚠️ **Attenzione:** si tratta di un prototipo sviluppato a scopo di ricerca, "
    "non rappresenta un dispositivo medico. I risultati non costituiscono una diagnosi. Per qualsiasi dubbio rivolgersi sempre "
    "a un dermatologo."
)

info1, info2 = st.columns(2)
with info1.expander("Informazioni sul progetto"):
    st.markdown(INFO_PROGETTO)
with info2.expander("Le lesioni che il modello riconosce"):
    st.markdown(INFO_LESIONI)

opzioni_sesso = [s for s in sex_to_idx if s != '__unseen__']
opzioni_sede = [s for s in site_to_idx if s != '__unseen__']

col_input, col_output = st.columns(2)

with col_input:
    foto = st.file_uploader("Carica un'immagine dermoscopica", type=['jpg', 'jpeg', 'png'])
    eta = st.number_input("Età (lascia vuoto se non nota)", min_value=0, max_value=100, value=None, step=5)
    sesso = st.selectbox("Sesso", opzioni_sesso, format_func=lambda v: NOMI_SESSO.get(v, v))
    sede = st.selectbox("Sede anatomica", opzioni_sede, index=opzioni_sede.index('NaN'), format_func=lambda v: NOMI_SEDE.get(v, v))
    avvia = st.button("Analizza", disabled=foto is None)

with col_output:
    if avvia:
        immagine = Image.open(foto).convert('RGB')
        image_t = eval_transform(immagine).unsqueeze(0)

        if distanza_ood(image_t) > soglia_ood:
            st.error(
                "L'immagine caricata non sembra un'immagine dermoscopica, non è possibile procedere con la classificazione. "
                "Il modello è stato addestrato solo su immagini dermoscopiche di lesioni cutanee."
            )
            st.stop()

        with st.spinner("Analisi in corso..."):
            classe, probabilita, overlay = predict(immagine, eta, sesso, sede)

        st.subheader(f"Classe predetta: {NOMI_DIAGNOSI.get(classe, classe)}")
        c1, c2 = st.columns(2)
        c1.image(immagine.resize((IMG_SIZE, IMG_SIZE)), caption="Immagine caricata")
        c2.image(overlay, caption="Grad-CAM: zone più rilevanti per la predizione", clamp=True)
        with c2.popover("Cos'è Grad-CAM?"):
            st.markdown(INFO_GRADCAM)

        st.markdown("**Probabilità per classe**")
        for nome, p in sorted(probabilita.items(), key=lambda x: -x[1]):
            st.progress(p, text=f"{NOMI_DIAGNOSI.get(nome, nome)}: {p:.1%}")
