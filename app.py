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

import base64
import io

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


# ---------- Componente di caricamento robusto per smartphone ----------

caricatore = st.components.v2.component(
    "caricatore_immagine",
    html="""
    <div class="scheda" id="scheda">
      <div class="vuoto" id="vuoto">
        <svg class="icona" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M4 16.5v2A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-2"/>
          <path d="M12 15V4"/><path d="M7.5 8.5 12 4l4.5 4.5"/>
        </svg>
        <div class="titolo">Carica un'immagine dermoscopica</div>
        <div class="sottotitolo">JPG o PNG, dalla galleria o dai file</div>
      </div>
      <div class="pieno" id="pieno" hidden>
        <img id="anteprima" alt="Anteprima dell'immagine caricata" />
        <div class="info">
          <div class="nome" id="nome"></div>
          <span class="stato" id="stato"></span>
        </div>
      </div>
      <button id="scegli" type="button">Scegli immagine</button>
      <input id="file" type="file" accept="image/*" hidden />
    </div>
    """,
    css="""
    .scheda {
      font-family: var(--st-font);
      color: var(--st-text-color);
      border: 2px dashed color-mix(in srgb, var(--st-text-color) 25%, transparent);
      border-radius: 14px;
      padding: 1.4rem 1.2rem;
      text-align: center;
      background: var(--st-secondary-background-color, rgba(128,128,128,0.06));
      transition: border-color .2s ease;
    }
    .scheda:hover { border-color: var(--st-primary-color); }
    .icona { width: 44px; height: 44px; color: var(--st-primary-color); margin-bottom: .4rem; }
    .titolo { font-weight: 600; font-size: 1.05rem; }
    .sottotitolo { font-size: .85rem; opacity: .7; margin-top: .2rem; }
    .pieno { display: flex; align-items: center; gap: 1rem; text-align: left; }
    .pieno[hidden], .vuoto[hidden] { display: none; }
    #anteprima {
      width: 84px; height: 84px; object-fit: cover; border-radius: 10px;
      box-shadow: 0 1px 6px rgba(0,0,0,.25);
    }
    .info { display: flex; flex-direction: column; gap: .4rem; min-width: 0; }
    .nome { font-weight: 600; font-size: .95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .stato {
      display: inline-block; width: fit-content; font-size: .8rem; font-weight: 600;
      padding: .2rem .6rem; border-radius: 999px;
    }
    .stato.attesa { background: #E8A317; color: #fff; }
    .stato.ok     { background: #2E9E5B; color: #fff; }
    .stato.errore { background: #D64545; color: #fff; }
    #scegli {
      margin-top: 1rem;
      background: var(--st-primary-color); color: #fff; border: none;
      border-radius: 999px; padding: .6rem 1.4rem; font-size: .95rem; font-weight: 600;
      cursor: pointer; transition: transform .1s ease, opacity .2s ease;
    }
    #scegli:hover { opacity: .9; }
    #scegli:active { transform: scale(.97); }
    """,
    js="""
    export default function ({ parentElement, data, setStateValue }) {
      const CHIAVE = "lesione_in_attesa";
      const el = parentElement;
      el._ack = data?.ack ?? null;
      el._set = setStateValue;

      const vuoto = el.querySelector("#vuoto");
      const pieno = el.querySelector("#pieno");
      const anteprima = el.querySelector("#anteprima");
      const nome = el.querySelector("#nome");
      const stato = el.querySelector("#stato");
      const bottone = el.querySelector("#scegli");

      const mostra = (p, testo, tipo) => {
        if (p) {
          vuoto.hidden = true; pieno.hidden = false;
          anteprima.src = p.data; nome.textContent = p.name;
          bottone.textContent = "Cambia immagine";
        }
        stato.textContent = testo; stato.className = "stato " + tipo;
      };

      const leggiAttesa = () => {
        try { return JSON.parse(sessionStorage.getItem(CHIAVE)) || el._attesa || null; }
        catch (e) { return el._attesa || null; }
      };

      el._invia = () => {
        const p = leggiAttesa();
        if (!p || p.id === el._ack) return;
        el._tentativi = el._tentativi || {};
        const n = (el._tentativi[p.id] || 0) + 1;
        if (n > 5) { mostra(p, "Invio non riuscito: ricarica la pagina", "errore"); return; }
        el._tentativi[p.id] = n;
        mostra(p, "Caricamento…", "attesa");
        el._set("image", { ...p, tentativo: n });
      };

      const p = leggiAttesa();
      if (p && p.id === el._ack) mostra(p, "Immagine caricata", "ok");

      if (!el._init) {
        el._init = true;
        const input = el.querySelector("#file");

        function leggiERidimensiona(file, maxLato) {
          return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
              const img = new Image();
              img.onload = () => {
                const scala = Math.min(1, maxLato / Math.max(img.width, img.height));
                const c = document.createElement("canvas");
                c.width = Math.round(img.width * scala);
                c.height = Math.round(img.height * scala);
                c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
                resolve(c.toDataURL("image/jpeg", 0.92));
              };
              img.onerror = () => reject(new Error("formato non leggibile"));
              img.src = reader.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
        }

        bottone.onclick = () => input.click();
        input.onchange = async () => {
          const f = input.files && input.files[0];
          if (!f) return;
          try {
            const url = await leggiERidimensiona(f, 800);
            const nuova = { id: String(Date.now()), name: f.name, data: url };
            el._attesa = nuova;
            try { sessionStorage.setItem(CHIAVE, JSON.stringify(nuova)); } catch (e) {}
            el._invia();
          } catch (e) {
            mostra(null, "File non leggibile: usa un'immagine JPG o PNG", "errore");
            vuoto.hidden = false; pieno.hidden = false;
          }
          input.value = "";
        };

        el._visibile = () => {
          if (document.visibilityState === "visible") setTimeout(() => el._invia(), 1500);
        };
        document.addEventListener("visibilitychange", el._visibile);
        el._timer = setInterval(() => el._invia(), 4000);

        if (p && p.id !== el._ack) el._invia();
      }

      return () => {
        document.removeEventListener("visibilitychange", el._visibile);
        clearInterval(el._timer);
        el._init = false;
      };
    }
    """,
)

def carica_immagine():
    """Mostra il componente e restituisce l'immagine PIL ricevuta (o None)."""
    risultato = caricatore(
        key="caricatore",
        data={"ack": st.session_state.get("img_id")},
        default={"image": None},
        on_image_change=lambda: None,
    )
    valore = risultato.image
    if valore and valore.get("id") != st.session_state.get("img_id"):
        st.session_state["img_id"] = valore["id"]
        st.session_state["img_bytes"] = base64.b64decode(valore["data"].split(",", 1)[1])
        st.rerun()   # rilancio così il componente riceve la conferma (ack)
    if "img_bytes" in st.session_state:
        return Image.open(io.BytesIO(st.session_state["img_bytes"])).convert("RGB")
    return None
    
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
  tutte le foto della stessa lesione finiscono nello stesso set, ciò ha consentito di valutare il modello solo su
  lesioni mai viste. Molti studi basati sullo stesso dataset dividono per singola immagine, con il rischio che
  la stessa lesione compaia sia in training sia in test (*data leakage*).
- **Dati mancanti**: età imputata con la mediana del training; per sesso e sede è stata definita una categoria
  esplicita "non specificato".
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
**rosso** - molto, **giallo e verde** - in modo moderato, **blu** - poco o nulla.

Serve a capire se il modello si è concentrato sulla lesione o su dettagli irrilevanti
(peli, bordi della foto, riflessi). 
"""
COMMENTO_CM = """
**Cosa emerge**
- **Le lesioni più comuni o più caratteristiche sono riconosciute bene**: nevo melanocitico (93%),
  carcinoma basocellulare (91%) e dermatofibroma (80%).
- **Il melanoma è riconosciuto nel 71% dei casi**, ma nel 20% viene scambiato per un nevo.
  È l'errore più rilevante dal punto di vista clinico, in quanto una lesione maligna viene interpretata come
  benigna. L'errore opposto è più contenuto: solo il 5% dei nevi viene classificato come melanoma.
- **Le lesioni cheratinocitiche sono le più difficili.** La cheratosi attinica è riconosciuta solo
  nel 36% dei casi e viene confusa soprattutto con la cheratosi benigna (20%) e con il carcinoma
  squamocellulare (20%). Il carcinoma squamocellulare (51%) viene a sua volta confuso con la
  cheratosi attinica (14%). Sono lesioni visivamente molto simili, e la cheratosi attinica può
  evolvere proprio in carcinoma squamocellulare: la confusione riflette una vicinanza reale.
- **Diverse lesioni rare vengono attratte verso il nevo** (dermatofibroma 16%, lesione vascolare 12%),
  la classe più numerosa del dataset.

**Nota sui numeri.** Alcune classi hanno pochissime immagini nel test set (25 per cheratosi attinica,
dermatofibroma e lesione vascolare, 35 per il carcinoma squamocellulare): per queste una singola
immagine vale circa 3-4 punti percentuali, quindi le percentuali vanno lette con cautela.
"""

COMMENTO_CONTRIBUTO_CLINICI = """
**Verifica sui dati clinici.** Per verificare che il modello usi davvero i dati clinici è stata condotta un'analisi di ablazione: il modello 
finale è stato rivalutato due volte sugli stessi dati, una con i dati clinici veri e una con età, sesso e sede impostati come dato mancante, 
sia sul validation sia sul test set. 
- In entrambi i casi, senza dati clinici, F1 macro, accuratezza e AUC peggiorano: l'F1 macro perde 3,7 punti in 
validation e 1,7 nel test. Il contributo è quindi coerente nella direzione ma di entità moderata. 
- Classe per classe i guadagni sono stabili 
per dermatofibroma, carcinoma squamocellulare, lesione vascolare e melanoma; per la cheratosi attinica invece il segno cambia tra validation 
e test, perché con 21-25 immagini una o due predizioni spostano il risultato di diversi punti. In circa il 69% delle immagini i dati clinici 
aumentano la probabilità della classe corretta, e il dato è identico nei due insiemi. L'analisi misura quanto il modello finale si appoggia 
ai dati clinici, non quanto renderebbe un modello allenato senza di essi.

"""

TABELLA_RISULTATI = """
| Metrica | Validation | Test |
|---|:---:|:---:|
| Accuratezza | 85,7% | 85,6% |
| F1 macro | 0,724 | 0,720 |
| AUC macro | – | 0,967 |
"""

BIBLIOGRAFIA = """
**Dataset**
- Tschandl P., Rosendahl C., Kittler H. (2018). *The HAM10000 dataset, a large collection of multi-source
  dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.
- ISIC Archive – International Skin Imaging Collaboration, https://www.isic-archive.com

**Studi di riferimento sulla classificazione multimodale**
- Aksoy S., Demircioglu P., Bogrekci I. (2025). *Web-Based Multimodal Deep Learning Platform with XRAI
  Explainability for Real-Time Skin Lesion Classification and Clinical Decision Support.* Cosmetics, 12, 194.
- Das A., Agarwal V., Shetty N. P. (2025). *Comparative analysis of multimodal architectures for effective
  skin lesion detection using clinical and image data.* Frontiers in Artificial Intelligence, 8, 1608837.
- Tran-Van N.-Y., Le K.-H. (2025). *A multimodal skin lesion classification through cross-attention fusion
  and collaborative edge computing.* Computerized Medical Imaging and Graphics, 124, 102588.
- Suresh P., Keerthika P., Nitesh Kumar A. R. (2026). *Text guided cross attentive multimodal learning with
  visual feature modulation for automated skin lesion detection.* Scientific Reports.
- Atiq M. E., Fattah S. A. (2025). *Towards Explainable Skin Cancer Classification: A Dual-Network Attention
  Model with Lesion Segmentation and Clinical Metadata Fusion.* arXiv:2510.17773.
- Adebiyi A. et al. (2024). *Accurate Skin Lesion Classification Using Multimodal Learning on the
  HAM10000 Dataset.* medRxiv, doi:10.1101/2024.05.30.24308213.
  """

st.title("Classificatore multimodale di lesioni cutanee")
st.info(
    "Carica l'immagine di una lesione della pelle e, se li conosci, inserisci età, "
    "sesso e sede della lesione. Premendo **Analizza**, il modello stima a quale tipo di lesione "
    "appartiene.\n\n"
    "⚠️ **Attenzione:** si tratta di un prototipo sviluppato a scopo di ricerca, "
    "non rappresenta un dispositivo medico. I risultati non costituiscono una diagnosi. Per qualsiasi dubbio rivolgersi sempre "
    "a un dermatologo."
)



opzioni_sesso = [s for s in sex_to_idx if s != '__unseen__']
opzioni_sede = [s for s in site_to_idx if s != '__unseen__']

col_input, col_output = st.columns(2)

with col_input:
    immagine_caricata = carica_immagine()
    eta = st.number_input("Età (lascia vuoto se non nota)", min_value=0, max_value=100, value=None, step=5)
    sesso = st.selectbox("Sesso", opzioni_sesso, format_func=lambda v: NOMI_SESSO.get(v, v))
    sede = st.selectbox("Sede anatomica", opzioni_sede, index=opzioni_sede.index('NaN'), format_func=lambda v: NOMI_SEDE.get(v, v))
    avvia = st.button("Analizza", disabled=immagine_caricata is None)

with col_output:
    if avvia:
        immagine = immagine_caricata
        image_t = eval_transform(immagine).unsqueeze(0)
        if distanza_ood(image_t) > soglia_ood:
            st.warning(
                "L'immagine caricata è lontana dalle immagini "
                "dermoscopiche su cui il modello è stato addestrato: la classificazione viene comunque "
                "mostrata, ma potrebbe essere poco affidabile."
            )

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


with st.expander("Informazioni sul progetto"):
    st.markdown(INFO_PROGETTO)
    
with st.expander("Le lesioni che il modello riconosce"):
    st.markdown(INFO_LESIONI)

with st.expander("Architettura del modello e flusso del lavoro"):
    st.markdown("#### Struttura del modello")
    st.markdown(
        "Il modello elabora in parallelo l'immagine e i dati clinici con due rami separati. "
        "Le due rappresentazioni vengono poi unite e passate a una rete che produce le probabilità "
        "delle 8 classi. Accanto a ogni passaggio è indicato quanti valori lo attraversano."
    )
    st.image("schema_modello.png")
    
with st.expander("Risultati"):
    st.markdown("#### Confusion matrix sul test set")
    st.image("cm_test.png")
    st.markdown(COMMENTO_CM)
    st.markdown("#### Metriche per classe")
    st.image("contributi_clinici.png")
    st.markdown(COMMENTO_CONTRIBUTO_CLINICI)
    st.markdown("#### Riepilogo")
    st.markdown(TABELLA_RISULTATI)
    st.caption("Il test set è stato usato una sola volta, al termine dello sviluppo. I valori quasi identici "
               "a quelli di validation indicano che il modello generalizza bene su lesioni mai viste. "
               "L'AUC è stata calcolata solo sul test set.")

with st.expander("Bibliografia"):
    st.markdown(BIBLIOGRAFIA)

st.divider()
st.caption(
    "Autrice: **Valeria Martinisi** · Project work realizzato per il master: 2025-2026 - Machine Learning e Big Data nella medicina di precisione e la Ricerca Biomedica, Università degli Studi di Padova· "
    "[LinkedIn](https://www.linkedin.com/in/valeria-martinisi-7349b4203/)"
)
