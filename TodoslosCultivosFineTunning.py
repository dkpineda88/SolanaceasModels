# -*- coding: utf-8 -*-
"""
Experimento B - un cultivo por ejecución: tomato | potato | bellpepper
Particiones oficiales de PlantSeg: train (85/15 train/valid) y test oficial (sin partir).
Deduplicación pHash dentro de cada partición; el test se depura frente al train.
Orden de clases, orden de canales y preprocesado: del mapping.json del modelo A.
Test evaluado una sola vez por ejecución.
"""
import os, json, csv, base64, hashlib, random, itertools
import numpy as np, pandas as pd, cv2, tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt
from scipy.fft import dctn
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2, resnet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===== Selección de cultivo =====
CROP = "potato"                 # "tomato" | "potato" | "bellpepper" 
RUN  = "B_pool_2026"

# ===== Rutas =====
PS_ROOT    = "D:/DATASETS/Imagenes/Imagenes de Campo/plantseg/plantseg"
METADATA   = os.path.join(PS_ROOT, "Metadata.csv")
JSON_ROOT  = os.path.join(PS_ROOT, "json")
IMG_ROOT   = os.path.join(PS_ROOT, "images")
BASE       = "D:/Proyectos/Solanaceas Modelos/ExperimentoB"
REEVAL_DIR = "D:/Proyectos/Solanaceas Modelos/ExperimentoA/Reevaluacion"   # CONFIRMAR: carpeta que contiene *_reevaluacion/

# ===== Parámetros fijados antes de ver resultados =====
SEED, IMG, BATCH, EPOCHS, LR, PATIENCE, TAU = 42, 256, 8, 30, 1e-5, 5, 0.60
RATIO_TOL, DUP_THR = 0.02, 8
VALID_FRAC_OF_TRAIN = 0.15   # 85/15 del entrenamiento oficial; el test oficial no se parte
TV_ALPHA, TV_BETA, TV_GAMMA = 0.3, 0.7, 1.33      # CONFIRMAR con el entrenamiento de A
LAB_MIN_ACC = 0.90
SOURCES = ["Training", "Test"]
FORCE_ORDER_SEARCH = True

def norm(s):
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

# ===== Configuración por cultivo =====
CROPS = {
    "tomato": dict(
        plant="tomato",
        plantseg_map={"tomato bacterial leaf spot": "bacterialspot",
                      "tomato early blight": "earlyblight",
                      "tomato late blight": "lateblight",
                      "tomato yellow leaf curl virus": "yellowleafcurlvirus"},
        arch="maskrcnn_mobilenet",          # CONFIRMAR: modelo A elegido (Tabla 5)
        model_a="D:/Proyectos/Solanaceas Modelos/ExperimentoA/Tomate/Old models/TMTMaskMobilNet5Class.keras",
        lab_root="D:/DATASETS/Imagenes/Solanaceas/Zonodo/TomatoDisease.v6i.coco",
        cls_candidates=None, seg_default=None, preproc_default=None),
    "potato": dict(
        plant="potato",
        plantseg_map={"potato early blight": "earlyblight",
                      "potato late blight": "lateblight"},
        arch="unet_mobilenet",
        model_a="D:/Proyectos/Solanaceas Modelos/ExperimentoA/Potato/Oldmodels/PotatoUNetMobilenetDTN2.keras",
        lab_root="D:/DATASETS/Imagenes/Solanaceas/Zonodo/PotatoDisease.v7i.coco",
        cls_candidates=["healthy", "earlyblight", "lateblight", "other"], # 4 candidatos para que el script encuentre el orden
        seg_default=["Background", "Healthy", "EarlyBlight", "LateBlight"],
        preproc_default="mobilenet"),
    "bellpepper": dict(
        plant="bell pepper",
        plantseg_map={"bell pepper bacterial spot bing": "bacterialspot",
                      "bell pepper bacterial spot google": "bacterialspot"},
        arch="deeplab_mobilenet",           # CONFIRMAR: modelo A elegido (Tabla 5)
        model_a="D:/Proyectos/Solanaceas Modelos/ExperimentoA/Pepper/Oldmodels/PepDeepkMobilnetDTN.keras",
        lab_root="D:/DATASETS/Imagenes/Solanaceas/Zonodo/BellPepper.v2i.coco",
        cls_candidates=["healthy", "bacterialspot"], # Clases esperadas para pimiento
        seg_default=["Background", "Healthy", "BacterialSpot"], # Canales de segmentación
        preproc_default="mobilenet"),
   
}
C = CROPS[CROP]
# Arriba, junto a SEED
SEED = 42                                                  # split y deduplicación: fijo
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))  
MODEL_A     = C["model_a"]
LAB_ROOT    = C["lab_root"]
RESULT_DIR  = os.path.join(BASE, RUN, f"resultados_{CROP}_seed{TRAIN_SEED}")
MODEL_B_OUT = os.path.join(BASE, RUN, f"{CROP}_B_seed{TRAIN_SEED}.keras")
FLAG        = os.path.join(RESULT_DIR, "TEST_EVALUADO.flag")

DIS_NAMES   = set(C["plantseg_map"].values())
os.makedirs(RESULT_DIR, exist_ok=True)
     # solo inicialización y orden de batches

np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

def load_config():
    """Lee clases, canales y preprocesado del mapping.json del modelo A elegido."""
    mp = os.path.join(REEVAL_DIR, f"{CROP}_reevaluacion", C["arch"], "mapping.json") if C["arch"] else None
    if mp and os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        cls = [norm(x) for x in m["class_names_by_output"]]
        seg = [norm(x) for x in m["seg"]["names_by_channel"]]
        return cls, seg, m["preproc"], mp
    if C["seg_default"] is None:
        raise RuntimeError(f"No se encuentra mapping.json en {mp}")
    seg = [norm(x) for x in C["seg_default"]]  # <--- Aquí aplicamos norm() para que "Background" pase a "background"
    cls = None if C["cls_candidates"] is None else list(C["cls_candidates"])
    return cls, seg, C["preproc_default"], "valores por defecto (verificar)"

_cls0, SEG_CH, PREPROC, CFG_SRC = load_config()
SEG_IDX = {n: i for i, n in enumerate(SEG_CH)}
BG = SEG_IDX["background"]
NSEG = len(SEG_CH)


CLS_OUT = _cls0 if _cls0 else list(C["cls_candidates"])        # True: el orden de clases se buscará con el set de laboratorio

CLS_IDX = {}
DISEASES, PRIORITY = [], []

def set_class_order(order):
    global CLS_OUT, CLS_IDX, DISEASES, PRIORITY
    CLS_OUT = list(order)
    CLS_IDX = {n: i for i, n in enumerate(CLS_OUT)}
    DISEASES = [n for n in CLS_OUT if n in DIS_NAMES]
    PRIORITY = DISEASES + (["healthy"] if "healthy" in CLS_OUT else [])

set_class_order(CLS_OUT)
for n in DIS_NAMES:
    assert n in SEG_IDX, f"'{n}' no está en los canales de segmentación {SEG_CH}"
    assert n in CLS_OUT, f"'{n}' no está en las clases {CLS_OUT}"

# ===== Pérdida =====
def focal_tversky_loss(alpha=TV_ALPHA, beta=TV_BETA, gamma=TV_GAMMA):
    def loss(y_true, y_pred):
        yt = tf.reshape(tf.cast(y_true, tf.float32), [-1, NSEG])
        yp = tf.reshape(tf.cast(y_pred, tf.float32), [-1, NSEG])
        tp = tf.reduce_sum(yt * yp, axis=0)
        fp = tf.reduce_sum((1 - yt) * yp, axis=0)
        fn = tf.reduce_sum(yt * (1 - yp), axis=0)
        ti = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return tf.reduce_mean(tf.pow(1.0 - ti, 1.0 / gamma))
    return loss

# ===== Imagen y preprocesado (el del modelo A, sin CLAHE) =====
def to256(rgb):
    return cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)

def prep_arr(x8):
    x = x8.astype(np.float32)
    if PREPROC in ("mobilenet", "tf_pm1"):
        return mobilenet_v2.preprocess_input(x)
    if PREPROC == "caffe":
        return resnet50.preprocess_input(x)
    raise ValueError(f"Preprocesado desconocido: {PREPROC}")

def index_files(root, exts):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                idx.setdefault(os.path.splitext(fn)[0], os.path.join(dp, fn))
    print(f"Índice {root}: {len(idx)} archivos")
    return idx

JSON_INDEX = index_files(JSON_ROOT, (".json",))
IMG_INDEX  = index_files(IMG_ROOT, (".jpg", ".jpeg", ".png"))

def rasterize(js):
    m = np.zeros((js["imageHeight"], js["imageWidth"]), np.uint8)
    for s in js["shapes"]:
        if s.get("shape_type") != "polygon":
            continue
        pts = np.round(np.array(s["points"], np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m

def decode_image(js, img_path):
    if js.get("imageData"):
        buf = np.frombuffer(base64.b64decode(js["imageData"]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.imread(img_path) if img_path else None

# ===== Pool: PlantSeg Training + Test =====
def load_plantseg_pool():
    df = pd.read_csv(METADATA)
    df["Plant"] = df["Plant"].str.strip()
    df["Disease"] = df["Disease"].str.strip().str.lower()
    t = df[(df.Plant.str.lower() == C["plant"]) & (df["Training/Test"].isin(SOURCES))].copy()
    t["cls"] = t.Disease.map(C["plantseg_map"])
    excluded = int(t.cls.isna().sum())
    t = t[t.cls.notna()].copy()
    t["stem"] = t.Name.str.rsplit(".", n=1).str[0]
    dups = t.Name[t.Name.duplicated()].tolist()
    if dups:
        raise RuntimeError(f"Nombres repetidos entre particiones: {dups[:5]}")
    items, rejected = [], []
    for _, r in t.iterrows():
        jp = JSON_INDEX.get(r.stem)
        if jp is None:
            rejected.append((r.Name, "json no encontrado")); continue
        js = json.load(open(jp, encoding="utf-8"))
        m_full = rasterize(js)
        ratio = float(m_full.mean())
        if abs(ratio - r["Mask ratio"]) > RATIO_TOL:
            rejected.append((r.Name, f"ratio {ratio:.4f} vs CSV {r['Mask ratio']:.4f}")); continue
        bgr = decode_image(js, IMG_INDEX.get(r.stem))
        if bgr is None:
            rejected.append((r.Name, "imagen no decodificada")); continue
        x8 = to256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        m8 = cv2.resize(m_full, (IMG, IMG), interpolation=cv2.INTER_NEAREST) > 0
        mask = np.full((IMG, IMG), BG, np.uint8)     # fondo por defecto
        mask[m8] = SEG_IDX[r.cls]                    # lesión con su canal
        items.append({"file": r.Name, "src": r["Training/Test"], "x8": x8, "m8": mask, "true": r.cls})
    print(f"PlantSeg pool ({CROP}): {len(items)} válidas, {excluded} excluidas por clase, {len(rejected)} rechazadas")
    for name, why in rejected[:10]:
        print("   rechazada:", name, "-", why)
    return items

# ===== Duplicados: pHash invariante a rotación y volteo =====
def phash_all(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    out = []
    for v in [g, np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3),
              g[:, ::-1], g[::-1, :], g.T, np.rot90(g, 2).T]:
        d = dctn(v, norm="ortho")[:8, :8].flatten()[1:]
        out.append(int("".join("1" if x else "0" for x in (d > np.median(d))), 2))
    return out

def dedupe_pool(items, thr=DUP_THR):
    kept, kept_h, removed = [], [], []
    for it in items:
        h = phash_all(it["x8"])
        if any(bin(h[0] ^ r).count("1") <= thr for hs in kept_h for r in hs):
            removed.append(it)
        else:
            kept.append(it); kept_h.append(h)
    by_src = {s: sum(x["src"] == s for x in removed) for s in SOURCES}
    print(f"Deduplicación del pool (distancia <= {thr}): {len(removed)} eliminadas {by_src}, {len(kept)} conservadas")
    return kept, removed

def cross_remove(test_items, train_items, thr=DUP_THR):
    """Quita del test oficial las imágenes casi duplicadas de cualquier imagen de entrenamiento."""
    train_h = [phash_all(it["x8"]) for it in train_items]
    kept, removed = [], []
    for it in test_items:
        h = phash_all(it["x8"])
        if any(bin(h[0] ^ r).count("1") <= thr for hs in train_h for r in hs):
            removed.append(it)
        else:
            kept.append(it)
    print(f"Test oficial vs train (distancia <= {thr}): {len(removed)} eliminadas, {len(kept)} conservadas")
    return kept, removed

def count_table(items):
    return {n: sum(x["true"] == n for x in items) for n in CLS_OUT}

# ===== Laboratorio (verificación de orden de clases; no entrena) =====
def lab_items(root, split="valid"):
    d = os.path.join(root, split)
    coco = COCO(os.path.join(d, "_annotations.coco.json"))
    cat2 = {c["id"]: norm(c["name"]) for c in coco.loadCats(coco.getCatIds())}
    out = []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        names = {cat2[a["category_id"]] for a in anns}
        lab = next((c for c in PRIORITY if c in names), None)
        bgr = cv2.imread(os.path.join(d, info["file_name"]))
        if lab is None or bgr is None:
            continue
        out.append({"x8": to256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), "true": lab})
    return out

def search_cls_order(model, lab):
    """Solo si no hay mapping.json: prueba todas las permutaciones de las clases sobre laboratorio."""
    ci = model.output_names.index("class_out")
    cands = C["cls_candidates"]
    probs = np.array([model.predict(prep_arr(it["x8"])[None], verbose=0)[ci][0] for it in lab])
    y = np.array([it["true"] for it in lab])
    
    scored = sorted(
        (
            ((np.array(p)[probs.argmax(1)] == y).mean(), p)
            for p in itertools.permutations(cands)
        ),
        reverse=True
    )
    
    print("Búsqueda de orden de clases (laboratorio):")
    for acc, p in scored[:3]:
        print(f"   {acc*100:6.2f}%  {list(p)}")
    best, second = scored[0][0], scored[1][0]
    if best < LAB_MIN_ACC or best - second < 0.02:
        raise RuntimeError("No hay un orden de clases claro. Revisa el modelo A o su mapping.json.")
    return list(scored[0][1])

# ===== Modelo =====
def check_outputs(model):
    names = model.output_names
    assert {"mask_out", "class_out"} <= set(names), names
    m_ch = int(model.outputs[names.index("mask_out")].shape[-1])
    c_ch = int(model.outputs[names.index("class_out")].shape[-1])
    print(f"mask_out: {m_ch} canales (esperados {NSEG}); class_out: {c_ch} unidades")
    assert m_ch == NSEG, f"mask_out tiene {m_ch} canales; esperados {SEG_CH}"
    assert c_ch == len(CLS_OUT), f"class_out tiene {c_ch} unidades; esperadas {len(CLS_OUT)}"
    print(f"Modelo A: {model.count_params()/1e6:.3f} M parámetros")

class SegSeq(tf.keras.utils.Sequence):
    def __init__(self, items, augment=False):
        self.items, self.augment = items, augment
        self.order = np.arange(len(items))
    def __len__(self):
        return int(np.ceil(len(self.items) / BATCH))
    def __getitem__(self, i):
        Xs, Ms, Cs = [], [], []
        for j in self.order[i * BATCH:(i + 1) * BATCH]:
            it = self.items[j]
            x, m = it["x8"], it["m8"]
            if self.augment:
                if np.random.rand() < 0.5: x, m = x[:, ::-1], m[:, ::-1]
                if np.random.rand() < 0.5: x, m = x[::-1], m[::-1]
            x = np.ascontiguousarray(x); m = np.ascontiguousarray(m)
            Xs.append(prep_arr(x))
            Ms.append(np.eye(NSEG, dtype=np.float32)[m])
            Cs.append(np.eye(len(CLS_OUT), dtype=np.float32)[CLS_IDX[it["true"]]])
        return np.stack(Xs), {"mask_out": np.stack(Ms), "class_out": np.stack(Cs)}
    def on_epoch_end(self):
        np.random.shuffle(self.order)

def predict(model, items):
    ci = model.output_names.index("class_out")
    mi = model.output_names.index("mask_out")
    rows, pix = [], np.zeros((NSEG, NSEG), np.int64)
    for it in items:
        out = model.predict(prep_arr(it["x8"])[None], verbose=0)
        pc = out[ci][0]
        k = int(np.argmax(pc))                 # argmax sobre todas las salidas de class_out
        row = {"file": it.get("file", ""), "src": it.get("src", ""), "true": it["true"],
               "pred": CLS_OUT[k], "conf": round(float(pc[k]), 4), "accepted": bool(pc[k] >= TAU)}
        for n in CLS_OUT:
            row[f"p_{n}"] = round(float(pc[CLS_IDX[n]]), 4)
        rows.append(row)
        if "m8" in it:
            pm = out[mi][0].argmax(-1)
            pix += confusion_matrix(it["m8"].ravel(), pm.ravel(), labels=range(NSEG))
    return rows, pix

# ===== Evaluación =====
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def summarize(name, rows, pix):
    n = len(rows)
    cm = confusion_matrix([r["true"] for r in rows], [r["pred"] for r in rows], labels=CLS_OUT)
    k = int(np.trace(cm)); lo, hi = wilson(k, n)
    print(f"\n[{name}] n = {n}")
    print(f"Accuracy: {k/n*100:.2f}%  IC95 Wilson [{lo*100:.1f}, {hi*100:.1f}]")
    print(cm)
    for i, d in enumerate(CLS_OUT):
        ni, ki = cm[i].sum(), cm[i, i]
        if ni == 0:
            continue
        l, h = wilson(ki, ni)
        print(f"  recall {d}: {ki}/{ni} = {ki/ni*100:.1f}%  IC95 [{l*100:.1f}, {h*100:.1f}]")
    freq = max(sum(r["true"] == d for r in rows) for d in DISEASES)
    print(f"Referencia: clase mayoritaria {freq}/{n} = {freq/n*100:.1f}%")
    other = sum(r["pred"] not in DISEASES + ["healthy"] for r in rows)
    print(f"Predicciones fuera de las clases de PlantSeg: {other}")
    ious = {}
    for c, nm in enumerate(SEG_CH):
        tp = pix[c, c]; fp = pix[:, c].sum() - tp; fn = pix[c, :].sum() - tp
        if tp + fp + fn > 0:
            ious[nm] = float(tp / (tp + fp + fn))
    fg = [v for nm, v in ious.items() if nm != "background"]
    print(f"Segmentación: pixel acc {np.trace(pix)/max(pix.sum(),1)*100:.2f}%  "
          f"IoU lesión medio {np.mean(fg)*100:.2f}%  mIoU (con fondo) {np.mean(list(ious.values()))*100:.2f}%")
    for nm, v in ious.items():
        print(f"    IoU {nm}: {v*100:.2f}%")
    return cm

def save_cm_png(cmA, cmB, n, path):
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 3 * len(CLS_OUT)), 7))
    titles = ["Model A (laboratory only)", "Model B (fine-tuned on PlantSeg train)"]
    for ax, cm, title in zip(axes, [cmA, cmB], titles):
        pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
        im = ax.imshow(pct, cmap="Greens", vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                        fontsize=9, color="white" if pct[i, j] > 60 else "black")
        ax.set_xticks(range(len(CLS_OUT))); ax.set_xticklabels(CLS_OUT, rotation=30, ha="right")
        ax.set_yticks(range(len(CLS_OUT))); ax.set_yticklabels(CLS_OUT)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.025, label="% of row (true class)")
    fig.suptitle(f"Confusion matrices, official PlantSeg test partition: {CROP} (n = {n})", fontsize=13)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

# ===== Ejecución =====
if __name__ == "__main__":
    if os.path.exists(FLAG):
        raise SystemExit("El test de esta ejecución ya se evaluó. No se repite.")
    print(f"Cultivo: {CROP} | config: {CFG_SRC} | preproc: {PREPROC} | clases: {CLS_OUT} | canales: {SEG_CH}")

    # 1) Particiones oficiales de PlantSeg (carpetas Training y Test)
    pool = load_plantseg_pool()
    tr_items = [x for x in pool if x["src"] == "Training"]
    te_items = [x for x in pool if x["src"] == "Test"]
    pool_n = len(tr_items) + len(te_items)
    print(f"PlantSeg oficial: train {len(tr_items)} | test {len(te_items)}")

    # 2) Deduplicación dentro de cada partición oficial
    tr_all, rem_tr = dedupe_pool(tr_items)
    te_all, rem_te = dedupe_pool(te_items)

    # 3) El test oficial no puede contener copias de imágenes de entrenamiento
    te_clean, rem_cross = cross_remove(te_all, tr_all)

    # 4) Entrenamiento oficial -> train 85% / valid 15% (estratificado, semilla 42).
    #    Test oficial: se usa tal cual, sin partirlo.
    labels = np.array([x["true"] for x in tr_all])
    tr_i, va_i = train_test_split(np.arange(len(tr_all)), test_size=VALID_FRAC_OF_TRAIN,
                                  stratify=labels, random_state=SEED)
    tr = [tr_all[i] for i in tr_i]
    va = [tr_all[i] for i in va_i]
    te = te_clean
    removed = rem_tr + rem_te + rem_cross
    assert not ({x["file"] for x in te} & {x["file"] for x in tr + va}), "Fuga entre train y test"
    split_info = {"train_valid": "official PlantSeg train folder, 85/15 stratified by class, seed 42",
                   "test": "official PlantSeg test folder, deduplicated against train",
                   "removed_within_train": len(rem_tr), "removed_within_test": len(rem_te),
                   "removed_test_vs_train": len(rem_cross)}

    print(f"Split final ({CROP}): train {len(tr)} | valid {len(va)} | test {len(te)}")
    for name, part in [("train", tr), ("valid", va), ("test", te)]:
        print(f"{name} por clase: {count_table(part)}")
        for s in SOURCES:
            print(f"   origen {s}: {sum(x['src'] == s for x in part)}")
        cnt = count_table(part)
        small = [n for n in DISEASES if cnt[n] < 20]
        if small:
            print(f"   AVISO: menos de 20 muestras en {name} para {small}: resultados exploratorios")

    # 4) Modelo A, orden de clases y comprobación con laboratorio
    model = load_model(MODEL_A, compile=False)
    check_outputs(model)
    lab = lab_items(LAB_ROOT, "valid")
    if TENTATIVE:
        set_class_order(search_cls_order(model, lab))
        print(f"Orden de clases elegido: {CLS_OUT}")
        check_outputs(model)
    rows_lab, _ = predict(model, lab)
    acc_lab = sum(r["pred"] == r["true"] for r in rows_lab) / max(len(rows_lab), 1)
    print(f"Comprobación de laboratorio (valid): {acc_lab*100:.2f}% en {len(rows_lab)} imágenes")
    if acc_lab < LAB_MIN_ACC:
        raise RuntimeError("Accuracy baja en laboratorio: revisar orden de clases, preprocesado o modelo A.")

    # 5) Fine-tuning
   
    tf.keras.utils.set_random_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED); random.seed(TRAIN_SEED)

    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss={"mask_out": focal_tversky_loss(),
                        "class_out": "categorical_crossentropy"},
                  loss_weights={"mask_out": 1.5, "class_out": 0.5},
                  metrics={"class_out": "accuracy"})
    cbs = [ModelCheckpoint(MODEL_B_OUT, monitor="val_loss", save_best_only=True),
           EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
    model.fit(SegSeq(tr, augment=True), validation_data=SegSeq(va),
              epochs=EPOCHS, callbacks=cbs)

    # 6) Evaluación única del test
    best = load_model(MODEL_B_OUT, compile=False)
    rowsB, pixB = predict(best, te)
    rowsA, pixA = predict(load_model(MODEL_A, compile=False), te)
    open(FLAG, "w").write("evaluado")

    save_csv(os.path.join(RESULT_DIR, "pred_B_test.csv"), rowsB)
    save_csv(os.path.join(RESULT_DIR, "pred_A_test.csv"), rowsA)

    cmA = summarize("A (laboratorio) en test retenido", rowsA, pixA)
    cmB = summarize("B (fine-tuning) en test retenido", rowsB, pixB)
    save_cm_png(cmA, cmB, len(te), os.path.join(RESULT_DIR, f"matriz_confusion_A_vs_B_{CROP}.png"))

    okA = np.array([r["pred"] == r["true"] for r in rowsA])
    okB = np.array([r["pred"] == r["true"] for r in rowsB])
    solo_A, solo_B = int((okA & ~okB).sum()), int((okB & ~okA).sum())
    disc = solo_A + solo_B
    p = binomtest(solo_A, disc, 0.5).pvalue if disc else 1.0
    print(f"\nMcNemar ({CROP}): solo A {solo_A}, solo B {solo_B}, p exacto = {p:.4g}")

    manifest = {
        "run": RUN, "crop": CROP, "config_source": CFG_SRC, "arch_A": C["arch"],
        "cls_out": CLS_OUT, "seg_channels": SEG_CH, "preproc": PREPROC, "clahe": False,
        "sources": SOURCES, "dedup_thr": DUP_THR, "seed": SEED, "img_size": IMG,
        "batch": BATCH, "max_epochs": EPOCHS, "lr": LR, "patience": PATIENCE, "tau": TAU,
        "loss": {"alpha_FP": TV_ALPHA, "beta_FN": TV_BETA, "gamma": TV_GAMMA},
        "split": split_info,
        "pool_n": pool_n, "removed_dedup": [x["file"] for x in removed],
        "train_n": len(tr), "valid_n": len(va), "test_n": len(te),
        "sha256_metadata": sha256(METADATA),
        "sha256_model_A": sha256(MODEL_A),
        "sha256_model_B": sha256(MODEL_B_OUT),
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
        "train_seed": TRAIN_SEED,
    }
    json.dump(manifest, open(os.path.join(RESULT_DIR, "manifest.json"), "w"),
              indent=2, ensure_ascii=False)
    print("Resultados guardados en", RESULT_DIR)