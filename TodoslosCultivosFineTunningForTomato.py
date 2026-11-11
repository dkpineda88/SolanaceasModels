# -*- coding: utf-8 -*-
"""
Experimento B - TOMATE, con ambas particiones de PlantSeg (Training + Test).
Ninguna de las dos se usó en el experimento A.
Flujo:
  1) Pool = PlantSeg Training + PlantSeg Test (tomate, clases con lesión).
  2) Deduplicación pHash dentro del pool, ANTES de partir.
  3) Split estratificado 70/15/15 del pool deduplicado.
  4) Fine-tuning de A sobre train; valid para checkpoint y early stopping.
  5) Test evaluado una sola vez por ejecución.
"""
import os, json, csv, base64, hashlib, random
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===== Rutas =====
PS_ROOT   = "D:/DATASETS/Imagenes/Imagenes de Campo/plantseg/plantseg"
METADATA  = os.path.join(PS_ROOT, "Metadata.csv")
JSON_ROOT = os.path.join(PS_ROOT, "json")
IMG_ROOT  = os.path.join(PS_ROOT, "images")
BASE      = "D:/Proyectos/Solanaceas Modelos/ExperimentoB"
RUN       = "B_pool_2026"                       # nueva carpeta: no reutilizar ejecuciones anteriores
MODEL_A   = "D:/Proyectos/Solanaceas Modelos/ExperimentoA/Tomate/Old models/TMTMaskMobilNet5Class.keras"  # CONFIRMAR
LAB_ROOT  = "D:/DATASETS/Imagenes/Solanaceas/Zonodo/TomatoDisease.v6i.coco"

RESULT_DIR  = os.path.join(BASE, RUN, "resultados_tomato")
MODEL_B_OUT = os.path.join(BASE, RUN, "tomato_B.keras")
FLAG        = os.path.join(RESULT_DIR, "TEST_EVALUADO.flag")

# ===== Parámetros fijados antes de ver resultados =====
SEED, IMG, BATCH, EPOCHS, LR, PATIENCE, TAU = 42, 256, 8, 30, 1e-5, 5, 0.60
RATIO_TOL, DUP_THR = 0.02, 8
TRAIN_FRAC, VALID_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SOURCES = ["Training", "Test"]                 # ambas particiones de PlantSeg
# Preprocesado de A para tomate con MobileNet: sin CLAHE (pipeline validado en la reevaluación)
USE_CLAHE = False
# Pérdida focal Tversky (Ecs. 6-7): TI = TP/(TP + a·FP + b·FN), loss = (1-TI)^(1/g)
# CONFIRMAR con el código de entrenamiento de A
TV_ALPHA, TV_BETA, TV_GAMMA = 0.3, 0.7, 1.33

# ===== Orden EXACTO de salidas del modelo A (mapping.json de tomate) =====
CLS_OUT = ["earlyblight", "bacterialspot", "healthy", "lateblight", "yellowleafcurlvirus"]
SEG_CH  = ["earlyblight", "bacterialspot", "background", "lateblight", "yellowleafcurlvirus"]
CLS_IDX = {n: i for i, n in enumerate(CLS_OUT)}
SEG_IDX = {n: i for i, n in enumerate(SEG_CH)}
BG      = SEG_IDX["background"]
NCLS    = len(CLS_OUT)
NSEG    = len(SEG_CH)
DISEASES = [n for n in CLS_OUT if n != "healthy"]   # PlantSeg no tiene sanos para tomate
PRIORITY = ["lateblight", "earlyblight", "yellowleafcurlvirus", "bacterialspot", "healthy"]
PLANTSEG_PLANT = "tomato"
PLANTSEG_MAP = {
    "tomato bacterial leaf spot": "bacterialspot",
    "tomato early blight": "earlyblight",
    "tomato late blight": "lateblight",
    "tomato yellow leaf curl virus": "yellowleafcurlvirus",
}

np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
os.makedirs(RESULT_DIR, exist_ok=True)

def norm(s):
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

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

# ===== Imagen =====
def to256(rgb):
    img = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    if USE_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

def prep_arr(x8):
    return preprocess_input(x8.astype(np.float32))

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

# ===== Carga del pool: PlantSeg Training + Test =====
def load_plantseg_pool():
    df = pd.read_csv(METADATA)
    df["Plant"] = df["Plant"].str.strip()
    df["Disease"] = df["Disease"].str.strip().str.lower()
    t = df[(df.Plant.str.lower() == PLANTSEG_PLANT) & (df["Training/Test"].isin(SOURCES))].copy()
    t["cls"] = t.Disease.map(PLANTSEG_MAP)
    excluded = int(t.cls.isna().sum())
    t = t[t.cls.notna()].copy()
    t["stem"] = t.Name.str.rsplit(".", n=1).str[0]
    # Un mismo nombre de archivo en Training y en Test rompería la correspondencia imagen-anotación
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
        items.append({"file": r.Name, "src": r["Training/Test"], "x8": x8, "m8": mask,
                      "label": CLS_IDX[r.cls], "true": r.cls})
    print(f"PlantSeg pool (tomato): {len(items)} válidas, {excluded} excluidas por clase, {len(rejected)} rechazadas")
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
    """Deduplicación global del pool. Se conserva la primera aparición de cada grupo de copias."""
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

def count_table(items):
    return {n: sum(x["true"] == n for x in items) for n in CLS_OUT}

# ===== Verificación de orden de clases con laboratorio (no entrena) =====
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

# ===== Modelo =====
def check_outputs(model):
    names = model.output_names
    assert {"mask_out", "class_out"} <= set(names), names
    m_ch = int(model.outputs[names.index("mask_out")].shape[-1])
    c_ch = int(model.outputs[names.index("class_out")].shape[-1])
    print(f"mask_out: {m_ch} canales (esperados {NSEG}); class_out: {c_ch} unidades (esperadas {NCLS})")
    assert m_ch == NSEG, f"mask_out tiene {m_ch} canales; esperados {NSEG} {SEG_CH}"
    assert c_ch == NCLS, f"class_out tiene {c_ch} unidades; esperadas {NCLS} {CLS_OUT}"
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
            if self.augment:                       # el volteo se aplica igual a imagen y máscara
                if np.random.rand() < 0.5: x, m = x[:, ::-1], m[:, ::-1]
                if np.random.rand() < 0.5: x, m = x[::-1], m[::-1]
            x = np.ascontiguousarray(x); m = np.ascontiguousarray(m)
            Xs.append(prep_arr(x))
            Ms.append(np.eye(NSEG, dtype=np.float32)[m])
            Cs.append(np.eye(NCLS, dtype=np.float32)[it["label"]])
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
        k = int(np.argmax(pc))                     # argmax sobre las 5 salidas, igual que en la reevaluación de A
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
            print(f"  recall {d}: sin muestras en test (no evaluable)"); continue
        l, h = wilson(ki, ni)
        print(f"  recall {d}: {ki}/{ni} = {ki/ni*100:.1f}%  IC95 [{l*100:.1f}, {h*100:.1f}]")
    freq = max(sum(r["true"] == d for r in rows) for d in DISEASES)
    print(f"Referencia: clase mayoritaria {freq}/{n} = {freq/n*100:.1f}%")
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    titles = ["Model A (laboratory only)", "Model B (fine-tuned on PlantSeg pool)"]
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
    fig.suptitle(f"Confusion matrices, held-out split of PlantSeg pool: tomato (n = {n})", fontsize=13)
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

    # 1) Pool = PlantSeg Training + Test
    pool = load_plantseg_pool()

    # 2) Deduplicación global ANTES de partir
    pool, removed = dedupe_pool(pool)

    # 3) Split estratificado 70/15/15 del pool deduplicado
    labels = np.array([x["label"] for x in pool])
    tr_i, tmp_i = train_test_split(np.arange(len(pool)), test_size=1 - TRAIN_FRAC,
                                   stratify=labels, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=TEST_FRAC / (VALID_FRAC + TEST_FRAC),
                                  stratify=labels[tmp_i], random_state=SEED)
    tr = [pool[i] for i in tr_i]
    va = [pool[i] for i in va_i]
    te = [pool[i] for i in te_i]

    print(f"Split final (tomato): train {len(tr)} | valid {len(va)} | test {len(te)}")
    for name, part in [("train", tr), ("valid", va), ("test", te)]:
        print(f"{name} por clase: {count_table(part)}")
        for s in SOURCES:
            print(f"   origen {s}: {sum(x['src'] == s for x in part)}")
        small = [n for n in DISEASES if count_table(part)[n] < 20]
        if small:
            print(f"   AVISO: menos de 20 muestras en {name} para {small}: resultados exploratorios")

    # 4) Modelo A y comprobación de orden de clases con laboratorio
    model = load_model(MODEL_A, compile=False)
    check_outputs(model)
    rows_lab, _ = predict(model, lab_items(LAB_ROOT, "valid"))
    acc_lab = sum(r["pred"] == r["true"] for r in rows_lab) / max(len(rows_lab), 1)
    print(f"Comprobación de orden de clases (laboratorio, valid): {acc_lab*100:.2f}% en {len(rows_lab)} imágenes")
    if acc_lab < 0.90:
        raise RuntimeError("Accuracy baja en laboratorio: revisar orden de clases, preprocesado o modelo A.")

    # 5) Fine-tuning
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
    save_cm_png(cmA, cmB, len(te), os.path.join(RESULT_DIR, "matriz_confusion_A_vs_B_tomato.png"))

    okA = np.array([r["pred"] == r["true"] for r in rowsA])
    okB = np.array([r["pred"] == r["true"] for r in rowsB])
    solo_A, solo_B = int((okA & ~okB).sum()), int((okB & ~okA).sum())
    disc = solo_A + solo_B
    p = binomtest(solo_A, disc, 0.5).pvalue if disc else 1.0
    print(f"\nMcNemar (tomato): solo A {solo_A}, solo B {solo_B}, p exacto = {p:.4g}")

    manifest = {
        "run": RUN, "crop": "tomato", "cls_out": CLS_OUT, "seg_channels": SEG_CH,
        "sources": SOURCES, "dedup_thr": DUP_THR,
        "seed": SEED, "img_size": IMG, "batch": BATCH, "max_epochs": EPOCHS,
        "lr": LR, "patience": PATIENCE, "tau": TAU,
        "preproc": "mobilenet", "clahe": USE_CLAHE,
        "loss": {"alpha_FP": TV_ALPHA, "beta_FN": TV_BETA, "gamma": TV_GAMMA},
        "split": {"train": TRAIN_FRAC, "valid": VALID_FRAC, "test": TEST_FRAC},
        "pool_n": len(pool), "removed_dedup": [x["file"] for x in removed],
        "train_n": len(tr), "valid_n": len(va), "test_n": len(te),
        "sha256_metadata": sha256(METADATA),
        "sha256_model_A": sha256(MODEL_A),
        "sha256_model_B": sha256(MODEL_B_OUT),
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
    }
    json.dump(manifest, open(os.path.join(RESULT_DIR, "manifest.json"), "w"),
              indent=2, ensure_ascii=False)
    print("Resultados guardados en", RESULT_DIR)