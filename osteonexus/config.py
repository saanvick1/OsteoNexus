import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

DATASET_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache/kagglehub/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset/versions/1",
    "OS Collected Data"
)

NORMAL_DIR = os.path.join(DATASET_PATH, "Normal")
OSTEOPENIA_DIR = os.path.join(DATASET_PATH, "Osteopenia")
OSTEO_DIR = os.path.join(DATASET_PATH, "Osteoporosis")

DATASET_SOURCES = [
    {
        "name": "Multi-Class Knee Osteoporosis X-Ray Dataset",
        "author": "Mohamed Gobara et al.",
        "url": "https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset",
        "annotation": "Orthopedic surgery specialists",
        "classes": ["Normal", "Osteopenia", "Osteoporosis"],
        "total_images": 1947,
    },
]

IMG_SIZE = 224
RANDOM_SEEDS = [42, 123, 456, 789, 1024]
PRIMARY_SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.2

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
GAUSSIAN_KERNEL = (5, 5)
ELASTIC_ALPHA = 34
ELASTIC_SIGMA = 4

LEARNING_RATES = [0.001, 0.01, 0.0001]
EPOCH_CONFIGS = [30, 170, 200, 260]
OPTIMAL_LR = 0.002
OPTIMAL_EPOCHS = 200
BATCH_SIZE = 32
PCA_COMPONENTS = 200

MAML_INNER_LR = 0.01
MAML_OUTER_LR = 0.001
MAML_INNER_STEPS = 3
MAML_N_WAY = 2
MAML_K_SHOT = 5
MAML_Q_QUERY = 10
MAML_META_EPOCHS = 50

CLINICAL_BENCHMARKS = {
    "accuracy": 75.0,
    "precision": 80.0,
    "recall": 80.0,
    "f1": 80.0,
    "auc": 85.0,
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_resize(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def normalize_pixels(img):
    return img.astype(np.float64) / 255.0


def apply_clahe(img):
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(img_uint8)


def elastic_transform(image, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,)))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def gaussian_blur(img):
    return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, 0)


def edge_detection(img):
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img
    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scharr_x = cv2.Scharr(img_uint8, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img_uint8, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
    return np.maximum(sobel, scharr)


def morphological_ops(edge_map):
    if edge_map.dtype != np.uint8:
        edge_uint8 = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        edge_uint8 = edge_map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edge_uint8, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded


def frequency_filter(img):
    if img.dtype == np.uint8:
        img_float = img.astype(np.float64)
    else:
        img_float = img.astype(np.float64)
    f_transform = np.fft.fft2(img_float)
    f_shift = np.fft.fftshift(f_transform)
    rows, cols = img_float.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float64)
    r = 10
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0.0
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def full_preprocessing_pipeline(path, augment=False, random_state=None):
    img = load_and_resize(path)
    clahe_img = apply_clahe(img)
    if augment:
        clahe_img = elastic_transform(clahe_img, random_state=random_state)
    blurred = gaussian_blur(clahe_img)
    edges = edge_detection(blurred)
    morph_edges = morphological_ops(edges)
    freq_filtered = frequency_filter(morph_edges)
    morph_float = morph_edges.astype(np.float64) / 255.0
    freq_norm = freq_filtered / (freq_filtered.max() + 1e-8)
    final = np.clip(morph_float + freq_norm, 0, 1)
    processed = normalize_pixels(clahe_img)
    return processed, final


def extract_hog_features(img):
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img
    features = hog(
        img_uint8,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
        block_norm='L2-Hys'
    )
    return features


def extract_lbp_features(img, n_points=24, radius=3):
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img
    lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_all_features(path, augment=False, random_state=None):
    processed, final_img = full_preprocessing_pipeline(path, augment, random_state)
    hog_feat = extract_hog_features(processed)
    lbp_feat = extract_lbp_features(processed)
    pixel_feat = processed.flatten()
    combined = np.concatenate([pixel_feat, hog_feat, lbp_feat])
    return combined, processed


def build_dataframe():
    records = []
    for fname in os.listdir(NORMAL_DIR):
        fpath = os.path.join(NORMAL_DIR, fname)
        if os.path.isfile(fpath):
            records.append({"path": fpath, "label": 0, "class_name": "Normal", "original_class": "Normal"})
    for fname in os.listdir(OSTEOPENIA_DIR):
        fpath = os.path.join(OSTEOPENIA_DIR, fname)
        if os.path.isfile(fpath):
            records.append({"path": fpath, "label": 1, "class_name": "At-Risk", "original_class": "Osteopenia"})
    for fname in os.listdir(OSTEO_DIR):
        fpath = os.path.join(OSTEO_DIR, fname)
        if os.path.isfile(fpath):
            records.append({"path": fpath, "label": 1, "class_name": "At-Risk", "original_class": "Osteoporosis"})
    df = pd.DataFrame(records)
    return df


def get_image_dimensions(df):
    dims = []
    for _, row in df.iterrows():
        img = cv2.imread(row["path"])
        if img is not None:
            h, w = img.shape[:2]
            dims.append({"path": row["path"], "label": row["label"],
                         "class_name": row["class_name"], "width": w, "height": h})
    return pd.DataFrame(dims)


def extract_features_from_df(df, augment=False, seed=None):
    features = []
    labels = []
    rng = np.random.RandomState(seed) if seed is not None else None
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            feat, _ = extract_all_features(row["path"], augment=augment, random_state=rng)
            features.append(feat)
            labels.append(row["label"])
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
    return np.array(features), np.array(labels)


def split_data(df, seed=PRIMARY_SEED):
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=seed, stratify=train_val_df["label"]
    )
    return train_df, val_df, test_df


def standardize_features_per_fold(X_train, X_val, X_test, apply_pca=True, pca_components=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = None
    if apply_pca:
        if pca_components is None:
            pca_components = PCA_COMPONENTS
        n_components = min(pca_components, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"PCA: {X_train.shape[1]} -> {n_components} dims ({explained:.1f}% variance explained)")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_kfold_splits(df, n_splits=5, seed=PRIMARY_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in skf.split(df, df["label"]):
        splits.append((train_idx, test_idx))
    return splits


def prepare_data(seed=PRIMARY_SEED):
    print(f"Building dataframe from dataset...")
    df = build_dataframe()
    print(f"Total images: {len(df)} | Normal: {(df['label']==0).sum()} | Osteoporosis (incl. Osteopenia): {(df['label']==1).sum()}")
    if "original_class" in df.columns:
        osteopenia_count = (df["original_class"] == "Osteopenia").sum()
        osteoporosis_count = (df["original_class"] == "Osteoporosis").sum()
        print(f"  Breakdown: Osteopenia={osteopenia_count}, Osteoporosis={osteoporosis_count}")

    train_df, val_df, test_df = split_data(df, seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("Extracting training features...")
    X_train, y_train = extract_features_from_df(train_df, augment=True, seed=seed)
    print("Extracting validation features...")
    X_val, y_val = extract_features_from_df(val_df, augment=False, seed=seed)
    print("Extracting test features...")
    X_test, y_test = extract_features_from_df(test_df, augment=False, seed=seed)

    X_train, X_val, X_test, scaler = standardize_features_per_fold(X_train, X_val, X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, df
