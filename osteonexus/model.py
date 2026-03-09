import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    brier_score_loss, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from config import (
    OPTIMAL_LR, OPTIMAL_EPOCHS, BATCH_SIZE,
    MAML_INNER_LR, MAML_OUTER_LR, MAML_INNER_STEPS,
    MAML_N_WAY, MAML_K_SHOT, MAML_Q_QUERY, MAML_META_EPOCHS
)


def build_osteonexus_model(input_dim, use_attention=True, use_lstm=True, use_autoencoder=True):
    reg = l2(0.0005)
    inputs = layers.Input(shape=(input_dim,), name='input')
    x = layers.BatchNormalization()(inputs)

    if use_attention:
        attn_weights = layers.Dense(input_dim, activation='sigmoid', name='attention',
                                    kernel_regularizer=reg)(x)
        x = layers.Multiply(name='attended_features')([x, attn_weights])

    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.25)(x)

    if use_lstm:
        x = layers.Reshape((1, 64))(x)
        x = layers.LSTM(32, return_sequences=False, name='lstm',
                        kernel_regularizer=reg)(x)
    else:
        x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)

    x = layers.Dropout(0.25)(x)

    if use_autoencoder:
        encoded = layers.Dense(16, activation='relu', name='encoder',
                               kernel_regularizer=reg)(x)
        x = encoded

    x = layers.Dense(16, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='OsteoNexus')
    return model


def compile_model(model, lr=OPTIMAL_LR):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=OPTIMAL_EPOCHS, batch_size=BATCH_SIZE):
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {i: w for i, w in enumerate(cw)}

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True,
                      mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0
    )
    return history


def build_se_attention_block(x, ratio=8):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(filters // ratio, 4), activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def build_cnn_with_se_attention(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = build_se_attention_block(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = build_se_attention_block(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = build_se_attention_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs, name='CNN_SE_Attention')
    return model


def compute_all_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }
    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob) * 100
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics


def bootstrap_confidence_intervals(y_true, y_pred, y_prob=None, n_bootstrap=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": []
    }
    if y_prob is not None:
        boot_metrics["auc"] = []
        boot_metrics["brier_score"] = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]

        if len(np.unique(y_t)) < 2:
            continue

        boot_metrics["accuracy"].append(accuracy_score(y_t, y_p) * 100)
        boot_metrics["precision"].append(precision_score(y_t, y_p, zero_division=0) * 100)
        boot_metrics["recall"].append(recall_score(y_t, y_p, zero_division=0) * 100)
        boot_metrics["f1"].append(f1_score(y_t, y_p, zero_division=0) * 100)

        if y_prob is not None:
            y_pr = y_prob[indices]
            try:
                boot_metrics["auc"].append(roc_auc_score(y_t, y_pr) * 100)
                boot_metrics["brier_score"].append(brier_score_loss(y_t, y_pr))
            except ValueError:
                pass

    ci_results = {}
    for metric_name, values in boot_metrics.items():
        values = np.array(values)
        lower = np.percentile(values, (alpha / 2) * 100)
        upper = np.percentile(values, (1 - alpha / 2) * 100)
        mean = np.mean(values)
        ci_results[metric_name] = {"mean": mean, "lower": lower, "upper": upper, "std": np.std(values)}

    return ci_results


def delong_test(y_true, y_prob_1, y_prob_2):
    n = len(y_true)
    auc1 = roc_auc_score(y_true, y_prob_1)
    auc2 = roc_auc_score(y_true, y_prob_2)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return {"auc_diff": auc1 - auc2, "z_stat": 0, "p_value": 1.0}

    v10_1 = np.array([np.mean(y_prob_1[pos] > t) for t in y_prob_1[neg]])
    v01_1 = np.array([np.mean(y_prob_1[neg] < t) for t in y_prob_1[pos]])
    v10_2 = np.array([np.mean(y_prob_2[pos] > t) for t in y_prob_2[neg]])
    v01_2 = np.array([np.mean(y_prob_2[neg] < t) for t in y_prob_2[pos]])

    s10 = np.cov(np.stack([v10_1, v10_2]))
    s01 = np.cov(np.stack([v01_1, v01_2]))

    s = s10 / n_neg + s01 / n_pos
    diff = auc1 - auc2

    if s[0, 0] + s[1, 1] - 2 * s[0, 1] <= 0:
        return {"auc_diff": diff, "z_stat": 0, "p_value": 1.0}

    z = diff / np.sqrt(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {"auc_diff": diff, "z_stat": z, "p_value": p_value}


def mcnemar_test(y_true, y_pred_1, y_pred_2):
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    b = np.sum(correct_1 & ~correct_2)
    c = np.sum(~correct_1 & correct_2)

    if b + c == 0:
        return {"chi2": 0, "p_value": 1.0, "b": int(b), "c": int(c)}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {"chi2": chi2, "p_value": p_value, "b": int(b), "c": int(c)}


def calibration_curve_data(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    ece = sum(
        (count / len(y_true)) * abs(acc - center)
        for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts)
    )

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def cross_validate_model(build_fn, X, y, n_splits=5, seed=42, epochs=200, lr=0.002):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        tf.random.set_seed(seed + fold)
        np.random.seed(seed + fold)

        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model = build_fn(X.shape[1])
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy']
        )

        y_tr_cat = tf.keras.utils.to_categorical(y_tr, 2)
        y_vl_cat = tf.keras.utils.to_categorical(y_vl, 2)

        cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        class_weight = {i: w for i, w in enumerate(cw)}

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, mode='max', verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ]

        model.fit(X_tr, y_tr_cat, validation_data=(X_vl, y_vl_cat),
                  epochs=epochs, batch_size=32, callbacks=callbacks,
                  class_weight=class_weight, verbose=0)

        y_prob_raw = model.predict(X_vl, verbose=0)
        y_prob = y_prob_raw[:, 1] if y_prob_raw.shape[1] == 2 else y_prob_raw.flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_all_metrics(y_vl, y_pred, y_prob)
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}/{n_splits}: Acc={metrics['accuracy']:.2f}% F1={metrics['f1']:.2f}% AUC={metrics.get('auc',0):.2f}%")

    cv_summary = {}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [m[key] for m in fold_metrics if key in m]
        cv_summary[key] = {'mean': np.mean(values), 'std': np.std(values), 'values': values}
    return cv_summary, fold_metrics


def evaluate_model(model, X_test, y_test):
    y_prob_raw = model.predict(X_test, verbose=0)
    if y_prob_raw.shape[1] == 2:
        y_prob = y_prob_raw[:, 1]
    else:
        y_prob = y_prob_raw.flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob)
    cal = calibration_curve_data(y_test, y_prob)

    return {
        "metrics": metrics,
        "confidence_intervals": ci,
        "calibration": cal,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test, seed=42):
    model = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob, seed=seed)
    return {"model": model, "metrics": metrics, "ci": ci, "y_pred": y_pred, "y_prob": y_prob, "name": "Logistic Regression"}


def train_svm(X_train, y_train, X_test, y_test, seed=42):
    model = SVC(kernel='rbf', probability=True, random_state=seed, C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob, seed=seed)
    return {"model": model, "metrics": metrics, "ci": ci, "y_pred": y_pred, "y_prob": y_prob, "name": "SVM (RBF)"}


def train_random_forest(X_train, y_train, X_test, y_test, seed=42):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob, seed=seed)
    return {"model": model, "metrics": metrics, "ci": ci, "y_pred": y_pred, "y_prob": y_prob, "name": "Random Forest"}


def train_gradient_boosting(X_train, y_train, X_test, y_test, seed=42):
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=seed, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob, seed=seed)
    return {"model": model, "metrics": metrics, "ci": ci, "y_pred": y_pred, "y_prob": y_prob, "name": "Gradient Boosting"}


def run_all_baselines(X_train, y_train, X_test, y_test, seed=42):
    results = {}
    print("\n" + "="*60)
    print("RUNNING BASELINE MODELS")
    print("="*60)

    print("\n[1/4] Logistic Regression...")
    results["logistic_regression"] = train_logistic_regression(X_train, y_train, X_test, y_test, seed)

    print("[2/4] SVM (RBF kernel)...")
    results["svm"] = train_svm(X_train, y_train, X_test, y_test, seed)

    print("[3/4] Random Forest...")
    results["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test, seed)

    print("[4/4] Gradient Boosting...")
    results["gradient_boosting"] = train_gradient_boosting(X_train, y_train, X_test, y_test, seed)

    print("\n" + "-"*60)
    print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-"*60)
    for key, res in results.items():
        m = res["metrics"]
        print(f"{res['name']:<25} {m['accuracy']:>7.2f}% {m['precision']:>7.2f}% {m['recall']:>7.2f}% {m['f1']:>7.2f}% {m.get('auc', 0):>7.2f}%")
    print("-"*60)

    return results


def run_ablation_study(X_train, y_train, X_val, y_val, X_test, y_test, seed=42, epochs=60):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    configs = [
        {"name": "Full OsteoNexus", "attention": True, "lstm": True, "autoencoder": True},
        {"name": "No Attention", "attention": False, "lstm": True, "autoencoder": True},
        {"name": "No LSTM", "attention": True, "lstm": False, "autoencoder": True},
        {"name": "No Autoencoder", "attention": True, "lstm": True, "autoencoder": False},
        {"name": "Baseline (None)", "attention": False, "lstm": False, "autoencoder": False},
    ]

    results = {}
    input_dim = X_train.shape[1]

    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['name']}...")
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = build_osteonexus_model(
            input_dim,
            use_attention=cfg["attention"],
            use_lstm=cfg["lstm"],
            use_autoencoder=cfg["autoencoder"]
        )
        model = compile_model(model, lr=OPTIMAL_LR)
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
        eval_result = evaluate_model(model, X_test, y_test)

        results[cfg["name"]] = {
            "config": cfg,
            "metrics": eval_result["metrics"],
            "ci": eval_result["confidence_intervals"],
            "history": {
                "train_acc": history.history.get("accuracy", []),
                "val_acc": history.history.get("val_accuracy", []),
                "train_loss": history.history.get("loss", []),
                "val_loss": history.history.get("val_loss", []),
            }
        }

        m = eval_result["metrics"]
        print(f"  Accuracy: {m['accuracy']:.2f}% | F1: {m['f1']:.2f}% | AUC: {m.get('auc', 0):.2f}%")

    print("\n" + "-"*70)
    print(f"{'Configuration':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-"*70)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<30} {m['accuracy']:>7.2f}% {m['precision']:>7.2f}% {m['recall']:>7.2f}% {m['f1']:>7.2f}% {m.get('auc', 0):>7.2f}%")
    print("-"*70)

    return results


def run_feature_ablation(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, seed=42):
    print("\n" + "="*60)
    print("FEATURE ABLATION STUDY (Logistic Regression - fast)")
    print("="*60)
    print("Note: PCA was already applied. Testing on PCA-reduced features.")
    print("Feature ablation compares full PCA features vs subsets.\n")

    dim = X_train_raw.shape[1]
    half = dim // 2
    feature_configs = [
        {"name": "All PCA Features", "start": 0, "end": None},
        {"name": "First Half PCA", "start": 0, "end": half},
        {"name": "Second Half PCA", "start": half, "end": None},
    ]

    results = {}
    for i, cfg in enumerate(feature_configs):
        print(f"[{i+1}/{len(feature_configs)}] {cfg['name']}...")
        s, e = cfg["start"], cfg["end"]
        Xtr = X_train_raw[:, s:e]
        Xte = X_test_raw[:, s:e]

        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(Xtr, y_train)
        y_pred = lr.predict(Xte)
        y_prob = lr.predict_proba(Xte)[:, 1]
        m = compute_all_metrics(y_test, y_pred, y_prob)
        results[cfg["name"]] = m
        print(f"  Dim: {Xtr.shape[1]} | Accuracy: {m['accuracy']:.2f}% | F1: {m['f1']:.2f}%")

    print("\n" + "-"*60)
    print(f"{'Feature Set':<35} {'Dim':>6} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-"*60)
    for name, m in results.items():
        cfg = next(c for c in feature_configs if c["name"] == name)
        s, e = cfg["start"], cfg["end"]
        dim = X_train_raw[:, s:e].shape[1]
        print(f"{name:<35} {dim:>6} {m['accuracy']:>7.2f}% {m['f1']:>7.2f}% {m.get('auc', 0):>7.2f}%")
    print("-"*60)

    return results


def create_episodic_task(X, y, n_way=MAML_N_WAY, k_shot=MAML_K_SHOT, q_query=MAML_Q_QUERY):
    classes = np.unique(y)
    selected_classes = np.random.choice(classes, n_way, replace=False)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for i, cls in enumerate(selected_classes):
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, k_shot + q_query, replace=len(cls_indices) < k_shot + q_query)

        support_x.append(X[selected[:k_shot]])
        support_y.append(np.full(k_shot, i))
        query_x.append(X[selected[k_shot:k_shot + q_query]])
        query_y.append(np.full(q_query, i))

    support_x = np.concatenate(support_x)
    support_y = np.concatenate(support_y)
    query_x = np.concatenate(query_x)
    query_y = np.concatenate(query_y)

    return support_x, support_y, query_x, query_y


def build_maml_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(MAML_N_WAY, activation='softmax')(x)
    model = Model(inputs, outputs, name='MAML_OsteoNexus')
    return model


def compute_loss(model, x, y):
    y_cat = tf.keras.utils.to_categorical(y, MAML_N_WAY)
    predictions = model(x, training=True)
    loss = tf.keras.losses.categorical_crossentropy(y_cat, predictions)
    return tf.reduce_mean(loss)


def maml_inner_loop(model, support_x, support_y, inner_lr=MAML_INNER_LR, inner_steps=MAML_INNER_STEPS):
    adapted_weights = [tf.identity(w) for w in model.trainable_variables]

    for _ in range(inner_steps):
        with tf.GradientTape() as tape:
            for w, aw in zip(model.trainable_variables, adapted_weights):
                w.assign(aw)
            loss = compute_loss(model, support_x, support_y)

        gradients = tape.gradient(loss, model.trainable_variables)
        adapted_weights = [
            w - inner_lr * g if g is not None else w
            for w, g in zip(adapted_weights, gradients)
        ]

    for w, aw in zip(model.trainable_variables, adapted_weights):
        w.assign(aw)

    return model


def reptile_meta_train(model, X_train, y_train, meta_epochs=MAML_META_EPOCHS,
                       inner_steps=MAML_INNER_STEPS, inner_lr=MAML_INNER_LR,
                       outer_lr=MAML_OUTER_LR, tasks_per_epoch=10):
    meta_losses = []

    for epoch in range(meta_epochs):
        original_weights = [w.numpy().copy() for w in model.trainable_variables]
        epoch_loss = 0.0

        for task_i in range(tasks_per_epoch):
            support_x, support_y, query_x, query_y = create_episodic_task(
                X_train, y_train
            )

            for w, ow in zip(model.trainable_variables, original_weights):
                w.assign(ow)

            optimizer = Adam(learning_rate=inner_lr)
            for step in range(inner_steps):
                with tf.GradientTape() as tape:
                    loss = compute_loss(model, support_x, support_y)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            query_loss = compute_loss(model, query_x, query_y).numpy()
            epoch_loss += query_loss

            task_weights = [w.numpy().copy() for w in model.trainable_variables]
            updated_weights = [
                ow + outer_lr * (tw - ow)
                for ow, tw in zip(original_weights, task_weights)
            ]
            original_weights = updated_weights

        for w, uw in zip(model.trainable_variables, original_weights):
            w.assign(uw)

        avg_loss = epoch_loss / tasks_per_epoch
        meta_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Reptile Epoch {epoch+1}/{meta_epochs} | Avg Query Loss: {avg_loss:.4f}")

    return meta_losses


def prototypical_network_eval(model_encoder, support_x, support_y, query_x, query_y):
    support_embeddings = model_encoder.predict(support_x, verbose=0)
    query_embeddings = model_encoder.predict(query_x, verbose=0)

    classes = np.unique(support_y)
    prototypes = []
    for cls in classes:
        cls_mask = support_y == cls
        proto = support_embeddings[cls_mask].mean(axis=0)
        prototypes.append(proto)
    prototypes = np.array(prototypes)

    distances = np.linalg.norm(
        query_embeddings[:, np.newaxis, :] - prototypes[np.newaxis, :, :],
        axis=2
    )
    predictions = np.argmin(distances, axis=1)

    accuracy = np.mean(predictions == query_y)
    return accuracy, predictions


def build_protonet_encoder(input_dim, embedding_dim=32):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    embeddings = layers.Dense(embedding_dim, activation=None, name='embedding')(x)
    encoder = Model(inputs, embeddings, name='ProtoNet_Encoder')
    return encoder


def train_protonet(encoder, X_train, y_train, epochs=100, tasks_per_epoch=20, lr=0.001):
    optimizer = Adam(learning_rate=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(tasks_per_epoch):
            support_x, support_y, query_x, query_y = create_episodic_task(X_train, y_train)

            with tf.GradientTape() as tape:
                support_emb = encoder(support_x, training=True)
                query_emb = encoder(query_x, training=True)

                classes = tf.unique(support_y)[0]
                prototypes = []
                for cls in classes:
                    mask = tf.equal(support_y, cls)
                    cls_emb = tf.boolean_mask(support_emb, mask)
                    prototypes.append(tf.reduce_mean(cls_emb, axis=0))
                prototypes = tf.stack(prototypes)

                dists = tf.reduce_sum(
                    tf.square(tf.expand_dims(query_emb, 1) - tf.expand_dims(prototypes, 0)),
                    axis=2
                )
                log_probs = tf.nn.log_softmax(-dists, axis=1)

                query_y_int = tf.cast(query_y, tf.int32)
                loss = -tf.reduce_mean(
                    tf.gather_nd(log_probs, tf.stack([tf.range(tf.shape(query_y_int)[0]), query_y_int], axis=1))
                )

            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / tasks_per_epoch
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"ProtoNet Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return losses
