import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_curve, confusion_matrix as sk_confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PRIMARY_SEED, RANDOM_SEEDS, OPTIMAL_LR, OPTIMAL_EPOCHS,
    RESULTS_DIR, CLINICAL_BENCHMARKS, MAML_META_EPOCHS, DATASET_SOURCES,
    build_dataframe, get_image_dimensions, prepare_data
)
from model import (
    build_osteonexus_model, compile_model, train_model,
    evaluate_model, delong_test, mcnemar_test, compute_all_metrics,
    run_all_baselines, run_ablation_study, run_feature_ablation,
    build_maml_model, reptile_meta_train,
    build_protonet_encoder, train_protonet,
    create_episodic_task, prototypical_network_eval
)

plt.rcParams.update({
    'figure.facecolor': '#0f1318',
    'axes.facecolor': '#141a22',
    'text.color': '#d4dae4',
    'axes.labelcolor': '#d4dae4',
    'xtick.color': '#8899aa',
    'ytick.color': '#8899aa',
    'axes.edgecolor': '#1e2a36',
    'grid.color': '#1e2a36',
    'font.family': 'sans-serif',
})

CYAN = '#2db5b5'
GREEN = '#50c878'
RED = '#e74c6f'
AMBER = '#f0a030'
BLUE = '#4488cc'
PURPLE = '#8866cc'


def plot_training_curves(history, title_suffix="", save_name="training_curves"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_acc = history.get("accuracy", history.get("train_acc", []))
    val_acc = history.get("val_accuracy", history.get("val_acc", []))
    train_loss = history.get("loss", history.get("train_loss", []))
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_acc) + 1)

    axes[0].plot(epochs, train_acc, color=CYAN, linewidth=2, label='Train Accuracy')
    axes[0].plot(epochs, val_acc, color=GREEN, linewidth=2, label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Accuracy Curves {title_suffix}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_loss, color=CYAN, linewidth=2, label='Train Loss')
    axes[1].plot(epochs, val_loss, color=RED, linewidth=2, label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Loss Curves {title_suffix}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_name="confusion_matrix"):
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn',
                xticklabels=['Healthy', 'Osteoporosis'],
                yticklabels=['Healthy', 'Osteoporosis'],
                ax=ax, cbar_kws={'shrink': 0.8},
                linewidths=0.5, linecolor='#1e2a36',
                annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_roc_curve(y_true, y_prob, auc_val, title="ROC Curve", save_name="roc_curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=CYAN, linewidth=2.5, label=f'OsteoNexus (AUC = {auc_val:.2f}%)')
    ax.fill_between(fpr, tpr, alpha=0.15, color=CYAN)
    ax.plot([0, 1], [0, 1], '--', color='#555', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_metrics_vs_benchmarks(metrics, save_name="metrics_vs_benchmarks"):
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    achieved = [metrics.get(m, 0) for m in metric_names]
    benchmarks = [CLINICAL_BENCHMARKS.get(m, 0) for m in metric_names]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, achieved, width, label='Achieved', color=CYAN, alpha=0.9)
    bars2 = ax.bar(x + width/2, benchmarks, width, label='Clinical Benchmark', color=RED, alpha=0.6)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('OsteoNexus Performance vs Clinical Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, color=CYAN)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_confidence_intervals(ci_results, save_name="confidence_intervals"):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    available = [(m, l) for m, l in zip(metrics, labels) if m in ci_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(available))

    for i, (m, l) in enumerate(available):
        ci = ci_results[m]
        mean = ci["mean"]
        lower = ci["lower"]
        upper = ci["upper"]
        ax.barh(i, mean, color=CYAN, alpha=0.7, height=0.5)
        ax.errorbar(mean, i, xerr=[[mean - lower], [upper - mean]],
                     fmt='o', color='white', markersize=5, capsize=5, capthick=2)
        ax.text(mean + 1, i, f'{mean:.1f}% [{lower:.1f}%, {upper:.1f}%]',
                va='center', fontsize=9, color='#d4dae4')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l for _, l in available])
    ax.set_xlabel('Percentage (%)')
    ax.set_title('95% Bootstrap Confidence Intervals')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_calibration_curve(cal_data, save_name="calibration_curve"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], '--', color='#555', linewidth=1, label='Perfectly Calibrated')
    ax.plot(cal_data["bin_centers"], cal_data["bin_accuracies"],
             'o-', color=CYAN, linewidth=2, markersize=8, label=f'OsteoNexus (ECE={cal_data["ece"]:.4f})')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_baseline_comparison(baseline_results, osteonexus_metrics, save_name="baseline_comparison"):
    models = list(baseline_results.keys()) + ["OsteoNexus"]
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.15
    colors = [CYAN, GREEN, AMBER, PURPLE, RED]

    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        values = []
        for key in baseline_results:
            values.append(baseline_results[key]["metrics"].get(metric, 0))
        values.append(osteonexus_metrics.get(metric, 0))
        ax.bar(x + i * width, values, width, label=metric.upper(), color=color, alpha=0.85)

    ax.set_xlabel('Model')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Baseline Comparison')
    ax.set_xticks(x + width * 2)
    model_labels = [baseline_results[k]["name"] for k in baseline_results] + ["OsteoNexus"]
    ax.set_xticklabels(model_labels, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_ablation_results(ablation_results, save_name="ablation_results"):
    names = list(ablation_results.keys())
    accs = [ablation_results[n]["metrics"]["accuracy"] for n in names]
    f1s = [ablation_results[n]["metrics"]["f1"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.barh(x - width/2, accs, width, label='Accuracy', color=CYAN, alpha=0.85)
    ax.barh(x + width/2, f1s, width, label='F1-Score', color=GREEN, alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Component Ablation Study')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_image_dimensions(dims_df, save_name="image_dimensions"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    healthy = dims_df[dims_df["label"] == 0]
    osteo = dims_df[dims_df["label"] == 1]

    axes[0].hist(healthy["width"], bins=20, alpha=0.7, color=CYAN, label="Healthy")
    axes[0].hist(osteo["width"], bins=20, alpha=0.7, color=RED, label="Osteoporosis")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Image Width Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(healthy["height"], bins=20, alpha=0.7, color=CYAN, label="Healthy")
    axes[1].hist(osteo["height"], bins=20, alpha=0.7, color=RED, label="Osteoporosis")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Image Height Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def plot_meta_learning_loss(losses, title="Meta-Learning Training Loss", save_name="meta_loss"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, color=CYAN, linewidth=2)
    ax.set_xlabel('Meta-Epoch')
    ax.set_ylabel('Average Query Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}.png")


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    print_header("OSTEONEXUS: ATTENTION-DRIVEN META-LEARNING FRAMEWORK")
    print("Osteoporosis Detection from Knee X-Rays")
    print(f"Results directory: {RESULTS_DIR}")

    tf.random.set_seed(PRIMARY_SEED)
    np.random.seed(PRIMARY_SEED)

    print_header("PHASE 1: DATA LOADING & EXPLORATORY ANALYSIS")

    for src in DATASET_SOURCES:
        print(f"Dataset: {src['name']}")
        print(f"  Author: {src['author']}")
        print(f"  Source: {src['url']}")
        print(f"  Annotation: {src['annotation']}")
        print(f"  Original classes: {src['classes']}")
        print(f"  Total images: {src['total_images']}")
    print()

    df = build_dataframe()
    print(f"Total images: {len(df)}")
    print(f"Normal: {(df['label']==0).sum()} | At-Risk (Osteopenia+Osteoporosis): {(df['label']==1).sum()}")
    if "original_class" in df.columns:
        for cls in df["original_class"].unique():
            print(f"  {cls}: {(df['original_class']==cls).sum()}")

    dims_df = get_image_dimensions(df)
    for label, name in [(0, "Normal"), (1, "At-Risk")]:
        sub = dims_df[dims_df["label"] == label]
        print(f"{name} - Width: median={sub['width'].median():.0f}, range=[{sub['width'].min()}-{sub['width'].max()}]")
        print(f"{name} - Height: median={sub['height'].median():.0f}, range=[{sub['height'].min()}-{sub['height'].max()}]")
    plot_image_dimensions(dims_df)

    print_header("PHASE 2: FEATURE EXTRACTION & DATA PREPARATION")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, df = prepare_data(PRIMARY_SEED)
    print(f"Feature dimension after PCA: {X_train.shape[1]}")
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print("Per-fold standardization applied (fit on train only) - addresses reviewer leakage concern")

    print_header("PHASE 3: OSTEONEXUS MODEL TRAINING")
    model = build_osteonexus_model(X_train.shape[1])
    model = compile_model(model, lr=OPTIMAL_LR)
    print(f"Model parameters: {model.count_params()}")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=OPTIMAL_EPOCHS)
    final_epoch = len(history.history['accuracy'])
    print(f"Training completed at epoch {final_epoch}")
    print(f"Final train acc: {history.history['accuracy'][-1]*100:.2f}% | val acc: {history.history['val_accuracy'][-1]*100:.2f}%")

    print_header("PHASE 4: EVALUATION WITH STATISTICAL RIGOR")
    eval_result = evaluate_model(model, X_test, y_test)
    m = eval_result["metrics"]
    ci = eval_result["confidence_intervals"]
    cal = eval_result["calibration"]

    print("\n--- Primary Metrics ---")
    for name in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"  {name.upper()}: {m.get(name, 0):.2f}%")
    print(f"  Brier Score: {m.get('brier_score', 0):.4f}")

    print("\n--- 95% Bootstrap Confidence Intervals (1000 resamples) ---")
    for name, vals in ci.items():
        print(f"  {name.upper()}: {vals['mean']:.2f}% [{vals['lower']:.2f}%, {vals['upper']:.2f}%]")

    print(f"\n  ECE: {cal['ece']:.4f}")

    cm = m["confusion_matrix"]
    print(f"\n--- Confusion Matrix ---")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    print("\n--- Clinical Benchmark Check ---")
    for bench, val in CLINICAL_BENCHMARKS.items():
        achieved = m.get(bench, 0)
        status = "PASS" if achieved >= val else "FAIL"
        print(f"  {bench.upper()}: {achieved:.2f}% vs {val}% -> [{status}]")

    plot_training_curves(history.history, save_name="osteonexus_training")
    plot_confusion_matrix(y_test, eval_result["y_pred"], save_name="osteonexus_confusion")
    plot_roc_curve(y_test, eval_result["y_prob"], m.get("auc", 0), save_name="osteonexus_roc")
    plot_metrics_vs_benchmarks(m, save_name="osteonexus_benchmarks")
    plot_confidence_intervals(ci, save_name="osteonexus_ci")
    plot_calibration_curve(cal, save_name="osteonexus_calibration")

    print_header("PHASE 5: BASELINE MODELS")
    baseline_results = run_all_baselines(X_train, y_train, X_test, y_test, seed=PRIMARY_SEED)
    plot_baseline_comparison(baseline_results, m, save_name="baseline_comparison")

    print_header("PHASE 6: STATISTICAL TESTS vs BASELINES")
    for name, bres in baseline_results.items():
        print(f"\n  OsteoNexus vs {bres['name']}:")
        delong = delong_test(y_test, eval_result["y_prob"], bres["y_prob"])
        print(f"    DeLong: z={delong['z_stat']:.4f}, p={delong['p_value']:.4f}")
        mcnemar = mcnemar_test(y_test, eval_result["y_pred"], bres["y_pred"])
        print(f"    McNemar: chi2={mcnemar['chi2']:.4f}, p={mcnemar['p_value']:.4f}")

    print_header("PHASE 7: COMPONENT ABLATION STUDY")
    ablation_results = run_ablation_study(
        X_train, y_train, X_val, y_val, X_test, y_test,
        seed=PRIMARY_SEED, epochs=60
    )
    plot_ablation_results(ablation_results, save_name="ablation_components")

    print_header("PHASE 8: FEATURE ABLATION STUDY")
    feature_ablation = run_feature_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test, seed=PRIMARY_SEED
    )

    print_header("PHASE 9: META-LEARNING (REPTILE)")
    meta_model = build_maml_model(X_train.shape[1])
    meta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    meta_losses = reptile_meta_train(
        meta_model, X_train, y_train,
        meta_epochs=MAML_META_EPOCHS, tasks_per_epoch=5
    )
    plot_meta_learning_loss(meta_losses, title="Reptile Meta-Training Loss", save_name="reptile_loss")
    meta_eval = evaluate_model(meta_model, X_test, y_test)
    mm = meta_eval["metrics"]
    print(f"\nReptile: Acc={mm['accuracy']:.2f}% | F1={mm['f1']:.2f}% | AUC={mm.get('auc',0):.2f}%")

    print_header("PHASE 10: PROTOTYPICAL NETWORK")
    proto_encoder = build_protonet_encoder(X_train.shape[1], embedding_dim=32)
    proto_losses = train_protonet(proto_encoder, X_train, y_train, epochs=30, tasks_per_epoch=10)
    plot_meta_learning_loss(proto_losses, title="ProtoNet Training Loss", save_name="protonet_loss")

    proto_accs = []
    for _ in range(30):
        sx, sy, qx, qy = create_episodic_task(X_test, y_test, k_shot=5, q_query=10)
        acc, _ = prototypical_network_eval(proto_encoder, sx, sy, qx, qy)
        proto_accs.append(acc * 100)
    print(f"\nProtoNet Few-Shot: {np.mean(proto_accs):.2f}% +/- {np.std(proto_accs):.2f}%")

    print_header("FINAL SUMMARY")
    all_results = {
        "osteonexus": {
            "metrics": {k: v for k, v in m.items() if k != "confusion_matrix"},
            "confidence_intervals": {k: v for k, v in ci.items()},
            "calibration_ece": cal["ece"],
            "confusion_matrix": cm.tolist(),
        },
        "baselines": {
            name: {"metrics": {k: v for k, v in res["metrics"].items() if k != "confusion_matrix"}}
            for name, res in baseline_results.items()
        },
        "meta_learning": {
            "reptile": {k: v for k, v in mm.items() if k != "confusion_matrix"},
            "protonet_accuracy_mean": float(np.mean(proto_accs)),
            "protonet_accuracy_std": float(np.std(proto_accs)),
        },
        "ablation": {
            name: {k: v for k, v in res["metrics"].items() if k != "confusion_matrix"}
            for name, res in ablation_results.items()
        },
        "config": {
            "lr": OPTIMAL_LR, "epochs": final_epoch,
            "seed": PRIMARY_SEED, "pca_dim": X_train.shape[1],
            "dataset_size": len(df),
            "dataset_sources": DATASET_SOURCES,
            "class_distribution": {
                "normal": int((df['label']==0).sum()),
                "osteopenia": int((df['original_class']=='Osteopenia').sum()) if 'original_class' in df.columns else 0,
                "osteoporosis": int((df['original_class']=='Osteoporosis').sum()) if 'original_class' in df.columns else 0,
            },
        }
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_DIR}/results.json")
    print(f"Plots saved to {RESULTS_DIR}/")

    print("\n--- Hypothesis Test ---")
    all_pass = all(m.get(b, 0) >= v for b, v in CLINICAL_BENCHMARKS.items())
    if all_pass:
        print("H0 REJECTED: OsteoNexus meets all clinical benchmarks.")
    else:
        print("H0 NOT REJECTED: Some benchmarks not met.")
        for b, v in CLINICAL_BENCHMARKS.items():
            if m.get(b, 0) < v:
                print(f"  FAILED: {b.upper()} = {m.get(b,0):.2f}% < {v}%")

    print("\n" + "=" * 70)
    print("  OSTEONEXUS EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
