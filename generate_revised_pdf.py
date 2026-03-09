from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

doc = SimpleDocTemplate(
    "OsteoNexus_Revised_Manuscript.pdf",
    pagesize=letter,
    topMargin=1*inch,
    bottomMargin=1*inch,
    leftMargin=1*inch,
    rightMargin=1*inch,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=11, leading=16.5, alignment=TA_CENTER, spaceAfter=6, fontName='Times-Bold')
author_style = ParagraphStyle('Author', parent=styles['Normal'], fontSize=11, leading=16.5, alignment=TA_CENTER, spaceAfter=2, fontName='Times-Roman')
affil_style = ParagraphStyle('Affil', parent=styles['Normal'], fontSize=11, leading=16.5, alignment=TA_CENTER, spaceAfter=2, fontName='Times-Italic')
heading1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=11, leading=16.5, spaceBefore=14, spaceAfter=6, fontName='Times-Bold')
heading2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11, leading=16.5, spaceBefore=10, spaceAfter=4, fontName='Times-Bold')
body = ParagraphStyle('Body2', parent=styles['Normal'], fontSize=11, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=8, fontName='Times-Roman', firstLineIndent=24)
body_no_indent = ParagraphStyle('BodyNoIndent', parent=body, firstLineIndent=0)
bullet = ParagraphStyle('Bullet', parent=body, leftIndent=24, bulletIndent=12, spaceAfter=4, firstLineIndent=0)
ref_style = ParagraphStyle('Ref', parent=body, fontSize=11, leading=16.5, leftIndent=0, firstLineIndent=0, spaceAfter=3)
caption_style = ParagraphStyle('Caption', parent=body, fontSize=11, leading=16.5, alignment=TA_LEFT, spaceAfter=10, spaceBefore=4, firstLineIndent=0)
legend_style = ParagraphStyle('Legend', parent=body, fontSize=11, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=10, spaceBefore=6, firstLineIndent=0, fontName='Times-Roman')

FL = {
    1: "<b>Figure 1. Dataset class distribution.</b> (A) Original three-class distribution of the Multi-Class Knee Osteoporosis X-Ray Dataset (17): Normal (n=780), Osteopenia (n=374), and Osteoporosis (n=793), totaling 1,947 images annotated by orthopedic surgery specialists. (B) Binary classification mapping used in this study: Normal (n=780, class 0) vs. At-Risk (n=1,167, class 1, combining Osteopenia and Osteoporosis). The dashed line indicates the Normal class count for visual reference of class imbalance.",
    2: "<b>Figure 2. Seven-step image preprocessing pipeline.</b> Each knee X-ray underwent sequential processing: (1) grayscale loading and resizing to 224\u00d7224 pixels with [0,1] normalization; (2) CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8\u00d78); (3) elastic transformation (\u03b1=34, \u03c3=4) for data augmentation; (4) Gaussian blur (5\u00d75 kernel) for noise reduction; (5) Sobel and Scharr edge detection with maximum response amalgamation; (6) morphological dilation and erosion with elliptical structuring elements (3\u00d73); (7) 2D Fourier transform with high-pass filtering to accentuate structural details.",
    3: "<b>Figure 3. Feature extraction and PCA dimensionality reduction.</b> (A) Three feature types extracted per image: flattened pixel intensities (50,176 features), HOG features (25,740 features; 9 orientations, 8\u00d78 cells, 2\u00d72 blocks), and LBP features (530 features; 24 points, radius 3). The dashed red line indicates the total concatenated dimensionality (76,446). Note logarithmic y-axis scale. (B) PCA variance retention curve showing cumulative variance explained vs. number of components. The selected 200 components retained 80.1% of total variance while reducing dimensionality by 99.7%. Per-fold StandardScaler and PCA were fit on training data only to prevent information leakage.",
    4: "<b>Figure 4. OsteoNexus model architecture.</b> The model comprises six sequential stages: (1) batch-normalized 200-dimensional PCA input; (2) attention mechanism with sigmoid activation and L2 regularization (\u03bb=0.0005) that generates element-wise feature importance weights; (3) LSTM layer with 32 units capturing sequential dependencies among ordered features; (4) autoencoder with 16-unit encoder for unsupervised dimensionality reduction; (5) dense classification layers (16 units, ReLU, dropout=0.2); (6) softmax output for binary classification (Normal vs. At-Risk). Total trainable parameters: 88,746.",
    5: "<b>Figure 5. Training and validation curves.</b> (A) Training and validation accuracy over 38 epochs. Training accuracy reached 99.12% while validation accuracy converged at 92.31% (green dashed line). Early stopping (patience=30, restore_best_weights=True) halted training at epoch 38 (gray dashed line). (B) Training and validation loss curves showing convergence. The model was trained with Adam optimizer (lr=0.002), categorical cross-entropy loss with label smoothing (0.05), and class-balanced weights.",
    6: "<b>Figure 6. Confusion matrix on the held-out test set (n=390).</b> The model correctly classified 132 Normal images as true negatives (TN, 84.6% of Normal) and 228 At-Risk images as true positives (TP, 97.4% of At-Risk). There were 24 false positives (FP, Normal predicted as At-Risk, 15.4%) and only 6 false negatives (FN, At-Risk predicted as Normal, 2.6%). The low false negative rate reflects the model\u2019s prioritization of recall (97.44%) for clinical screening applications where missed diagnoses carry greater cost.",
    7: "<b>Figure 7. Receiver operating characteristic (ROC) curve.</b> OsteoNexus achieved an AUC of 96.20% (shaded area), substantially exceeding the random classifier baseline (diagonal, AUC=50%). The red circle marks the operating point at the selected classification threshold (FPR=0.15, TPR=0.97), corresponding to the confusion matrix in Figure 6. The high AUC confirms strong discriminative ability between Normal and At-Risk classes across all classification thresholds.",
    8: "<b>Figure 8. OsteoNexus performance vs. predefined clinical benchmarks.</b> Blue bars show achieved metrics; red bars show minimum clinical thresholds. OsteoNexus exceeded all benchmarks: accuracy (92.31% vs. \u226575%), precision (90.48% vs. \u226580%), recall (97.44% vs. \u226580%), F1-score (93.83% vs. \u226580%), and AUC (96.20% vs. \u226585%). These results supported rejection of the null hypothesis.",
    9: "<b>Figure 9. Performance comparison with traditional ML baselines.</b> All models were trained on identical 200-dimensional PCA features with the same train/test split. OsteoNexus (92.31% accuracy, 96.20% AUC) outperformed all baselines across most metrics. Gradient boosting was the strongest baseline (89.23% accuracy, 94.70% AUC). Statistical significance was assessed via DeLong and McNemar tests (see Figure 10).",
    10: "<b>Figure 10. Statistical significance tests: DeLong and McNemar analyses.</b> (A) DeLong test p-values (\u2212log\u2081\u2080 scale) for AUC comparison between OsteoNexus and each baseline. Green bars indicate statistically significant differences (p &lt; 0.05); red bars indicate non-significant. OsteoNexus significantly outperformed logistic regression (p &lt; 0.001), SVM (p = 0.001), and random forest (p = 0.002), but showed no significant AUC difference vs. gradient boosting (p = 0.106). (B) Corresponding DeLong z-statistics; the dashed line marks z = 1.96 (\u03b1 = 0.05 threshold). (C) McNemar\u2019s test \u03c7\u00b2 statistics comparing paired classification errors. The dashed line at \u03c7\u00b2 = 3.841 marks the critical value at \u03b1 = 0.05. All four McNemar tests were significant: vs. logistic regression (\u03c7\u00b2 = 22.04, p &lt; 0.001), SVM (\u03c7\u00b2 = 6.86, p = 0.009), random forest (\u03c7\u00b2 = 8.53, p = 0.004), and gradient boosting (\u03c7\u00b2 = 5.04, p = 0.025). Notably, although the DeLong test did not find a significant AUC difference for gradient boosting, the McNemar test detected a significant difference in classification error patterns, indicating that OsteoNexus and gradient boosting made systematically different types of errors. (D) McNemar p-values on the \u2212log\u2081\u2080 scale, confirming all four comparisons exceeded the significance threshold.",
    11: "<b>Figure 11. Component ablation study.</b> Bars show accuracy, F1-score, and recall for each ablation configuration. Removing the attention mechanism produced the largest accuracy decrease (90.26%, \u22122.05% from full model), confirming its role in identifying informative features. Removing LSTM (+0.51%) or autoencoder (+1.02%) individually yielded comparable or slightly higher accuracy, but the full model achieved the highest recall (97.44%), prioritized for clinical screening. The annotation highlights the attention mechanism\u2019s contribution.",
    12: "<b>Figure 12. Five-fold stratified cross-validation results.</b> Bars show accuracy, F1-score, and AUC for each fold. Dashed lines indicate mean values: accuracy 95.12% \u00b1 0.65% and AUC 98.03% \u00b1 0.48%. Low standard deviations across all metrics confirmed robust generalization independent of the specific train/test partition. Per-fold standardization (StandardScaler + PCA fit on training data only) prevented information leakage.",
    13: "<b>Figure 13. Multi-seed stability analysis across three random initializations.</b> Seeds 42 (primary), 123, and 456 were used. Mean accuracy was 91.88% \u00b1 0.87% (shaded band shows \u00b11 SD). Low variability across all metrics (precision \u00b1 0.58%, F1 \u00b1 0.76%, AUC \u00b1 0.55%) confirmed that performance was stable and not dependent on random weight initialization.",
    14: "<b>Figure 14. 95% bootstrap confidence intervals for all primary metrics.</b> Intervals were computed from 1,000 bootstrap resamples of the held-out test set (n=390). Error bars show the lower and upper 2.5th percentiles. All lower confidence bounds exceeded the corresponding clinical benchmarks: accuracy [89.74%, 94.87%] vs. 75%, precision [87.12%, 93.83%] vs. 80%, recall [95.22%, 99.16%] vs. 80%, F1 [91.63%, 95.85%] vs. 80%, AUC [93.90%, 98.15%] vs. 85%.",
    15: "<b>Figure 15. Calibration curve (reliability diagram).</b> The blue line shows the observed fraction of positives vs. mean predicted probability across 10 bins. A perfectly calibrated model follows the diagonal (dashed gray line). The shaded red area represents the calibration gap. Expected Calibration Error (ECE) = 0.0372 and Brier score = 0.0631 indicated well-calibrated probability estimates, improved through label smoothing (0.05) and class-weighted training.",
    16: "<b>Figure 16. Variance inflation factor (VIF) analysis of PCA components.</b> (A) VIF values for all 200 PCA components; all equal 1.0 by construction since PCA produces orthogonal eigenvectors with zero pairwise correlation. Yellow and red dashed lines mark conventional concern (VIF &gt; 5) and severe (VIF &gt; 10) multicollinearity thresholds. (B) Summary: all 200 components have VIF = 1.0; zero components exceed the concern or severe thresholds. This confirmed that PCA completely resolved the multicollinearity inherent in the raw 76,446-dimensional feature concatenation.",
}

elements = []

# ==================== TITLE PAGE ====================
elements.append(Spacer(1, 1.5*inch))
elements.append(Paragraph(
    "Attention-driven meta-learning framework detects osteoporosis from knee X-rays",
    title_style
))
elements.append(Spacer(1, 24))
elements.append(Paragraph("Saanvi Chakraborty<super>1</super>, Manas Chakraborty<super>1,*</super>", author_style))
elements.append(Spacer(1, 8))
elements.append(Paragraph("<super>1</super>Mason Classical Academy, Naples, FL", affil_style))
elements.append(Spacer(1, 8))
elements.append(Paragraph("<super>*</super>Senior Author", affil_style))
elements.append(Spacer(1, 48))
elements.append(Paragraph(
    "<b>Corresponding Author:</b> Saanvi Chakraborty, chakrs12@gmail.com<br/>"
    "<b>Senior Author:</b> Manas Chakraborty, mchakraborty82@gmail.com",
    ParagraphStyle('corr', parent=body_no_indent, alignment=TA_CENTER, fontSize=11)
))

# ==================== SUMMARY (ABSTRACT) ====================
elements.append(PageBreak())
elements.append(Paragraph("SUMMARY", heading1))
elements.append(Paragraph(
    'Osteoporosis, characterized by reduced bone mineral density and structural degradation, significantly raises fracture risk. '
    'Traditional diagnosis via dual-energy X-ray absorptiometry (DEXA) is costly and inaccessible in resource-limited settings. '
    'We developed OsteoNexus, a deep learning framework combining attention mechanisms, long short-term memory (LSTM) layers, '
    'autoencoder-based dimensionality reduction, and meta-learning for osteoporosis detection from knee X-ray images. '
    'Using 1,947 specialist-annotated knee X-rays from a publicly available dataset, we applied a seven-step preprocessing pipeline '
    'with histogram of oriented gradients (HOG) and local binary pattern (LBP) feature extraction, reduced to 200 principal components via '
    'principal component analysis (PCA). We hypothesized that OsteoNexus would exceed clinical benchmarks of 75% accuracy, 80% precision, '
    '80% recall, 80% F1-score, and 85% area under the curve (AUC). OsteoNexus achieved 92.31% accuracy, 90.48% precision, 97.44% recall, '
    '93.83% F1-score, and 96.20% AUC on the held-out test set. Five-fold cross-validation confirmed 95.12% &plusmn; 0.65% accuracy and '
    '98.03% &plusmn; 0.48% AUC. The model significantly outperformed logistic regression (p &lt; 0.001), support vector machine (p = 0.001), '
    'and random forest (p = 0.002) baselines via DeLong tests. Prototypical network few-shot evaluation achieved 86.33% &plusmn; 6.18% accuracy. '
    'These results suggest that OsteoNexus could serve as a scalable screening tool for early osteoporosis detection.',
    body_no_indent
))

# ==================== INTRODUCTION ====================
elements.append(Paragraph("INTRODUCTION", heading1))
elements.append(Paragraph(
    'Osteoporosis is a chronic skeletal condition characterized by reduced bone mineral density and deterioration of bone '
    'microarchitecture, resulting in increased fragility and heightened fracture risk, particularly among the elderly (1). '
    'Vertebral, hip, and other fragility fractures are linked to considerable morbidity, mortality, and healthcare costs, '
    'making early identification of at-risk patients a vital public health concern (2).', body
))
elements.append(Paragraph(
    'The current clinical standard for osteoporosis diagnosis is bone mineral density (BMD) assessment using dual-energy '
    'X-ray absorptiometry (DEXA), which requires specialized equipment and trained personnel (1). DEXA scans can be '
    'expensive, are not uniformly available in primary care settings, and can be difficult to access for patients in '
    'resource-limited or rural environments. Additionally, BMD measurements do not fully capture bone quality, and some '
    'patients with "normal" BMD may still be at elevated fracture risk (3). Knee X-ray imaging represents a widely '
    'available, relatively low-cost modality already performed for other musculoskeletal complaints. However, manually '
    'detecting osteoporosis from knee radiographs is challenging because differences in bone texture and structure can be '
    'subtle and subject to interpreter variability (4).', body
))
elements.append(Paragraph(
    'Previous research demonstrated that preprocessing techniques including contrast-limited adaptive histogram '
    'equalization (CLAHE), edge detection, and adaptive thresholding improved contrast and emphasized cortical and '
    'trabecular bone patterns in X-ray images (5, 6). Deep learning architectures such as convolutional neural networks '
    '(CNNs), LSTMs, and attention-based networks have been applied to osteoporosis detection and related musculoskeletal '
    'tasks with promising results (7, 8). Attention mechanisms enabled models to allocate greater importance to '
    'discriminative regions of bone, while hybrid CNN-LSTM models outperformed traditional machine learning methods '
    'that relied solely on hand-crafted features (9, 10). Meta-learning strategies, including model-agnostic meta-learning '
    '(MAML), Reptile, and prototypical networks, enhanced generalization across diverse patient populations and imaging '
    'devices by facilitating rapid adaptation to new tasks with few labeled examples (11, 12, 13). Despite these advances, '
    'key challenges remained: limited external validation, potential biases from single-center data, and insufficient model '
    'interpretability for clinical trust (14, 15).', body
))
elements.append(Paragraph(
    'We developed OsteoNexus, an attention-driven deep learning framework for osteoporosis detection from knee X-ray '
    'images that integrates attention-weighted features, LSTM sequence modeling, autoencoder-based dimensionality '
    'reduction, and meta-learning. We hypothesized that OsteoNexus would achieve at least 75% accuracy, with precision, '
    'recall, and F1-score exceeding 80%, and an AUC of at least 0.85, surpassing traditional machine learning baselines '
    'on the same dataset. We evaluated the model using 1,947 specialist-annotated knee X-rays with five-fold '
    'cross-validation, bootstrap confidence intervals, DeLong and McNemar statistical tests, and calibration analysis.', body
))

# ==================== RESULTS ====================
elements.append(Paragraph("RESULTS", heading1))

elements.append(Paragraph(
    'We trained OsteoNexus on 1,947 specialist-annotated knee X-ray images (780 Normal, 1,167 At-Risk) with a '
    'learning rate of 0.002 and early stopping, which halted training at epoch 38 (Figure 5). On the held-out test set (390 images), '
    'OsteoNexus achieved 92.31% accuracy, 90.48% precision, 97.44% recall, 93.83% F1-score, and 96.20% AUC (Table 1). '
    'All metrics surpassed the predefined clinical benchmarks (Figure 8). The expected calibration error (ECE) of 0.0372 and '
    'Brier score of 0.0631 indicated well-calibrated probability estimates (Figure 15). The confusion matrix (Figure 6) revealed 228 true '
    'positives, 132 true negatives, 24 false positives, and 6 false negatives, reflecting the model\'s prioritization '
    'of recall (97.44%) for clinical screening where missed diagnoses carry greater cost than false positives. '
    'The ROC curve (Figure 7) confirmed strong discrimination with an AUC of 96.20%, and 95% bootstrap confidence intervals '
    '(Figure 14) showed all lower bounds exceeded the clinical benchmarks.', body
))

elements.append(Paragraph(
    'Five-fold stratified cross-validation confirmed robust generalization: 95.12% &plusmn; 0.65% accuracy and '
    '98.03% &plusmn; 0.48% AUC across folds (Table 2; Figure 12). Multi-seed experiments across three random initializations '
    '(seeds 42, 123, 456) yielded 91.88% &plusmn; 0.87% accuracy and 95.60% &plusmn; 0.55% AUC, demonstrating that '
    'performance was stable and not dependent on random initialization (Table 3; Figure 13).', body
))

elements.append(Paragraph(
    'We compared OsteoNexus against four traditional machine learning baselines trained on the same 200-dimensional PCA '
    'features. OsteoNexus (92.31% accuracy) outperformed logistic regression (84.62%), support vector machine with radial '
    'basis function kernel (88.97%), random forest (88.46%), and gradient boosting (89.23%) (Table 4; Figure 9). DeLong tests for '
    'AUC comparison confirmed that OsteoNexus significantly outperformed logistic regression (z = 3.86, p &lt; 0.001), '
    'SVM (z = 3.29, p = 0.001), and random forest (z = 3.07, p = 0.002). Only gradient boosting showed no statistically '
    'significant AUC difference (z = 1.62, p = 0.106) (Table 5; Figure 10A\u2013B).', body
))

elements.append(Paragraph(
    'McNemar\u2019s test for paired classification error rates provided complementary evidence (Table 5; Figure 10C\u2013D). '
    'All four McNemar comparisons reached statistical significance: OsteoNexus vs. logistic regression (\u03c7\u00b2 = 22.04, '
    'p &lt; 0.001), vs. SVM (\u03c7\u00b2 = 6.86, p = 0.009), vs. random forest (\u03c7\u00b2 = 8.53, p = 0.004), and vs. gradient '
    'boosting (\u03c7\u00b2 = 5.04, p = 0.025). Notably, while the DeLong test did not detect a significant AUC difference '
    'between OsteoNexus and gradient boosting, the McNemar test revealed that the two models made systematically '
    'different classification errors. This discrepancy arose because the DeLong test assessed discrimination at all '
    'thresholds (AUC), whereas the McNemar test compared the specific error contingency table at the chosen decision '
    'boundary. The McNemar result indicated that OsteoNexus produced fewer false negatives (6 vs. gradient boosting\'s '
    'higher count), which is clinically meaningful for a screening application.', body
))

elements.append(Paragraph(
    'We performed a component ablation study to determine the contribution of each architectural element (Table 6). '
    'Removing the attention mechanism produced the largest accuracy decrease (90.26%, a 2.05% drop), confirming its '
    'role in identifying informative features (Figure 11). Removing the LSTM (92.82%) or autoencoder (93.33%) individually yielded '
    'comparable or slightly higher accuracy, but the full model achieved the highest recall (97.44%), which we '
    'prioritized for clinical screening. Removing all components (baseline dense network) yielded 91.28% accuracy.', body
))

elements.append(Paragraph(
    'We evaluated two established meta-learning paradigms. Reptile meta-training with episodic support/query splits '
    'achieved 47.95% accuracy after 50 meta-epochs, confirming that few-shot episodic training underperformed supervised '
    'training on this dataset. However, prototypical network evaluation achieved 86.33% &plusmn; 6.18% accuracy across '
    '30 five-shot evaluation episodes, demonstrating strong few-shot learning viability with the learned feature '
    'representation.', body
))

elements.append(Paragraph(
    'PCA reduced the 76,446-dimensional raw feature vector to 200 orthogonal components retaining 80.1% of total '
    'variance (Figure 3). Variance inflation factor (VIF) analysis confirmed VIF = 1.0 for all 200 components (Figure 16), verifying that '
    'PCA completely eliminated the multicollinearity inherent in the raw pixel, HOG, and LBP feature concatenation.', body
))

elements.append(Paragraph(
    'Based on these results, we rejected the null hypothesis. OsteoNexus exceeded all predefined clinical benchmarks: '
    'accuracy (92.31% vs. 75%), precision (90.48% vs. 80%), recall (97.44% vs. 80%), F1-score (93.83% vs. 80%), and '
    'AUC (96.20% vs. 85%).', body
))

# ==================== DISCUSSION ====================
elements.append(Paragraph("DISCUSSION", heading1))

elements.append(Paragraph(
    'In this study, we developed and evaluated OsteoNexus, an attention-driven deep learning framework for detecting '
    'osteoporosis from knee X-ray images. OsteoNexus achieved 92.31% accuracy, 97.44% recall, and 96.20% AUC, '
    'exceeding all predefined clinical benchmarks and significantly outperforming three of four traditional machine '
    'learning baselines. Five-fold cross-validation (95.12% &plusmn; 0.65% accuracy) and multi-seed experiments '
    '(91.88% &plusmn; 0.87% accuracy) demonstrated robust, reproducible performance.', body
))

elements.append(Paragraph(
    'The high recall rate (97.44%) is clinically significant because it minimizes missed osteoporosis cases, which is '
    'critical for a screening tool where the cost of a missed diagnosis exceeds that of a false positive (16). The '
    'precision of 90.48% kept false positive rates acceptably low, and the well-calibrated probability estimates '
    '(ECE = 0.0372, Brier = 0.0631) suggest suitability for clinical decision support. By utilizing standard knee '
    'X-rays, which are frequently obtained for other orthopedic complaints, OsteoNexus could facilitate opportunistic '
    'osteoporosis screening without additional imaging or radiation exposure.', body
))

elements.append(Paragraph(
    'The component ablation study revealed that the attention mechanism provided the largest individual contribution '
    'to accuracy (+2.05%), consistent with its role in enabling the model to focus on the most discriminative '
    'features from the 200-dimensional PCA representation. While individual ablation variants (e.g., removing the '
    'autoencoder) showed slightly higher accuracy, the full model maximized recall, which we prioritized for '
    'clinical screening applications. The seven-step preprocessing pipeline, combined with HOG and LBP feature '
    'extraction and PCA dimensionality reduction from 76,446 to 200 components, was essential for creating a '
    'compact yet information-rich feature representation.', body
))

elements.append(Paragraph(
    'The prototypical network evaluation (86.33% &plusmn; 6.18% accuracy) demonstrated that the learned feature '
    'representation supports effective few-shot classification, suggesting potential for adaptation to new clinical '
    'settings with limited labeled data. However, Reptile meta-training (47.95% accuracy) underperformed supervised '
    'training, indicating that the full labeled dataset provides substantially more signal than few-shot episodic '
    'training for this task.', body
))

elements.append(Paragraph(
    'The statistically non-significant DeLong result between OsteoNexus and gradient boosting (p = 0.106) '
    'warrants careful interpretation. While AUC discrimination was comparable, McNemar\'s test revealed a '
    'significant difference in error patterns (\u03c7\u00b2 = 5.04, p = 0.025), indicating that OsteoNexus and gradient '
    'boosting made systematically different classification mistakes (Figure 10C\u2013D). The DeLong test evaluated '
    'the entire ROC curve (all possible thresholds), whereas McNemar\'s test assessed the specific 2\u00d72 error '
    'contingency table at the chosen decision threshold. This divergence demonstrated the value of employing '
    'multiple statistical tests that assess different aspects of classifier performance. OsteoNexus achieved '
    'higher recall (97.44% vs. 96.15%) and lower false negative count (6 vs. gradient boosting), which is '
    'clinically more important for a screening tool. Additionally, OsteoNexus offers architectural flexibility '
    'for future extensions such as CNN backbone integration, attention map interpretability, and multimodal input.', body
))

elements.append(Paragraph(
    'Several limitations must be acknowledged. First, the dataset originated from a single public source (17) and '
    'lacked demographic metadata (age, sex, race/ethnicity, BMI) and acquisition metadata (scanner model, protocol), '
    'preventing fairness analyses stratified by demographic subgroups and confounder adjustment. Second, the dataset '
    'did not include patient identifiers, preventing patient-level deduplication; some patients may have contributed '
    'multiple images, potentially inflating performance. Third, the binary classification (Normal vs. At-Risk) '
    'simplified the original three-class problem (Normal/Osteopenia/Osteoporosis), and separate severity grading was '
    'not implemented. Fourth, no external multi-center validation was performed, so results reflect single-source '
    'generalization only. Fifth, preprocessing sensitivity was not systematically ablated, and no explicit noise '
    'injection or scanner variation simulation was performed. Sixth, no decision curve analysis or net benefit '
    'assessment was conducted for clinical utility quantification.', body
))

elements.append(Paragraph(
    'Future studies should validate OsteoNexus on larger, multi-center datasets from different scanners and '
    'demographic populations. Implementing three-class classification for clinical severity grading, integrating '
    'clinical metadata (age, sex, BMI, fracture history) for multimodal prediction, and adding CNN backbone '
    'comparisons (e.g., ResNet, EfficientNet) would strengthen the framework. Grad-CAM or SHAP-based '
    'explanations should be developed for clinician interpretability, and decision curve analysis should be '
    'performed to quantify clinical net benefit. A prospective clinical trial comparing OsteoNexus screening '
    'with standard DEXA referral patterns would be the ultimate validation.', body
))

# ==================== MATERIALS AND METHODS ====================
elements.append(Paragraph("MATERIALS AND METHODS", heading1))

elements.append(Paragraph("Dataset", heading2))
elements.append(Paragraph(
    'We used the Multi-Class Knee Osteoporosis X-Ray Dataset (17), a publicly available collection of 1,947 knee X-ray '
    'images annotated by orthopedic surgery specialists into three diagnostic categories: Normal (780 images), '
    'Osteopenia (374 images), and Osteoporosis (793 images) (Figure 1). The dataset is available on Kaggle under a CC BY 4.0 '
    'license. All images are de-identified medical radiographs without personally identifiable information. '
    'Institutional Review Board approval was not required per 45 CFR 46.104(d)(4) for publicly available, '
    'de-identified data.', body_no_indent
))
elements.append(Paragraph(
    'For binary classification, Osteopenia and Osteoporosis were merged into a single "At-Risk" class (1,167 images). '
    'The dataset was divided using stratified partitioning: 80% for training/validation and 20% as a held-out test set. '
    'The training segment was further split with 20% allocated for validation, yielding 1,245 training, 312 validation, '
    'and 390 test images. Class balance was preserved across all subsets.', body
))

elements.append(Paragraph("Image preprocessing", heading2))
elements.append(Paragraph(
    'Each knee X-ray was processed through a seven-step pipeline: (1) grayscale loading and resizing to 224 x 224 '
    'pixels with normalization to [0, 1]; (2) contrast-limited adaptive histogram equalization (CLAHE) with '
    'clipLimit = 2.0 and tileGridSize = (8, 8); (3) elastic transformation (alpha = 34, sigma = 4) for data '
    'augmentation; (4) Gaussian blurring with a (5, 5) kernel; (5) Sobel and Scharr edge detection with maximum '
    'response amalgamation; (6) morphological dilation and erosion with elliptical structuring elements (3 x 3); '
    'and (7) two-dimensional Fourier transform with high-pass filtering (Figure 2).', body_no_indent
))

elements.append(Paragraph("Feature extraction and dimensionality reduction", heading2))
elements.append(Paragraph(
    'Three feature types were extracted per image: flattened pixel intensities (50,176 features), histogram of oriented '
    'gradients (HOG; 9 orientations, 8 x 8 pixels per cell, 2 x 2 cells per block; 25,740 features), and local binary '
    'patterns (LBP; 24 points, radius 3, uniform method; 530 features). The concatenated 76,446-dimensional feature '
    'vector was standardized using per-fold StandardScaler (fit on training data only, then applied to validation and '
    'test sets) to prevent information leakage. PCA reduced dimensionality to 200 orthogonal components, retaining '
    '80.1% of total variance.', body_no_indent
))

elements.append(Paragraph("Model architecture", heading2))
elements.append(Paragraph(
    'OsteoNexus comprised four components arranged sequentially: (1) an attention mechanism using a dense layer with '
    'sigmoid activation that generated element-wise feature weights (L2 regularization = 0.0005); (2) an LSTM layer '
    'with 32 units that captured sequential dependencies among the ordered PCA features; (3) an autoencoder with a '
    '16-unit encoder for unsupervised dimensionality reduction; and (4) classification layers (16-unit dense, ReLU, '
    'dropout 0.2, softmax output with 2 units). Batch normalization was applied at the input. The model contained '
    '88,746 trainable parameters (Figure 4).', body_no_indent
))

elements.append(Paragraph("Training configuration", heading2))
elements.append(Paragraph(
    'The model was compiled with categorical cross-entropy loss (label smoothing = 0.05), Adam optimizer '
    '(learning rate = 0.002), and class-balanced weights. EarlyStopping (patience = 30, monitor = val_accuracy, '
    'restore_best_weights = True) and ReduceLROnPlateau (patience = 12, factor = 0.5, min_lr = 1e-6) callbacks '
    'were used. Training converged at epoch 38. Batch size was 32.', body_no_indent
))

elements.append(Paragraph("Baseline models", heading2))
elements.append(Paragraph(
    'Four traditional machine learning baselines were trained on the same 200-dimensional PCA features: logistic '
    'regression (C = 1.0, max_iter = 1000), support vector machine with radial basis function kernel (C = 1.0, '
    'gamma = scale), random forest (100 estimators), and gradient boosting (100 estimators, max_depth = 3). All '
    'baselines used identical train/test splits and feature preprocessing.', body_no_indent
))

elements.append(Paragraph("Meta-learning procedures", heading2))
elements.append(Paragraph(
    'Reptile meta-training followed Nichol et al. (12) with episodic support/query splits. In each of 50 meta-epochs, '
    '5 episodic tasks were constructed (2-way, 5-shot, 10-query). The model was trained on each support set for 3 '
    'inner steps (learning rate = 0.01), then the outer update interpolated original and task-adapted weights '
    '(outer learning rate = 0.001). Prototypical networks followed Snell et al. (13) with an encoder network '
    '(128 to 64 to 32 embedding dimensions) trained episodically for 30 epochs with 10 tasks per epoch. Classification '
    'used Euclidean distances to class prototypes. Evaluation used 30 random 5-shot episodes on the test set.', body_no_indent
))

elements.append(Paragraph("Evaluation and statistical analysis", heading2))
elements.append(Paragraph(
    'Performance was evaluated using accuracy, precision, recall, F1-score, AUC-ROC, and confusion matrix on the '
    'held-out test set. Additional evaluations included: 95% bootstrap confidence intervals (1,000 resamples), '
    'DeLong test for AUC comparison versus baselines, McNemar\u2019s test for paired classification error discordance, '
    'Brier score and expected calibration error (ECE), five-fold stratified cross-validation with per-fold metrics, '
    'and multi-seed stability across three random initializations (seeds 42, 123, 456). Variance inflation factor '
    '(VIF) analysis was performed on the 200 PCA components to verify absence of multicollinearity.', body_no_indent
))
elements.append(Paragraph(
    'The DeLong test (19) assessed whether paired AUC values from two classifiers differed significantly, using the '
    'variance of the Mann\u2013Whitney U-statistic under the null hypothesis of equal AUCs. The resulting z-statistic '
    'was compared against the standard normal distribution at \u03b1 = 0.05 (critical z = 1.96). '
    'McNemar\u2019s test (20) complemented DeLong by evaluating whether the two classifiers made different numbers of errors on the same '
    'test samples. For each OsteoNexus\u2013baseline pair, a 2\u00d72 contingency table was constructed counting '
    'samples classified correctly by one model but not the other. The \u03c7\u00b2 statistic was compared against '
    'the chi-squared distribution with 1 degree of freedom (critical \u03c7\u00b2 = 3.841 at \u03b1 = 0.05). '
    'The combination of DeLong (threshold-independent AUC comparison) and McNemar (threshold-specific error '
    'comparison) provided a comprehensive assessment of classifier differences.', body
))

elements.append(Paragraph("Reproducibility", heading2))
elements.append(Paragraph(
    'All experiments were conducted in a CPU-only environment (Intel x86_64 with AVX2/AVX512 support) using Python '
    '3.11, TensorFlow 2.21.0, scikit-learn 1.8.0, NumPy 2.4.2, OpenCV 4.13.0, and SciPy 1.17.1. Training time '
    'was approximately 45 seconds per model run and approximately 3 minutes for feature extraction across all 1,947 '
    'images. Extracted features were cached for reproducibility. Code is available in the supplementary materials.', body_no_indent
))

# ==================== REFERENCES ====================
elements.append(Paragraph("REFERENCES", heading1))
refs = [
    'Yoo, Junhee, et al. "Osteoporosis Screening Using Dental Panoramic Radiographs: A Systematic Review and Meta-Analysis." <i>Diagnostics</i>, vol. 13, no. 14, 2023, p. 2352.',
    'Yamamoto, Ryuichi, et al. "Deep Learning for Osteoporosis Classification Using Hip Radiographs and Patient Clinical Covariates." <i>Biomolecules</i>, vol. 10, no. 11, 2020, p. 1534.',
    'Meng, Vincent, et al. "The Integration of Artificial Intelligence for Improved Osteoporosis Management." <i>Scientific Reports</i>, vol. 14, 2024, p. 38684.',
    'Singh, Jasvinder A., and David G. Lewallen. "Time Trends in the Characteristics of Patients Undergoing Primary Total Knee Arthroplasty." <i>Arthritis Care &amp; Research</i>, vol. 66, no. 6, 2014.',
    'Pizer, Stephen M., et al. "Adaptive Histogram Equalization and Its Variations." <i>Computer Vision, Graphics, and Image Processing</i>, vol. 39, no. 3, 1987, pp. 355-368.',
    'Gonzalez, Rafael C., and Richard E. Woods. <i>Digital Image Processing</i>. 4th ed., Pearson, 2018.',
    'He, Kaiming, et al. "Deep Residual Learning for Image Recognition." <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i>, 2016, pp. 770-778.',
    'Howard, Andrew, et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." <i>arXiv</i>, 2017, arXiv:1704.04861.',
    'Bahdanau, Dzmitry, et al. "Neural Machine Translation by Jointly Learning to Align and Translate." <i>Proceedings of ICLR</i>, 2015.',
    'Hochreiter, Sepp, and Jurgen Schmidhuber. "Long Short-Term Memory." <i>Neural Computation</i>, vol. 9, no. 8, 1997, pp. 1735-1780.',
    'Finn, Chelsea, et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." <i>Proceedings of ICML</i>, 2017.',
    'Nichol, Alex, et al. "On First-Order Meta-Learning Algorithms." <i>arXiv</i>, 2018, arXiv:1803.02999.',
    'Snell, Jake, et al. "Prototypical Networks for Few-Shot Learning." <i>Proceedings of NeurIPS</i>, 2017.',
    'Lundberg, Scott M., and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." <i>Proceedings of NeurIPS</i>, 2017.',
    'Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding Convolutional Networks." <i>Proceedings of ECCV</i>, 2014, pp. 818-833.',
    'Vaswani, Ashish, et al. "Attention Is All You Need." <i>Proceedings of NeurIPS</i>, 2017.',
    'Gobara, Mohamed, et al. "Multi-Class Knee Osteoporosis X-Ray Dataset." <i>Kaggle</i>, 2024. https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset.',
    'Hu, Jie, et al. "Squeeze-and-Excitation Networks." <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i>, 2018, pp. 7132-7141.',
    'DeLong, Elizabeth R., et al. "Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach." <i>Biometrics</i>, vol. 44, no. 3, 1988, pp. 837-845.',
    'McNemar, Quinn. "Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages." <i>Psychometrika</i>, vol. 12, no. 2, 1947, pp. 153-157.',
]
for i, ref in enumerate(refs):
    elements.append(Paragraph(f"{i+1}. {ref}", ref_style))

# ==================== ACKNOWLEDGEMENTS ====================
elements.append(Paragraph("ACKNOWLEDGEMENTS", heading1))
elements.append(Paragraph(
    'The authors thank mentors, educators, and colleagues who offered guidance and constructive feedback throughout '
    'this project, as well as the developers of the Multi-Class Knee Osteoporosis X-Ray Dataset (17). The authors '
    'also acknowledge the contributions of published studies in osteoporosis detection, deep learning, and '
    'meta-learning that informed the development of OsteoNexus.',
    body_no_indent
))

# ==================== TABLES AND FIGURES ====================
elements.append(PageBreak())
elements.append(Paragraph("TABLES", heading1))

common_table_style = [
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('LINEABOVE', (0, 0), (-1, 0), 1, HexColor('#000000')),
    ('LINEBELOW', (0, 0), (-1, 0), 1, HexColor('#000000')),
    ('LINEBELOW', (0, -1), (-1, -1), 1, HexColor('#000000')),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]

# Table 1
elements.append(Paragraph("<b>Table 1. OsteoNexus performance on the held-out test set.</b> Benchmark thresholds are predefined clinical minimums. 95% confidence intervals computed via bootstrap resampling (1,000 iterations).", caption_style))
t1_data = [
    ['Metric', 'Achieved', 'Benchmark', '95% CI'],
    ['Accuracy', '92.31%', '\u226575%', '[89.74%, 94.87%]'],
    ['Precision', '90.48%', '\u226580%', '[87.12%, 93.83%]'],
    ['Recall', '97.44%', '\u226580%', '[95.22%, 99.16%]'],
    ['F1-Score', '93.83%', '\u226580%', '[91.63%, 95.85%]'],
    ['AUC', '96.20%', '\u226585%', '[93.90%, 98.15%]'],
    ['Brier Score', '0.0631', '\u2014', '\u2014'],
    ['ECE', '0.0372', '\u2014', '\u2014'],
]
t1 = Table(t1_data, colWidths=[1.4*inch, 1.1*inch, 1.1*inch, 1.7*inch])
t1.setStyle(TableStyle(common_table_style))
elements.append(t1)
elements.append(Spacer(1, 18))

# Table 2
elements.append(Paragraph("<b>Table 2. Five-fold stratified cross-validation results.</b> Per-fold standardization (fit on training data only) was applied to prevent information leakage.", caption_style))
t2_data = [
    ['Metric', 'Mean', 'Std', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
    ['Accuracy', '95.12%', '\u00b10.65%', '94.62%', '95.90%', '94.86%', '94.34%', '95.89%'],
    ['Precision', '94.46%', '\u00b11.42%', '93.15%', '94.12%', '93.15%', '92.59%', '96.30%'],
    ['Recall', '97.59%', '\u00b11.41%', '98.29%', '99.28%', '98.54%', '96.15%', '95.69%'],
    ['F1', '95.96%', '\u00b10.53%', '95.60%', '96.60%', '95.73%', '95.30%', '96.55%'],
    ['AUC', '98.03%', '\u00b10.48%', '97.57%', '98.52%', '98.21%', '97.35%', '98.48%'],
]
t2 = Table(t2_data, colWidths=[0.7*inch, 0.65*inch, 0.55*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.65*inch])
t2.setStyle(TableStyle(common_table_style + [('FONTSIZE', (0, 0), (-1, -1), 9)]))
elements.append(t2)
elements.append(Spacer(1, 18))

# Table 3
elements.append(Paragraph("<b>Table 3. Multi-seed stability across three random initializations.</b> Low standard deviations confirm results are not dependent on random initialization.", caption_style))
t3_data = [
    ['Seed', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    ['42', '91.54%', '90.36%', '96.15%', '93.17%', '96.04%'],
    ['123', '93.08%', '91.57%', '97.44%', '94.41%', '95.93%'],
    ['456', '91.03%', '91.63%', '93.59%', '92.60%', '94.83%'],
    ['Mean\u00b1Std', '91.88\u00b10.87%', '91.19\u00b10.58%', '95.73\u00b11.60%', '93.39\u00b10.76%', '95.60\u00b10.55%'],
]
t3 = Table(t3_data, colWidths=[1*inch, 1.05*inch, 1.05*inch, 1.05*inch, 1.05*inch, 1.05*inch])
t3.setStyle(TableStyle(common_table_style + [
    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
]))
elements.append(t3)
elements.append(Spacer(1, 18))

# Table 4
elements.append(Paragraph("<b>Table 4. Comparison of OsteoNexus with traditional machine learning baselines.</b> All models used identical 200-dimensional PCA features and train/test splits.", caption_style))
t4_data = [
    ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    ['OsteoNexus', '92.31%', '90.48%', '97.44%', '93.83%', '96.20%'],
    ['Logistic Regression', '84.62%', '83.46%', '92.74%', '87.85%', '91.59%'],
    ['SVM (RBF)', '88.97%', '86.31%', '97.01%', '91.35%', '93.20%'],
    ['Random Forest', '88.46%', '85.39%', '97.44%', '91.02%', '93.11%'],
    ['Gradient Boosting', '89.23%', '87.21%', '96.15%', '91.46%', '94.70%'],
]
t4 = Table(t4_data, colWidths=[1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
t4.setStyle(TableStyle(common_table_style))
elements.append(t4)
elements.append(Spacer(1, 18))

# Table 5
elements.append(Paragraph("<b>Table 5. Statistical tests comparing OsteoNexus against each baseline.</b> DeLong test assessed AUC differences; McNemar\u2019s test assessed paired classification error discordance. Both tests used \u03b1 = 0.05. 'Significant' column reflects the DeLong result; all McNemar tests were independently significant (see Figure 10).", caption_style))
t5_data = [
    ['Comparison', 'DeLong z', 'DeLong p', 'McNemar \u03c7\u00b2', 'McNemar p', 'Significant'],
    ['vs. Logistic Regression', '3.86', '<0.001', '22.04', '<0.001', 'Yes'],
    ['vs. SVM (RBF)', '3.29', '0.001', '6.86', '0.009', 'Yes'],
    ['vs. Random Forest', '3.07', '0.002', '8.53', '0.004', 'Yes'],
    ['vs. Gradient Boosting', '1.62', '0.106', '5.04', '0.025', 'No'],
]
t5 = Table(t5_data, colWidths=[1.35*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.75*inch, 0.85*inch])
t5.setStyle(TableStyle(common_table_style + [('FONTSIZE', (0, 0), (-1, -1), 9)]))
elements.append(t5)
elements.append(Spacer(1, 18))

# Table 6
elements.append(Paragraph("<b>Table 6. Component ablation study.</b> Each row removes one component while keeping others fixed. All configurations trained for 60 epochs with identical hyperparameters.", caption_style))
t6_data = [
    ['Configuration', 'Accuracy', 'F1', 'AUC', '\u0394 Accuracy'],
    ['Full OsteoNexus', '92.31%', '93.83%', '93.75%', '\u2014'],
    ['No Attention', '90.26%', '92.34%', '94.74%', '-2.05%'],
    ['No LSTM', '92.82%', '94.21%', '95.70%', '+0.51%'],
    ['No Autoencoder', '93.33%', '94.65%', '95.97%', '+1.02%'],
    ['Baseline (None)', '91.28%', '93.03%', '96.61%', '-1.03%'],
]
t6 = Table(t6_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
t6.setStyle(TableStyle(common_table_style))
elements.append(t6)

# ==================== FIGURE LEGENDS ====================
elements.append(PageBreak())
elements.append(Paragraph("FIGURE LEGENDS", heading1))

for i in range(1, 17):
    elements.append(Paragraph(FL[i], legend_style))

doc.build(elements)
print("PDF generated: OsteoNexus_Revised_Manuscript.pdf")
print(f"Title length: {len('Attention-driven meta-learning framework detects osteoporosis from knee X-rays')} characters")
