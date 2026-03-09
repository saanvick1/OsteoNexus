from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

doc = SimpleDocTemplate(
    "OsteoNexus_Response_to_Reviewer.pdf",
    pagesize=letter,
    topMargin=1*inch,
    bottomMargin=1*inch,
    leftMargin=1*inch,
    rightMargin=1*inch,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=11, leading=16.5, alignment=TA_CENTER, spaceAfter=6, fontName='Times-Bold')
heading1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=11, leading=16.5, spaceBefore=14, spaceAfter=6, fontName='Times-Bold')
heading2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11, leading=16.5, spaceBefore=10, spaceAfter=4, fontName='Times-Bold')
body = ParagraphStyle('Body2', parent=styles['Normal'], fontSize=11, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=8, fontName='Times-Roman', firstLineIndent=0)
comment_style = ParagraphStyle('Comment', parent=body, fontSize=11, leading=16.5, leftIndent=18, rightIndent=18, fontName='Times-Italic', spaceAfter=6, spaceBefore=4, textColor=HexColor('#000000'))
response_style = ParagraphStyle('Response', parent=body, fontSize=11, leading=16.5, leftIndent=0, spaceAfter=10, spaceBefore=4, fontName='Times-Roman')
label_style = ParagraphStyle('Label', parent=body, fontSize=11, leading=16.5, fontName='Times-Bold', spaceAfter=2, spaceBefore=8)
ref_style = ParagraphStyle('Ref', parent=body, fontSize=11, leading=16.5, leftIndent=24, firstLineIndent=-24, spaceAfter=3)

elements = []

elements.append(Paragraph(
    "Response to Reviewer Comments",
    title_style
))
elements.append(Spacer(1, 6))
elements.append(Paragraph(
    "Manuscript: Attention-driven meta-learning framework detects osteoporosis from knee X-rays",
    ParagraphStyle('Sub', parent=body, fontSize=11, alignment=TA_CENTER, fontName='Times-Italic', spaceAfter=4)
))
elements.append(Paragraph(
    "Authors: Saanvi Chakraborty and Manas Chakraborty",
    ParagraphStyle('Auth', parent=body, fontSize=11, alignment=TA_CENTER, fontName='Times-Roman', spaceAfter=4)
))
elements.append(Paragraph(
    "Mason Classical Academy, Naples, FL",
    ParagraphStyle('Aff', parent=body, fontSize=11, alignment=TA_CENTER, fontName='Times-Roman', spaceAfter=12)
))
elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#000000')))
elements.append(Spacer(1, 12))

elements.append(Paragraph(
    "We thank the Editor and Reviewer 1 for the thorough and constructive evaluation of our manuscript. "
    "We have carefully addressed every comment and revised the manuscript accordingly. Below, each reviewer "
    "comment is reproduced verbatim (in italics), followed by our point-by-point response describing how "
    "and where in the revised manuscript the concern has been addressed.",
    body
))

elements.append(Spacer(1, 8))
elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#000000')))


def add_comment_response(num, comment, response):
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Comment {num}:</b>", label_style))
    elements.append(Paragraph(f"\u201c{comment}\u201d", comment_style))
    elements.append(Paragraph(f"<b>Response:</b>", label_style))
    elements.append(Paragraph(response, response_style))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#000000')))


add_comment_response(
    1,
    "Key assumptions (knee radiographs contain sufficient signal for osteoporosis, labels are accurate and "
    "clinically grounded (I did not see the dataset provenance and a reference), ordering of features for "
    "LSTM is meaningful) are not adequately justified or tested. Specify dataset source, licensing, "
    "de-identification, and IRB/ethics oversight (or exemption). Report demographic distributions and "
    "fairness analyses (performance stratified by sex/age/race) and discuss bias mitigation.",

    "We have substantially expanded the <b>Materials and Methods \u2014 Dataset</b> subsection to fully specify the dataset provenance. "
    "The revised manuscript now states: \u201cWe used the Multi-Class Knee Osteoporosis X-Ray Dataset (17), a publicly available "
    "collection of 1,947 knee X-ray images annotated by orthopedic surgery specialists into three diagnostic categories: "
    "Normal (780 images), Osteopenia (374 images), and Osteoporosis (793 images). The dataset is available on Kaggle under "
    "a CC BY 4.0 license. All images are de-identified medical radiographs without personally identifiable information. "
    "Institutional Review Board approval was not required per 45 CFR 46.104(d)(4) for publicly available, de-identified data.\u201d "
    "Reference 17 (Gobara et al., 2024, Kaggle) has been added to the reference list. "
    "The class distribution is now presented in <b>Figure 1</b> with legend.<br/><br/>"
    "Regarding the LSTM feature ordering: the revised <b>Feature extraction and dimensionality reduction</b> subsection clarifies "
    "that PCA produces orthogonal components ordered by decreasing variance explained, providing a natural and reproducible "
    "ordering for the LSTM to process sequentially.<br/><br/>"
    "Regarding demographic distributions and fairness analyses: the publicly available dataset (17) does not include patient-level "
    "demographic metadata (age, sex, race, BMI). We have added this as an explicit limitation in the <b>Discussion</b> section: "
    "\u201cThe dataset lacks patient-level demographic metadata (age, sex, race, BMI), precluding stratified fairness analyses "
    "and confounder adjustment. Future work should evaluate OsteoNexus on datasets with demographic annotations to assess "
    "potential bias.\u201d We acknowledge this is a meaningful gap and have discussed bias mitigation in the context of "
    "future multi-center validation."
)

add_comment_response(
    2,
    "No external validation, no repeated-seed experiments, and overfitting beyond 200 epochs indicate limited "
    "robustness; no sensitivity analyses to preprocessing choices, scanner variation, or noise.",

    "We have addressed each sub-concern:<br/><br/>"
    "<b>Repeated-seed experiments:</b> The revised manuscript now includes a <b>multi-seed stability analysis</b> across three "
    "random initializations (seeds 42, 123, 456). Results are reported in <b>Results</b> (\u201cMulti-seed experiments across "
    "three random initializations yielded 91.88% \u00b1 0.87% accuracy and 95.60% \u00b1 0.55% AUC, demonstrating that "
    "performance was stable and not dependent on random initialization\u201d), presented in <b>Table 3</b>, and visualized "
    "in <b>Figure 13</b>. Low standard deviations across all metrics confirmed robust stability.<br/><br/>"
    "<b>Overfitting:</b> The revised manuscript clarifies that training was halted at <b>epoch 38</b> (not 200) by early stopping "
    "(patience=30, restore_best_weights=True). This is documented in the <b>Training configuration</b> subsection and shown "
    "in <b>Figure 5</b> (training and validation curves). The original submission\u2019s mention of 200 epochs referred to the "
    "maximum epoch limit, not the actual training duration.<br/><br/>"
    "<b>External validation:</b> We acknowledge in the <b>Discussion</b> that external, multi-center validation remains a critical "
    "next step. The revised text states: \u201cThe primary limitation is the absence of external validation on independent "
    "datasets from different imaging centers, scanners, and patient demographics.\u201d We also note: \u201cFuture work should "
    "validate on multi-site datasets with different scanner manufacturers and acquisition protocols.\u201d<br/><br/>"
    "<b>Five-fold cross-validation</b> has been added as an internal validation measure (95.12% \u00b1 0.65% accuracy, "
    "98.03% \u00b1 0.48% AUC; <b>Table 2, Figure 12</b>), with per-fold standardization to prevent information leakage."
)

add_comment_response(
    3,
    "No confidence intervals, no hypothesis tests versus baselines, and no repeated runs; you will need to: "
    "bootstrap CIs for all metrics, run DeLong test for AUC, run McNemar\u2019s test for paired error rates, "
    "present calibration curves (Brier score), and nested cross-validation (X-fold) or external validation.",

    "All requested statistical analyses have been implemented and are reported in the revised manuscript:<br/><br/>"
    "<b>Bootstrap confidence intervals:</b> 95% CIs computed from 1,000 bootstrap resamples are reported in <b>Table 1</b> "
    "for all primary metrics: accuracy [89.74%, 94.87%], precision [87.12%, 93.83%], recall [95.22%, 99.16%], "
    "F1 [91.63%, 95.85%], AUC [93.90%, 98.15%]. All lower bounds exceeded the corresponding clinical benchmarks. "
    "Results are visualized in <b>Figure 14</b>.<br/><br/>"
    "<b>DeLong test for AUC:</b> DeLong tests (ref. 19: DeLong et al., <i>Biometrics</i>, 1988) comparing OsteoNexus AUC "
    "against each baseline are reported in the <b>Results</b> section and <b>Table 5</b>. OsteoNexus significantly "
    "outperformed logistic regression (z=3.86, p&lt;0.001), SVM (z=3.29, p=0.001), and random forest (z=3.07, p=0.002). "
    "Only gradient boosting showed no significant AUC difference (z=1.62, p=0.106). Methodology is detailed in "
    "<b>Materials and Methods \u2014 Evaluation and statistical analysis</b>. Visualized in <b>Figure 10A\u2013B</b>.<br/><br/>"
    "<b>McNemar\u2019s test for paired error rates:</b> McNemar\u2019s test (ref. 20: McNemar, <i>Psychometrika</i>, 1947) "
    "results are reported in <b>Results</b> and <b>Table 5</b>. All four comparisons were statistically significant: "
    "vs. logistic regression (\u03c7\u00b2=22.04, p&lt;0.001), vs. SVM (\u03c7\u00b2=6.86, p=0.009), vs. random forest "
    "(\u03c7\u00b2=8.53, p=0.004), and vs. gradient boosting (\u03c7\u00b2=5.04, p=0.025). The methodological details, "
    "including the 2\u00d72 contingency table construction and critical value (\u03c7\u00b2=3.841 at \u03b1=0.05), are "
    "described in <b>Materials and Methods</b>. Visualized in <b>Figure 10C\u2013D</b>.<br/><br/>"
    "<b>Calibration curve and Brier score:</b> The revised manuscript reports ECE=0.0372 and Brier score=0.0631 in "
    "<b>Table 1</b> and the <b>Results</b> section, with the calibration curve (reliability diagram) presented as "
    "<b>Figure 15</b>.<br/><br/>"
    "<b>Cross-validation:</b> Five-fold stratified cross-validation results are reported in <b>Table 2</b> and "
    "<b>Figure 12</b>: 95.12% \u00b1 0.65% accuracy and 98.03% \u00b1 0.48% AUC. Per-fold standardization (StandardScaler "
    "and PCA fit on training data only) was applied to prevent information leakage."
)

add_comment_response(
    4,
    "No corrections or adjustments for confounders (e.g., scanner model, acquisition protocol, age/sex/BMI) "
    "are reported; consider stratification, propensity weighting, or including nuisance covariates.",

    "We acknowledge this limitation explicitly in the revised <b>Discussion</b> section: \u201cThe dataset lacks patient-level "
    "demographic metadata (age, sex, race, BMI), precluding stratified fairness analyses and confounder adjustment.\u201d "
    "The publicly available dataset (17) does not include scanner model, acquisition protocol, or patient demographic "
    "variables, making stratification or propensity weighting infeasible with the available data.<br/><br/>"
    "We have added the following to <b>Discussion</b>: \u201cFuture work should validate on multi-site datasets with "
    "different scanner manufacturers and acquisition protocols\u201d and \u201cA prospective study comparing OsteoNexus "
    "predictions with standard DEXA referral patterns would be the ultimate validation.\u201d We recognize that confounder "
    "adjustment is essential for clinical deployment and have framed OsteoNexus as a screening tool requiring further "
    "validation rather than a diagnostic replacement."
)

add_comment_response(
    5,
    "High risk of redundant information when concatenating flattened pixels with HOG/LBP; no variance inflation "
    "factor (VIF) or redundancy analysis is reported; recommend dimensionality reduction and ablation to remove "
    "highly correlated features.",

    "We have addressed both the redundancy concern and the ablation request:<br/><br/>"
    "<b>Variance inflation factor (VIF) analysis:</b> VIF analysis was performed on all 200 PCA components and is now "
    "reported in the <b>Results</b> section: \u201cVariance inflation factor (VIF) analysis confirmed VIF=1.0 for all "
    "200 components, verifying that PCA completely eliminated the multicollinearity inherent in the raw pixel, HOG, "
    "and LBP feature concatenation.\u201d Results are visualized in <b>Figure 16</b> (VIF values for all 200 components, "
    "with concern and severe thresholds marked). VIF=1.0 for all components is expected by construction since PCA "
    "produces orthogonal eigenvectors with zero pairwise correlation.<br/><br/>"
    "<b>Dimensionality reduction:</b> The revised <b>Feature extraction and dimensionality reduction</b> subsection "
    "clarifies that PCA reduced the 76,446-dimensional raw feature concatenation to 200 orthogonal components retaining "
    "80.1% of total variance, a 99.7% reduction in dimensionality. This is visualized in <b>Figure 3</b> (feature "
    "extraction dimensions and PCA variance retention curve).<br/><br/>"
    "<b>Component ablation study:</b> A full ablation study isolating each architectural component (attention, LSTM, "
    "autoencoder) is reported in <b>Results</b> and <b>Table 6</b>, visualized in <b>Figure 11</b>. Removing the "
    "attention mechanism produced the largest accuracy decrease (90.26%, \u22122.05%), confirming its contribution. "
    "Removing LSTM or autoencoder individually yielded comparable accuracy, but the full model achieved the highest "
    "recall (97.44%), which we prioritized for clinical screening."
)

add_comment_response(
    6,
    "Pixel intensity normalization and CLAHE are used; clarify exact scaling (e.g., [0,1]) and whether HOG/LBP "
    "features were standardized per fold to avoid leakage.",

    "Both concerns are now explicitly addressed in the revised manuscript:<br/><br/>"
    "<b>Exact scaling:</b> The <b>Image preprocessing</b> subsection now specifies: \u201cgrayscale loading and resizing to "
    "224\u00d7224 pixels with normalization to [0, 1].\u201d The seven-step pipeline is fully enumerated with all parameter "
    "values (CLAHE clipLimit=2.0, tileGridSize=8\u00d78; elastic transformation \u03b1=34, \u03c3=4; Gaussian blur 5\u00d75 "
    "kernel; etc.). See <b>Figure 2</b> for visualization.<br/><br/>"
    "<b>Per-fold standardization:</b> The <b>Feature extraction and dimensionality reduction</b> subsection now explicitly "
    "states: \u201cThe concatenated 76,446-dimensional feature vector was standardized using per-fold StandardScaler "
    "(fit on training data only, then applied to validation and test sets) to prevent information leakage. PCA reduced "
    "dimensionality to 200 orthogonal components, retaining 80.1% of total variance.\u201d This is also noted in "
    "<b>Table 2</b> caption: \u201cPer-fold standardization (fit on training data only) was applied to prevent information "
    "leakage.\u201d"
)

add_comment_response(
    7,
    "Train/validation/test split is stratified, but no patient-level deduplication, no external site validation, "
    "and no detailed baselines or ablations; report hardware/software, seeds, training time, and release code; "
    "include strong baselines (e.g., ResNet/EfficientNet, logistic regression on HOG/LBP) and perform component-wise "
    "ablations.",

    "Each sub-concern has been addressed in the revised manuscript:<br/><br/>"
    "<b>Patient-level deduplication:</b> The dataset (17) contains one image per patient; there is no risk of duplicate "
    "patients across splits. This is noted in the <b>Dataset</b> subsection.<br/><br/>"
    "<b>External validation:</b> Acknowledged as a limitation in the <b>Discussion</b>: \u201cThe primary limitation is "
    "the absence of external validation on independent datasets from different imaging centers, scanners, and patient "
    "demographics.\u201d<br/><br/>"
    "<b>Hardware/software, seeds, training time:</b> A new <b>Reproducibility</b> subsection in Materials and Methods "
    "reports: \u201cAll experiments were conducted in a CPU-only environment (Intel x86_64 with AVX2/AVX512 support) "
    "using Python 3.11, TensorFlow 2.21.0, scikit-learn 1.8.0, NumPy 2.4.2, OpenCV 4.13.0, and SciPy 1.17.1. "
    "Training time was approximately 45 seconds per model run and approximately 3 minutes for feature extraction "
    "across all 1,947 images. Extracted features were cached for reproducibility. Code is available in the "
    "supplementary materials.\u201d<br/><br/>"
    "<b>Baselines:</b> Four traditional ML baselines are reported in <b>Table 4</b> and <b>Figure 9</b>: logistic "
    "regression (84.62% accuracy), SVM with RBF kernel (88.97%), random forest (88.46%), and gradient boosting (89.23%). "
    "All baselines were trained on the same 200-dimensional PCA features with identical train/test splits. Statistical "
    "comparisons via DeLong and McNemar tests are in <b>Table 5</b> and <b>Figure 10</b>.<br/><br/>"
    "<b>Component-wise ablations:</b> A full ablation study is reported in <b>Table 6</b> and <b>Figure 11</b>, "
    "systematically removing the attention mechanism, LSTM, and autoencoder individually and collectively."
)

add_comment_response(
    8,
    "I did not see that a true Meta-training was implemented. The contribution needs to: (1) implement true "
    "meta-learning (MAML/Reptile/ProtoNets) with episodic tasks; (2) add a CNN backbone (e.g., ResNet/EfficientNet) "
    "with attention (SE/CBAM) or a ViT; (3) perform rigorous ablations isolating attention, LSTM, autoencoder, "
    "handcrafted features; (4) external, multi-center validation; (5) calibration, decision curve analysis, and "
    "clinical utility evaluation; (6) code/data release and detailed training environment.",

    "We address each numbered point:<br/><br/>"
    "<b>(1) True meta-learning implementation:</b> The revised manuscript now reports both Reptile meta-training and "
    "prototypical network evaluation. The <b>Meta-learning procedures</b> subsection in Materials and Methods describes: "
    "\u201cReptile meta-training followed Nichol et al. (12) with episodic support/query splits. In each of 50 meta-epochs, "
    "5 episodic tasks were constructed (2-way, 5-shot, 10-query). The model was trained on each support set for 3 inner "
    "steps (learning rate=0.01), then the outer update interpolated original and task-adapted weights (outer learning "
    "rate=0.001).\u201d Results are reported in <b>Results</b>: Reptile achieved 47.95% accuracy (confirming few-shot "
    "episodic training underperformed supervised training), while prototypical network evaluation achieved "
    "86.33% \u00b1 6.18% accuracy across 30 five-shot episodes, demonstrating strong few-shot learning viability.<br/><br/>"
    "<b>(2) CNN backbone:</b> We acknowledge in the <b>Discussion</b> that a pre-trained CNN backbone (ResNet, "
    "EfficientNet) or Vision Transformer could replace the hand-crafted feature extraction pipeline. We discuss this "
    "as a promising direction for future work. The current architecture was designed to demonstrate that attention-driven "
    "meta-learning can achieve strong performance even with traditional features, providing a baseline for comparison "
    "with CNN-based approaches.<br/><br/>"
    "<b>(3) Rigorous ablations:</b> A full component ablation study is now presented in <b>Table 6</b> and "
    "<b>Figure 11</b>, isolating attention, LSTM, and autoencoder. Results show that removing the attention mechanism "
    "caused the largest accuracy drop (\u22122.05%), confirming its importance.<br/><br/>"
    "<b>(4) External validation:</b> Acknowledged as a limitation in the <b>Discussion</b>. We have added five-fold "
    "cross-validation (<b>Table 2, Figure 12</b>) and multi-seed stability (<b>Table 3, Figure 13</b>) as internal "
    "validation measures, while explicitly stating that external, multi-center validation remains a critical next step.<br/><br/>"
    "<b>(5) Calibration and clinical utility:</b> Calibration analysis is now reported: ECE=0.0372 and Brier "
    "score=0.0631 (<b>Table 1, Figure 15</b>). The <b>Discussion</b> addresses clinical utility: \u201cOsteoNexus "
    "achieved 97.44% recall with only 6 false negatives out of 234 At-Risk cases, supporting its potential as a "
    "high-sensitivity screening tool.\u201d The high recall and well-calibrated probabilities support clinical "
    "screening utility.<br/><br/>"
    "<b>(6) Code/data release and training environment:</b> The <b>Reproducibility</b> subsection reports the "
    "complete software stack (Python 3.11, TensorFlow 2.21.0, scikit-learn 1.8.0, NumPy 2.4.2, OpenCV 4.13.0, "
    "SciPy 1.17.1), hardware (CPU-only, Intel x86_64), training time (~45 seconds per run), and states: "
    "\u201cCode is available in the supplementary materials.\u201d The dataset is publicly available under CC BY 4.0 "
    "license (ref. 17)."
)

elements.append(Spacer(1, 12))
elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#000000')))
elements.append(Spacer(1, 12))

elements.append(Paragraph("SUMMARY OF REVISIONS", heading1))

summary_items = [
    "<b>Dataset provenance and ethics:</b> Dataset source (ref. 17), CC BY 4.0 license, de-identification status, and IRB exemption (45 CFR 46.104(d)(4)) now specified in Materials and Methods \u2014 Dataset.",
    "<b>Statistical rigor:</b> Bootstrap 95% CIs (Table 1, Figure 14), DeLong tests (Table 5, Figure 10A\u2013B), McNemar\u2019s tests (Table 5, Figure 10C\u2013D), and calibration analysis (ECE, Brier score, Figure 15) added throughout Results and Materials and Methods.",
    "<b>Robustness validation:</b> Five-fold stratified cross-validation (Table 2, Figure 12), multi-seed stability across three initializations (Table 3, Figure 13), and early stopping clarification (epoch 38, Figure 5) added.",
    "<b>VIF and redundancy analysis:</b> VIF=1.0 for all 200 PCA components reported in Results with Figure 16. PCA variance retention (80.1%) documented in Figure 3.",
    "<b>Preprocessing standardization:</b> Per-fold StandardScaler (fit on training data only) and exact [0,1] normalization explicitly documented in Materials and Methods.",
    "<b>Baselines and ablations:</b> Four ML baselines with statistical comparisons (Tables 4\u20135, Figures 9\u201310) and component ablation study (Table 6, Figure 11) added.",
    "<b>Meta-learning:</b> Reptile meta-training and prototypical network evaluation implemented and reported with episodic task construction details in Materials and Methods \u2014 Meta-learning procedures.",
    "<b>Reproducibility:</b> New subsection reporting hardware, software versions, seeds, training time, and code availability.",
    "<b>Limitations:</b> Discussion expanded to address absence of demographic metadata, external validation, confounder adjustment, and CNN backbone comparison as explicit limitations and future work.",
    "<b>Figures:</b> 16 publication-quality figures added (Figures 1\u201316) with detailed legends in the Figure Legends section per JEI guidelines. Separate Figures PDF provided.",
    "<b>References:</b> DeLong (ref. 19) and McNemar (ref. 20) citations added to support statistical methodology.",
]

for item in summary_items:
    elements.append(Paragraph(f"\u2022 {item}", ParagraphStyle('SumItem', parent=body, fontSize=11, leading=16.5, leftIndent=18, firstLineIndent=-12, spaceAfter=5)))

doc.build(elements)
print("PDF generated: OsteoNexus_Response_to_Reviewer.pdf")
