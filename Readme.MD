# Student Dropout & Academic Success Classification

BITS Pilani WILP — M.Tech (AIML) — Machine Learning — Assignment 2

**Live app:** `https://2025ac05706-ml-assignment-2.streamlit.app/`

**Repository:** `https://github.com/pushpenderpannu/ml_assignment`

---

## a. Problem statement

Higher-education institutions lose a substantial share of enrolled students
before graduation, which harms the students financially and socially and
damages institutional outcomes. If students at risk can be identified early —
using only information available at enrolment plus first- and second-semester
performance — support can be directed to them while intervention is still
useful.

This project frames that as a supervised classification task: given a
student's demographic, socio-economic and early academic record, predict
whether they will **drop out**, remain **enrolled**, or **graduate**. Five
classification algorithms are trained on identical data and compared across
six evaluation metrics to establish which family of model best suits this
problem, and why.

---

## b. Dataset description

| Property | Value |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, dataset id 697 |
| URL | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Instances | 4,424
| Features | 36
| Target | `Target` — 3 classes: Dropout / Enrolled / Graduate |

### Class distribution

| Class | Count | Proportion |
|---|---|---|
| Graduate | `2209` | `49.9%` |
| Dropout | `1421` | `32.1%` |
| Enrolled | `794` | `17.9%` |

The target is imbalanced, so the majority-class baseline accuracy is
`0.4993`. Any model scoring below that figure performs worse than a trivial
rule that always predicts the largest class — which is why accuracy alone is
an insufficient summary and MCC is reported alongside it.

### Feature groups

- **Demographic** — marital status, gender, age at enrolment, nationality,
  displaced status, special educational needs
- **Socio-economic** — parents' qualification and occupation, scholarship
  holder, debtor status, tuition fees up to date, plus macroeconomic context
  (unemployment rate, inflation rate, GDP)
- **Academic history** — application mode and order, course, previous
  qualification and its grade, admission grade
- **Early performance** — curricular units credited / enrolled / evaluated /
  approved / without evaluation, and mean grade, for each of semesters 1 and 2

### Preprocessing decisions

**Integer-coded nominal columns.** Nine columns (`Course`, `Marital Status`,
`Application mode`, `Previous qualification`, `Nacionality`, and the four
parental qualification/occupation columns) arrive as integers, but the values
are arbitrary identifiers — course code 9254 is not "greater than" 9003.
Left as-is, Logistic Regression would fit a coefficient implying a linear
trend across unrelated codes, kNN would treat adjacent codes as similar under
Euclidean distance, and Gaussian Naive Bayes would assume an ID column is
normally distributed within each class. These columns are therefore one-hot
encoded, expanding the feature count from 36 to `247`. Tree-based models
are unaffected by this choice, so a single shared representation keeps all
five models comparable on identical input.

**Standardisation.** Feature scales differ by orders of magnitude (inflation
rate spans roughly −1 to 4; course codes run into the thousands).
`StandardScaler` is fitted on the training split only and then applied to the
test split. Fitting on the full dataset would leak test-set statistics into
training and inflate every reported metric.

**Split.** 80/20 stratified split, `random_state=1987`.
Stratification preserves class proportions in both halves, so the metrics
reflect model quality rather than an accident of shuffling.

---

## c. GitHub repository link

`https://github.com/pushpenderpannu/ml_assignment`

```
.
├── app.py                      # Streamlit application
├── requirements.txt            # pinned dependencies
├── README.md                   # this file
├── test_data.csv               # held-out 20% test split, with labels
├── data/
│   └── dropout.csv             # full dataset as downloaded from UCI
└── model/
    ├── download_data.py        # one-time UCI download
    ├── train_models.py         # trains and serialises all five models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── feature_columns.pkl
```

---

## d. Models used

All five models are trained on the same 80% training split and evaluated on
the same held-out 20% test split. Because the target has three classes,
Precision, Recall and F1 use a **macro** average (every class weighted
equally, regardless of support) and AUC uses **one-vs-rest with a macro
average**. AUC is computed from predicted probabilities, not predicted
labels.

### Hyperparameters

| Model | Key settings | Reasoning |
|---|---|---|
| Logistic Regression | `max_iter=2000`, `C=1.0`, `class_weight='balanced'` | The default 100 iterations do not converge on this feature count; balanced weights stop the minority Enrolled class being ignored |
| Decision Tree | `max_depth=8`, `min_samples_leaf=15`, `criterion='gini'` | An unbounded tree memorises the training set; the depth cap and leaf minimum trade a little training accuracy for generalisation |
| kNN | `n_neighbors=21`, `weights='distance'` | Odd k avoids ties in a three-way vote; distance weighting lets nearer neighbours count for more |
| Gaussian Naive Bayes | `var_smoothing=1e-8` | Guards against zero-variance features after one-hot encoding |
| Random Forest | `n_estimators=300`, `max_depth=14`, `max_features='sqrt'` | More trees reduce variance; `sqrt` feature sampling decorrelates them, which is what makes averaging effective |

### Comparison table

| ML Model Name|Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression         |0.7458| 0.8680| 0.7067|  0.7088|  0.7027|  0.5956|
| Decision Tree               |0.6610| 0.8356| 0.6518|  0.6462|  0.6336|  0.4867|
| K-Nearest Neighbours        |0.6249| 0.7521| 0.6155|  0.4952|  0.4903|  0.3586|
| Naive Bayes (Gaussian)      |0.2294| 0.6414| 0.4465|  0.3756|  0.1983|  0.1019|
| Random Forest (Ensemble)    |0.7559| 0.8707| 0.7098|  0.6995|  0.7030|  0.60


### Observations on model performance


| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | *Strong and stable: 0.7458 accuracy and an MCC of 0.5956, second only to Random Forest, with the smallest train/test gap of any model that predicts well (+0.0480). The narrow gap shows it is not overfitting and that the three classes are largely linearly separable in this 247-dimensional space. Its high AUC (0.8680) confirms well-calibrated probabilities. class_weight='balanced' lets it give the minority Enrolled class fair weight, which is why its recall (0.7088) is actually the highest of all five models. For the effort involved, it is the best value in the table.* |
| Decision Tree | *Middle of the pack: 0.6610 accuracy, MCC 0.4867. Even with max_depth=8 and min_samples_leaf=15 the train/test gap (+0.0796) is the second largest among the well-behaved models, the expected signature of a single tree — its hard, axis-aligned splits fit the training data more tightly than they generalise. It trails Logistic Regression across every metric, which is the standard result: one greedy tree is high-variance, and the next model fixes exactly that.* |
| kNN | *Weakest of the distance/linear models: 0.6249 accuracy, MCC 0.3586, and a stark train accuracy of 1.0000 against 0.6249 on test — a +0.3751 gap. The perfect training score is an artifact of weights='distance' (a point is its own nearest neighbour), so only the test figure is meaningful. The real cause of the mediocre test result is the curse of dimensionality: one-hot encoding pushed the feature space to 247 dimensions, where Euclidean distances between points become nearly uniform and "nearest" loses meaning. The recall (0.4952) far below its precision (0.6155) shows it misses many true positives, especially in the minority classes.* |
| Naive Bayes | *The clear failure, and the most instructive result: 0.2294 accuracy — below the 0.4993 majority-class baseline, i.e. worse than blindly guessing Graduate — with MCC of just 0.1019. Two independent causes compound. First, GaussianNB assumes every feature is continuous and normally distributed within each class, but after one-hot encoding 240+ of the 247 features are binary 0/1 indicators, which are the opposite of Gaussian. Second, its core assumption that features are conditionally independent is heavily violated here: Curricular units 2nd sem (enrolled), (evaluated) and (approved) are strongly correlated by construction, so the model multiplies overlapping evidence and produces badly miscalibrated posteriors. The very low F1 (0.1983) versus higher precision (0.4465) indicates it collapses toward one class for most of the data.* |
| Random Forest (Ensemble) | *The overall winner on five of six metrics: 0.7559 accuracy, 0.8707 AUC and the top MCC of 0.6010. Averaging 300 decorrelated trees (max_features='sqrt') cancels the single Decision Tree's variance — accuracy rises from 0.6610 to 0.7559 and MCC from 0.4867 to 0.6010, a direct demonstration of why ensembles beat individual trees. Its remaining +0.1183 train/test gap is mild residual overfitting, well controlled by the depth and leaf limits. The top features by importance — 2nd-semester units approved (0.143), 2nd-semester grade (0.103), 1st-semester units approved (0.095), and tuition fees up to date (0.046) — make strong domain sense: early academic performance and financial standing are exactly what one would expect to drive dropout, which lends the model face validity beyond its metrics.* |
| **Overall winner** | *Random Forest (Ensemble), judged on MCC (0.6010) rather than accuracy alone. MCC is the fairer criterion on this imbalanced three-class target because it is computed from every cell of the confusion matrix and cannot be inflated by favouring the majority class — Naive Bayes' collapse is invisible to a naive accuracy read but obvious in its MCC of 0.10. Logistic Regression is a very close and much simpler runner-up (MCC 0.5956) and would be the pragmatic choice if interpretability or training cost mattered. Across every model, the middle Enrolled class is handled worst: it is both the smallest (17.9%) and semantically ambiguous — an "enrolled" student has neither dropped out nor graduated yet — so it overlaps both other classes in feature space and absorbs most of the misclassifications..* |

### Feature importance (Random Forest)

`Top 10 features by Random Forest importance:`
||||
|---|---|---|
|0.1426|  Curricular units 2nd sem |(approved)|
|0.1034|  Curricular units 2nd sem |(grade)|
|0.0952|  Curricular units 1st sem |(approved)|
|0.0696|  Curricular units 1st sem |(grade)|
|0.0533|  Curricular units 2nd sem |(evaluations)|
|0.0464|  Tuition fees up to date||
|0.0411|  Curricular units 1st sem| (evaluations)|
|0.0352|  Age at enrollment||
|0.0281|  Admission grade||
|0.0263|  Previous qualification| (grade)|
---

## Streamlit application features

| Requirement | Implementation |
|---|---|
| Dataset upload (CSV) | Sidebar file uploader; validates that all feature columns and the `Target` column are present, and reorders columns to the training layout before inference |
| Model selection dropdown | Sidebar `selectbox` over all five trained models |
| Display of evaluation metrics | Six metric tiles for the selected model, plus an optional table comparing all five |
| Confusion matrix / classification report | Both — an annotated heatmap and a per-class precision/recall/F1 table |

Additional features beyond the brief: a bar chart comparing any single metric
across models, downloadable per-row predictions with class probabilities, and
a downloadable comparison table.

---

## Reproducing this project

```bash
git clone https://github.com/pushpenderpannu/ml_assignment
cd ml_assignment

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

pip install ucimlrepo              # only needed for the one-time download
python model/download_data.py      # writes data/dropout.csv

python model/train_models.py       # writes model/*.pkl and test_data.csv

streamlit run app.py               # opens at http://localhost:8501
```

Upload `test_data.csv` in the running app to reproduce the metrics in the
comparison table above.

---

## BITS Virtual Lab execution

A screenshot of `model/train_models.py` executing on BITS Virtual Lab, showing
the metrics for all five models, is included in the submitted PDF.

---

## References

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
*Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5MC89