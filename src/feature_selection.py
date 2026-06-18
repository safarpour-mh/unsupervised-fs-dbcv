import os
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, pairwise_distances
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

# =========================================================
# 1. DATA PREPROCESSING
# =========================================================
def preprocess_data(df):
    """Preprocesses the dataset according to the manuscript's methodology."""
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    
    # FIX: Removed extra spaces in string comparisons
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))

    # FIX: Removed extra spaces in dtype strings
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    return X.reset_index(drop=True), y.reset_index(drop=True)

# =========================================================
# 2. OBJECTIVE 1: DISCRETIZED SHANNON ENTROPY (K=5)
# =========================================================
def calculate_entropy(feature, K=5):
    """Calculates Shannon entropy with UNIFORM discretization into K bins."""
    feature = pd.Series(feature).astype(float)
    bins = pd.cut(feature, bins=K, labels=False, include_lowest=True)
    if bins.isna().all():
        return 0.0

    bins = bins.dropna().astype(int)
    counts = np.bincount(bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

# =========================================================
# 3. OBJECTIVE 2: DBCV
# =========================================================
def calculate_dbcv_1d(feature):
    """Calculates the 1D Density-Based Clustering Validation (DBCV) index."""
    x = np.array(feature, dtype=float).reshape(-1, 1)
    n = len(x)
    if n < 4:
        return 0.0
    x_scaled = (x - np.mean(x)) / (np.std(x) + 1e-8)

    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
    except Exception:
        return 0.0

    try:
        dist_matrix = pairwise_distances(x_scaled, x_scaled)
    except Exception:
        return 0.0

    core_dists = np.zeros(n)
    for c in [0, 1]:
        idx = np.where(labels == c)[0]
        n_c = len(idx)
        if n_c <= 1:
            continue

        k_core = min(5, n_c - 1)
        if k_core < 1:
            continue

        sub_dist = dist_matrix[np.ix_(idx, idx)]
        sorted_dists = np.sort(sub_dist, axis=1)
        core_dists[idx] = sorted_dists[:, k_core]

    mrd_matrix = np.maximum(np.maximum(core_dists[:, np.newaxis], core_dists[np.newaxis, :]), dist_matrix)

    try:
        mst = minimum_spanning_tree(mrd_matrix).toarray()
        mst = np.maximum(mst, mst.T)
    except Exception:
        return 0.0

    edges = np.argwhere(mst > 0)
    sparseness_0, sparseness_1 = 0.0, 0.0
    separation = np.inf

    for i, j in edges:
        weight = mst[i, j]
        label_i, label_j = labels[i], labels[j]

        if label_i == label_j:
            if label_i == 0:
                sparseness_0 = max(sparseness_0, weight)
            else:
                sparseness_1 = max(sparseness_1, weight)
        else:
            separation = min(separation, weight)

    if separation == np.inf:
        separation = 0.0

    max_sparseness = max(sparseness_0, sparseness_1)
    denominator = max(separation, max_sparseness) + 1e-8
    return float((separation - max_sparseness) / denominator)

# =========================================================
# 4. FUZZY CROWDING-DISTANCE RANKING
# =========================================================
def fuzzy_crowding_distance(o1, o2):
    """Calculates the final fuzzy ranking score based on Equations 4-6."""
    n = len(o1)
    cd_fuzzy = np.zeros(n)
    epsilon = 1e-8
    for obj in [o1, o2]:
        sorted_idx = np.argsort(obj)[::-1]
        mu = np.zeros(n)
        max_val, min_val = np.max(obj), np.min(obj)

        sigma_p = (max_val - min_val) / 10.0
        if sigma_p < epsilon:
            sigma_p = epsilon

        for k in range(n):
            orig_idx = sorted_idx[k]
            if k == 0 or k == n - 1:
                mu[orig_idx] = 1.0
            else:
                prev_idx = sorted_idx[k - 1]
                next_idx = sorted_idx[k + 1]
                delta = (obj[prev_idx] - obj[next_idx]) / (max_val - min_val + epsilon)
                mu[orig_idx] = 1.0 - np.exp(-(delta ** 2) / (2 * sigma_p ** 2))

        cd_fuzzy += mu

    return cd_fuzzy / 2.0

# =========================================================
# 5. DOWNSTREAM EVALUATION
# =========================================================
def evaluate_features(X_data, y_data):
    """Evaluates features using 3 classifiers with 5-fold CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = {
        # FIX: Removed extra spaces in dictionary keys
        "SVM": GridSearchCV(
            SVC(kernel="rbf"),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
            cv=3,
            scoring="f1_weighted",
            n_jobs=1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, clf in classifiers.items():
        f1_scores, mcc_scores, spec_scores, npv_scores = [], [], [], []

        for train_idx, test_idx in skf.split(X_data, y_data):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]

            clf.fit(X_train, y_train)
            # FIX: Removed space in variable name
            y_pred = clf.predict(X_test)

            f1_scores.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            mcc_scores.append(matthews_corrcoef(y_test, y_pred))

            # FIX: Use y_test instead of y_data for metric calculation
            unique_classes = np.unique(y_test)
            spec_c, npv_c = [], []
            for c in unique_classes:
                tn_c = np.sum((y_test != c) & (y_pred != c))
                fp_c = np.sum((y_test != c) & (y_pred == c))
                fn_c = np.sum((y_test == c) & (y_pred != c))

                spec_c.append(tn_c / (tn_c + fp_c + 1e-8))
                npv_c.append(tn_c / (tn_c + fn_c + 1e-8))

            spec_scores.append(np.mean(spec_c))
            npv_scores.append(np.mean(npv_c))

        results[name] = {
            "F1": float(np.mean(f1_scores)),
            "MCC": float(np.mean(mcc_scores)),
            "Specificity": float(np.mean(spec_scores)),
            "NPV": float(np.mean(npv_scores)),
        }

    return results

# =========================================================
# 6. MAIN PIPELINE
# =========================================================
def run_pipeline_on_dataset(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, header=None)
    X, y = preprocess_data(df)
    n_samples, n_features = X.shape
    print(f"Shape: {n_samples} samples, {n_features} features")
    
    scores_o1 = np.zeros(n_features)
    scores_o2 = np.zeros(n_features)

    for i in range(n_features):
        scores_o1[i] = calculate_entropy(X.iloc[:, i], K=5)
        # FIX: Removed space in function name
        scores_o2[i] = calculate_dbcv_1d(X.iloc[:, i])

    cd_scores = fuzzy_crowding_distance(scores_o1, scores_o2)

    r = max(5, int(np.ceil(0.2 * n_features)))
    r = min(r, n_features)
    # FIX: Removed space in variable name
    selected_indices = np.argsort(cd_scores)[::-1][:r].tolist()

    eval_all = evaluate_features(X.values, y)
    X_sel = X.iloc[:, selected_indices].values
    eval_sel = evaluate_features(X_sel, y)

    selected_indices_1based = [i + 1 for i in selected_indices]

    return {
        "Dataset": os.path.basename(file_path).replace(".csv", ""),
        "Original_Features": n_features,
        "Selected_Features_Count": r,
        "Selected_Indices_1Based": str(selected_indices_1based),
        "All_SVM_F1": eval_all["SVM"]["F1"],
        "All_SVM_MCC": eval_all["SVM"]["MCC"],
        "All_KNN_F1": eval_all["KNN"]["F1"],
        "All_KNN_MCC": eval_all["KNN"]["MCC"],
        "All_RF_F1": eval_all["RF"]["F1"],
        "All_RF_MCC": eval_all["RF"]["MCC"],
        "All_RF_Spec": eval_all["RF"]["Specificity"],
        "All_RF_NPV": eval_all["RF"]["NPV"],
        "Sel_SVM_F1": eval_sel["SVM"]["F1"],
        "Sel_SVM_MCC": eval_sel["SVM"]["MCC"],
        "Sel_KNN_F1": eval_sel["KNN"]["F1"],
        "Sel_KNN_MCC": eval_sel["KNN"]["MCC"],
        "Sel_RF_F1": eval_sel["RF"]["F1"],
        "Sel_RF_MCC": eval_sel["RF"]["MCC"],
        "Sel_RF_Spec": eval_sel["RF"]["Specificity"],
        "Sel_RF_NPV": eval_sel["RF"]["NPV"],
    }

# =========================================================
# 7. EXECUTION BLOCK
# =========================================================
# FIX: Corrected the main block syntax
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # FIX: Removed extra space in directory name
    data_dir = "data"
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the 'data' directory.")
    else:
        print(f"Found {len(csv_files)} CSV files. Starting processing...")
        all_results = []

        for file in csv_files:
            try:
                res = run_pipeline_on_dataset(file)
                all_results.append(res)
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")

        results_df = pd.DataFrame(all_results)
        # FIX: Removed extra space in filename
        output_file = "feature_selection_results_FINAL.csv"
        results_df.to_csv(output_file, index=False)

        print("\nProcessing completed successfully!")
        print(f"Results saved to '{output_file}'.")

        if results_df.empty:
            print("No successful results were produced.")
        else:
            summary_cols = [
                "Dataset",
                "Original_Features",
                "Selected_Features_Count",
                "All_RF_F1",
                "Sel_RF_F1",
                "All_RF_MCC",
                "Sel_RF_MCC",
            ]
            available_cols = [c for c in summary_cols if c in results_df.columns]
            print("\n--- Summary Comparison (Random Forest) ---")
            print(results_df[available_cols].to_string(index=False))