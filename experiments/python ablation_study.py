import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. DATA PREPROCESSING
# =========================================================
def preprocess_data(df):
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns

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
    # FIX: Use pd.cut for uniform discretization
    bins = pd.cut(feature, bins=K, labels=False)
    bins = bins.dropna().astype(int)
    counts = np.bincount(bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# =========================================================
# 3. OBJECTIVE 2: DBCV
# =========================================================
def calculate_dbcv_1d(feature):
    x = np.array(feature).reshape(-1, 1)
    n = len(x)
    if n < 4:
        return 0.0
    x_scaled = (x - np.mean(x)) / (np.std(x) + 1e-8)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    dist_matrix = pairwise_distances(x_scaled, x_scaled)

    core_dists = np.zeros(n)
    for c in [0, 1]:
        idx = np.where(labels == c)[0]
        n_c = len(idx)
        if n_c == 0: 
            continue
        k_core = min(5, n_c - 1)
        if k_core > 0:
            sub_dist = dist_matrix[np.ix_(idx, idx)]
            sorted_dists = np.sort(sub_dist, axis=1)
            core_dists[idx] = sorted_dists[:, k_core]

    mrd_matrix = np.maximum(np.maximum(core_dists[:, np.newaxis], core_dists[np.newaxis, :]), dist_matrix)
    mst = minimum_spanning_tree(mrd_matrix).toarray()
    mst = np.maximum(mst, mst.T)

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
    # FIX: Correct DBCV denominator
    denominator = max(separation, max_sparseness)
    return (separation - max_sparseness) / (denominator + 1e-8)

# =========================================================
# 4. FUZZY CROWDING-DISTANCE RANKING
# =========================================================
def fuzzy_crowding_distance(o1, o2):
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
                mu[orig_idx] = 1.0 - np.exp(-(delta**2) / (2 * sigma_p**2))
        
        cd_fuzzy += mu
        
    return cd_fuzzy / 2.0

# =========================================================
# 5. ABLATION STUDY EVALUATION HELPER
# =========================================================
def evaluate_ablation(X, y, scores):
    n_features = X.shape[1]
    r = max(5, int(np.ceil(0.2 * n_features)))
    selected_indices = np.argsort(scores)[::-1][:r]
    X_sel = X.iloc[:, selected_indices].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    mcc_scores = []
    for train_idx, test_idx in skf.split(X_sel, y):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))
        
    return np.mean(mcc_scores)

# =========================================================
# 6. MAIN ABLATION PIPELINE
# =========================================================
def run_ablation_on_dataset(file_path):
    print(f"\nProcessing Ablation for: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df)
    n_features = X.shape[1]

    scores_o1 = np.zeros(n_features)
    scores_o2 = np.zeros(n_features)

    for i in range(n_features):
        scores_o1[i] = calculate_entropy(X.iloc[:, i], K=5)
        scores_o2[i] = calculate_dbcv_1d(X.iloc[:, i])
        
    # Variant A: Entropy Only
    mcc_a = evaluate_ablation(X, y, scores_o1)

    # Variant B: DBCV Only
    mcc_b = evaluate_ablation(X, y, scores_o2)

    # Variant C: Bi-Objective without Fuzzy Crowding (Simple Average)
    scores_c = (scores_o1 + scores_o2) / 2.0
    mcc_c = evaluate_ablation(X, y, scores_c)

    # Variant D: Full Proposed Method (with Fuzzy Crowding Distance)
    scores_d = fuzzy_crowding_distance(scores_o1, scores_o2)
    mcc_d = evaluate_ablation(X, y, scores_d)

    return {
        'Dataset': os.path.basename(file_path).replace('.csv', ''),
        'Variant_A_Entropy': round(mcc_a, 3),
        'Variant_B_DBCV': round(mcc_b, 3),
        'Variant_C_No_Fuzzy': round(mcc_c, 3),
        'Variant_D_Full_Proposed': round(mcc_d, 3)
    }

# =========================================================
# 7. EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    data_dir = 'data'
    
    # FIX: Corrected filenames
    target_datasets = [
        'dataYM2018.csv', 
        'heart.csv', 
        'ionosphere.csv',        
        'segment.csv', 
        'TCGA_Glioma.csv',       
        'wdbc.csv', 
        'wpbc.csv', 
        'zoo.csv'
    ]
    
    csv_files = []
    missing_files = []
    
    for f in target_datasets:
        full_path = os.path.join(data_dir, f)
        if os.path.exists(full_path):
            csv_files.append(full_path)
        else:
            missing_files.append(f)
    
    if missing_files:
        print(f"\n⚠️  WARNING: {len(missing_files)} file(s) NOT FOUND:")
        for mf in missing_files:
            print(f"   ❌ {mf}")
    
    if not csv_files:
        print("❌ No CSV files found. Aborting.")
    else:
        print(f"\n✅ Found {len(csv_files)}/8 datasets. Starting Ablation Study...")
        ablation_results = []
        
        for file in csv_files:
            try:
                res = run_ablation_on_dataset(file)
                ablation_results.append(res)
            except Exception as e:
                print(f"   ❌ Error processing {os.path.basename(file)}: {e}")
                
        results_df = pd.DataFrame(ablation_results)
        output_file = 'ablation_study_all_8_datasets.csv'
        results_df.to_csv(output_file, index=False)
        
        print("\n" + "="*100)
        print("TABLE 7: Ablation Study Results (MCC Scores)")
        print("="*100)
        print(f"{'Dataset':<20} {'A (Entropy)':<15} {'B (DBCV)':<15} {'C (No Fuzzy)':<15} {'D (Full)':<15} {'Best':<15}")
        print("-"*100)
        
        for _, row in results_df.iterrows():
            dataset = row['Dataset']
            a, b, c, d = row['Variant_A_Entropy'], row['Variant_B_DBCV'], row['Variant_C_No_Fuzzy'], row['Variant_D_Full_Proposed']
            best = max(a, b, c, d)
            if best == a: best_var = "A"
            elif best == b: best_var = "B"
            elif best == c: best_var = "C"
            else: best_var = "D ✓"
            
            print(f"{dataset:<20} {a:<15.3f} {b:<15.3f} {c:<15.3f} {d:<15.3f} {best_var:<15}")
        
        print("-"*100)
        print(f"\n✅ Results saved to '{output_file}'")