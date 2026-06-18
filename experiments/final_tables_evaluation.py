import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.feature_selection import VarianceThreshold
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
# 2. LAPLACIAN SCORE (Proper Implementation)
# =========================================================
def select_features_laplacian(X, n_features):
    """Proper Laplacian Score implementation based on He et al. (2005)."""
    try:
        n_samples, n_features_total = X.shape
        X_scaled = StandardScaler().fit_transform(X)
        
        # Construct affinity matrix using k-nearest neighbors
        k = min(5, n_samples - 1)
        dist_matrix = euclidean_distances(X_scaled, X_scaled)
        
        # Build adjacency matrix W
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            nearest_neighbors = np.argsort(dist_matrix[i])[1:k+1]
            for j in nearest_neighbors:
                W[i, j] = np.exp(-dist_matrix[i, j]**2 / (2 * np.var(X_scaled)))
                W[j, i] = W[i, j]
        
        # Degree matrix
        D = np.diag(W.sum(axis=1))
        L = D - W  # Laplacian matrix
        
        # Compute Laplacian Score for each feature
        scores = np.zeros(n_features_total)
        for f in range(n_features_total):
            f_vec = X_scaled[:, f]
            f_mean = f_vec.mean()
            f_tilde = f_vec - f_mean
            
            numerator = f_tilde.T @ L @ f_tilde
            denominator = f_tilde.T @ D @ f_tilde
            
            if denominator > 1e-10:
                scores[f] = numerator / denominator
            else:
                scores[f] = np.inf
        
        # Lower score = better feature
        selected_idx = np.argsort(scores)[:n_features]
        return selected_idx
    except Exception as e:
        print(f"Laplacian Score failed: {e}. Using variance as fallback.")
        return np.argsort(np.var(X, axis=0))[::-1][:n_features]

# =========================================================
# 3. VARIANCE THRESHOLD
# =========================================================
def select_features_variance(X, n_features):
    """Variance Threshold filter method."""
    selector = VarianceThreshold()
    selector.fit(X)
    scores = selector.variances_
    selected_idx = np.argsort(scores)[::-1][:n_features]
    return selected_idx

# =========================================================
# 4. METRIC CALCULATION
# =========================================================
def calculate_metrics(y_true, y_pred):
    """Calculate F1, MCC, Specificity, NPV."""
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # For multi-class, compute macro-averaged specificity and NPV
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    specificity_list = []
    npv_list = []
    
    for i in range(n_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        
        specificity_list.append(spec)
        npv_list.append(npv)
    
    specificity = np.mean(specificity_list)
    npv = np.mean(npv_list)
    
    return f1, mcc, specificity, npv

# =========================================================
# 5. MAIN EVALUATION PIPELINE
# =========================================================
def run_comprehensive_evaluation(file_path, proposed_selected_indices):
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df)
    n_features = X.shape[1]
    r = len(proposed_selected_indices)
    
    classifiers = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for metrics
    metrics_all = {clf: {'F1': [], 'MCC': [], 'Spec': [], 'NPV': []} for clf in classifiers}
    metrics_sel = {clf: {'F1': [], 'MCC': [], 'Spec': [], 'NPV': []} for clf in classifiers}
    metrics_lap = {clf: {'F1': [], 'MCC': [], 'Spec': [], 'NPV': []} for clf in classifiers}
    metrics_var = {clf: {'F1': [], 'MCC': [], 'Spec': [], 'NPV': []} for clf in classifiers}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # --- 1. ALL Features ---
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            f1, mcc, spec, npv = calculate_metrics(y_test, pred)
            metrics_all[clf_name]['F1'].append(f1)
            metrics_all[clf_name]['MCC'].append(mcc)
            metrics_all[clf_name]['Spec'].append(spec)
            metrics_all[clf_name]['NPV'].append(npv)
        
        # --- 2. PROPOSED Selected Features ---
        X_train_sel = X_train.iloc[:, proposed_selected_indices]
        X_test_sel = X_test.iloc[:, proposed_selected_indices]
        for clf_name, clf in classifiers.items():
            clf.fit(X_train_sel, y_train)
            pred = clf.predict(X_test_sel)
            f1, mcc, spec, npv = calculate_metrics(y_test, pred)
            metrics_sel[clf_name]['F1'].append(f1)
            metrics_sel[clf_name]['MCC'].append(mcc)
            metrics_sel[clf_name]['Spec'].append(spec)
            metrics_sel[clf_name]['NPV'].append(npv)
        
        # --- 3. LAPLACIAN Score ---
        lap_idx = select_features_laplacian(X_train.values, r)
        X_train_lap = X_train.iloc[:, lap_idx]
        X_test_lap = X_test.iloc[:, lap_idx]
        for clf_name, clf in classifiers.items():
            clf.fit(X_train_lap, y_train)
            pred = clf.predict(X_test_lap)
            f1, mcc, spec, npv = calculate_metrics(y_test, pred)
            metrics_lap[clf_name]['F1'].append(f1)
            metrics_lap[clf_name]['MCC'].append(mcc)
            metrics_lap[clf_name]['Spec'].append(spec)
            metrics_lap[clf_name]['NPV'].append(npv)
        
        # --- 4. VARIANCE Threshold ---
        var_idx = select_features_variance(X_train.values, r)
        X_train_var = X_train.iloc[:, var_idx]
        X_test_var = X_test.iloc[:, var_idx]
        for clf_name, clf in classifiers.items():
            clf.fit(X_train_var, y_train)
            pred = clf.predict(X_test_var)
            f1, mcc, spec, npv = calculate_metrics(y_test, pred)
            metrics_var[clf_name]['F1'].append(f1)
            metrics_var[clf_name]['MCC'].append(mcc)
            metrics_var[clf_name]['Spec'].append(spec)
            metrics_var[clf_name]['NPV'].append(npv)
    
    # --- Wilcoxon Test (Table 11) ---
    p_values = {}
    for clf_name in classifiers.keys():
        try:
            p_val = wilcoxon(metrics_all[clf_name]['F1'], metrics_sel[clf_name]['F1']).pvalue
        except:
            p_val = 1.0
        p_values[clf_name] = round(p_val, 4)
    
    # --- Average Metrics ---
    avg_all = {clf: {m: np.mean(vals) for m, vals in metrics.items()} for clf, metrics in metrics_all.items()}
    avg_sel = {clf: {m: np.mean(vals) for m, vals in metrics.items()} for clf, metrics in metrics_sel.items()}
    avg_lap = {clf: {m: np.mean(vals) for m, vals in metrics.items()} for clf, metrics in metrics_lap.items()}
    avg_var = {clf: {m: np.mean(vals) for m, vals in metrics.items()} for clf, metrics in metrics_var.items()}
    
    return {
        'Dataset': os.path.basename(file_path).replace('.csv', ''),
        'n_features': n_features,
        'r': r,
        'p_SVM': p_values['SVM'],
        'p_KNN': p_values['KNN'],
        'p_RF': p_values['RF'],
        'All': avg_all,
        'Proposed': avg_sel,
        'Laplacian': avg_lap,
        'Variance': avg_var
    }

# =========================================================
# 6. EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    data_dir = 'data'
    
    # All 8 datasets with their selected feature indices (0-based)
    target_datasets = {
        'heart.csv': [5, 0, 11, 12, 2],
        'ionosphere.csv': [0, 1 ,6, 23, 2, 4, 19],
        'wpbc.csv': [0, 1, 16, 33, 31, 20, 17],
        'wdbc.csv': [27, 20, 13, 19, 5, 2],
        'segment.csv': [18, 0, 7, 12, 2],
        'zoo.csv': [12, 10, 6, 14, 13],
        'dataYM2018.csv': [5, 6, 0, 8, 7],
        'TCGA_Glioma.csv': [22, 1, 3, 2, 5]
    }
    
    results = []
    
    for filename, indices in target_datasets.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            res = run_comprehensive_evaluation(file_path, indices)
            results.append(res)
        else:
            print(f"⚠️  Warning: {filename} not found in {data_dir}")
    
    # =========================================================
    # OUTPUT: TABLE 5 (RF Performance)
    # =========================================================
    print("\n" + "="*100)
    print("TABLE 5: Comprehensive evaluation using Random Forest")
    print("="*100)
    print(f"{'Dataset':<20} {'Features':<15} {'F1':<8} {'MCC':<8} {'Spec':<8} {'NPV':<8}")
    print("-"*100)
    
    for res in results:
        dataset = res['Dataset']
        n_feat = res['n_features']
        r = res['r']
        
        print(f"{dataset:<20} {'All('+str(n_feat)+')':<15} {res['All']['RF']['F1']:<8.3f} {res['All']['RF']['MCC']:<8.3f} {res['All']['RF']['Spec']:<8.3f} {res['All']['RF']['NPV']:<8.3f}")
        print(f"{'':<20} {'Sel('+str(r)+')':<15} {res['Proposed']['RF']['F1']:<8.3f} {res['Proposed']['RF']['MCC']:<8.3f} {res['Proposed']['RF']['Spec']:<8.3f} {res['Proposed']['RF']['NPV']:<8.3f}")
    
    # =========================================================
    # OUTPUT: TABLE 6 (3 Classifiers Comparison)
    # =========================================================
    print("\n" + "="*100)
    print("TABLE 6: Performance comparison between SVM, KNN, RF (F1 and MCC)")
    print("="*100)
    
    for res in results:
        dataset = res['Dataset']
        print(f"\n{dataset}:")
        print(f"  {'Classifier':<10} {'F1 (All)':<12} {'MCC (All)':<12} {'F1 (Sel)':<12} {'MCC (Sel)':<12}")
        for clf in ['SVM', 'KNN', 'RF']:
            print(f"  {clf:<10} {res['All'][clf]['F1']:<12.3f} {res['All'][clf]['MCC']:<12.3f} {res['Proposed'][clf]['F1']:<12.3f} {res['Proposed'][clf]['MCC']:<12.3f}")
    
    # =========================================================
    # OUTPUT: TABLE 11 (Wilcoxon Test)
    # =========================================================
    print("\n" + "="*100)
    print("TABLE 11: Wilcoxon signed-rank test p-values")
    print("="*100)
    print(f"{'Dataset':<20} {'SVM':<10} {'KNN':<10} {'RF':<10} {'Interpretation':<20}")
    print("-"*100)
    
    for res in results:
        dataset = res['Dataset']
        interp = "Not significant" if all(res[f'p_{clf}'] >= 0.0625 for clf in ['SVM', 'KNN', 'RF']) else "Significant"
        print(f"{dataset:<20} {res['p_SVM']:<10.4f} {res['p_KNN']:<10.4f} {res['p_RF']:<10.4f} {interp:<20}")
    
    # =========================================================
    # OUTPUT: TABLE 12 (Baseline Comparison)
    # =========================================================
    print("\n" + "="*100)
    print("TABLE 12: F1-scores comparison with unsupervised baselines")
    print("="*100)
    
    for res in results:
        dataset = res['Dataset']
        print(f"\n{dataset}:")
        print(f"  {'Classifier':<10} {'Proposed':<12} {'Laplacian':<12} {'Variance':<12}")
        for clf in ['SVM', 'KNN', 'RF']:
            print(f"  {clf:<10} {res['Proposed'][clf]['F1']:<12.3f} {res['Laplacian'][clf]['F1']:<12.3f} {res['Variance'][clf]['F1']:<12.3f}")
    
    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('comprehensive_evaluation_results.csv', index=False)
    print("\n✅ Results saved to 'comprehensive_evaluation_results.csv'")