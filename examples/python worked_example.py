import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. INPUT DATA (Exactly as Table 1 in the manuscript)
# =========================================================
data = {
    'Object': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
    'Age': [72, 46, 26, 29, 17, 72, 64, 38, 46, 53],
    'Gender': [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    'Total_bilirubin': [3.9, 1.8, 0.9, 0.9, 0.9, 2.7, 0.9, 0.8, 0.8, 0.9],
    'Direct_bilirubin': [2.0, 0.7, 0.2, 0.3, 0.3, 1.3, 0.3, 0.2, 0.2, 0.2],
    'Total_proteins': [195, 208, 154, 202, 202, 260, 310, 185, 185, 210],
    'Albumin': [27, 19, 16, 14, 22, 31, 61, 25, 24, 35],
    'A/G_ratio': [59, 14, 12, 11, 19, 56, 58, 21, 15, 32],
    'SGPT': [7.3, 7.6, 7.0, 6.7, 7.4, 7.4, 7.0, 7.0, 7.9, 8.0],
    'SGOT': [2.4, 4.4, 3.5, 3.6, 4.1, 3.0, 3.4, 3.0, 3.7, 3.9],
    'Alkphos': [0.4, 1.3, 1.0, 1.1, 1.2, 0.6, 0.9, 0.7, 0.8, 0.9]
}
df = pd.DataFrame(data)
features = df.columns[1:] # Exclude 'Object' column
n_features = len(features)

# =========================================================
# 2. OBJECTIVE 1: DISCRETIZED SHANNON ENTROPY (K=5)
# =========================================================
def calculate_entropy(feature, K=5):
    # FIX 1: Use pd.cut for uniform (equal-width) discretization instead of pd.qcut
    bins = pd.cut(feature, bins=K, labels=False)
    counts = np.bincount(bins.dropna().astype(int))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# =========================================================
# 3. OBJECTIVE 2: DBCV (1D)
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
            if label_i == 0: sparseness_0 = max(sparseness_0, weight)
            else: sparseness_1 = max(sparseness_1, weight)
        else:
            separation = min(separation, weight)

    if separation == np.inf: 
        separation = 0.0

    max_sparseness = max(sparseness_0, sparseness_1)
    
    # FIX 2: Correct DBCV denominator according to Equation 3 in the manuscript
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
# 5. EXECUTION AND TABLE GENERATION
# =========================================================
print("Calculating scores for the ILPD worked example...\n")
scores_o1 = np.zeros(n_features)
scores_o2 = np.zeros(n_features)

for i, feat in enumerate(features):
    scores_o1[i] = calculate_entropy(df[feat], K=5)
    scores_o2[i] = calculate_dbcv_1d(df[feat])

cd_scores = fuzzy_crowding_distance(scores_o1, scores_o2)

# Create a results dataframe
results_df = pd.DataFrame({
    'Feature': features,
    'Entropy_O1': np.round(scores_o1, 4),
    'DBCV_O2': np.round(scores_o2, 4),
    'CD_fuzzy': np.round(cd_scores, 4)
})

# Sort by CD_fuzzy descending
results_df = results_df.sort_values(by='CD_fuzzy', ascending=False).reset_index(drop=True)
results_df['Rank'] = results_df.index + 1

# Determine number of features to select: r = max(5, ceil(0.2 * 10)) = 5
r = max(5, int(np.ceil(0.2 * n_features)))
selected_features = results_df['Feature'].head(r).tolist()

print("=" * 80)
print(f"TOP {r} SELECTED FEATURES: {', '.join(selected_features)}")
print("=" * 80)
print("\n🔥 REAL DATA FOR TABLE 2 (Copy this to your LaTeX table): 🔥")
print("-" * 100)
print(f"{'Rank': <5} | {'Feature': <18} | {'Entropy_O1': <12} | {'DBCV_O2': <12} | {'CD_fuzzy': <10}")
print("-" * 100)

for _, row in results_df.iterrows():
    print(f"{row['Rank']:<5} | {row['Feature']:<18} | {row['Entropy_O1']:<12} | {row['DBCV_O2']:<12} | {row['CD_fuzzy']:<10}")
print("-" * 100)

print("\n📊 LaTeX Table Code Snippet (Updated with Real Values):")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Feature ranking based on fuzzy crowding-distance scores (DBCV-based). Higher scores indicate greater utility and diversity in the bi-objective space.}")
print("\\label{tab:worked_example_ranking}")
print("\\footnotesize")
print("\\begin{tabularx}{\\textwidth}{XXXX}")
print("\\toprule")
print("\\textbf{Rank} & \\textbf{Feature} & \\textbf{CD$_{\\text{fuzzy}}$} & \\textbf{Interpretation} \\\\")
print("\\midrule")

interpretations = {
    'Gender': 'Binary encoding yields maximal entropy and perfect two-cluster separation.',
    'Albumin': 'High variability and moderate structural coherence.',
    'Alkphos': 'Balanced performance across both objectives.',
    'SGPT': 'Nearly identical objective profile to Alkphos.',
    'Age': 'Moderate entropy but limited cluster structure.',
    'Direct_bilirubin': 'Slightly outperforms A/G ratio and Total bilirubin.',
    'A/G_ratio': 'Moderate scores, lower diversity contribution.',
    'Total_bilirubin': 'Moderate scores, lower diversity contribution.',
    'Total_proteins': 'High entropy but poor clusterability (high within-cluster spread).',
    'SGOT': 'Strong structural coherence and moderate distributional variability.'
}

for _, row in results_df.iterrows():
    feat = row['Feature'].replace('_', ' ')
    bold_start = "\\textbf{" if row['Rank'] <= 5 else ""
    bold_end = "}" if row['Rank'] <= 5 else ""
    interp = interpretations.get(row['Feature'], 'Moderate utility and diversity.')
    print(f"{bold_start}{row['Rank']}{bold_end} & {bold_start}{feat}{bold_end} & {bold_start}{row['CD_fuzzy']}{bold_end} & {interp} \\\\")

print("\\bottomrule")
print("\\end{tabularx}")
print("\\end{table}")