
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# --------------------------------------------------------------------------------
# --- CONFIGURATION: Change these variables for a new dataset ---
# --------------------------------------------------------------------------------


# SCENARIO 1: Iris Dataset (DBSCAN often separates 2-3 clusters, but tuning is key)
DATASET_FILE = 'Iris.csv'
FEATURES_TO_DROP = ['Id', 'Species'] # Drop ID and the known label
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
VISUALIZATION_FEATURE_1 = 'PetalLengthCm'
VISUALIZATION_FEATURE_2 = 'PetalWidthCm'




# # SCENARIO 2: Mall Customer Dataset (Best parameters for density-based segmentation)
# DATASET_FILE = 'Mall_Customers.csv'
# FEATURES_TO_DROP = ['CustomerID', 'Gender', 'Age'] # Focus on Income and Score
# DBSCAN_EPS = 0.35 # Requires a smaller epsilon due to feature scaling
# DBSCAN_MIN_SAMPLES = 4
# VISUALIZATION_FEATURE_1 = 'Annual Income (k$)'
# VISUALIZATION_FEATURE_2 = 'Spending Score (1-100)'




# --------------------------------------------------------------------------------
# --- END CONFIGURATION ---
# --------------------------------------------------------------------------------


# --- 1. Preprocess data (Scaling is essential for DBSCAN) ---


# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Successfully loaded '{DATASET_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found. Make sure the file is in the same directory.")
    exit()


# Drop specified features
X = df.drop(columns=[col for col in FEATURES_TO_DROP if col in df.columns], errors='ignore')


# Handle categorical features (One-Hot Encoding)
print("\nEncoding categorical features...")
X = pd.get_dummies(X, drop_first=True)


# Impute missing values (just in case)
if X.isnull().sum().any():
    print(f"Handling {X.isnull().sum().sum()} missing values using SimpleImputer (mean strategy)...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X.values)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("Missing values imputation complete.")
else:
    print("No missing values found. Skipping imputation.")




# Feature Scaling (mandatory for distance-based algorithms like DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled using StandardScaler.")
print("-" * 50)




# --- 2. Build a Clustering model using the inbuilt library function ---


# Initialize DBSCAN using configurable parameters
print(f"Building DBSCAN model with eps={DBSCAN_EPS} and min_samples={DBSCAN_MIN_SAMPLES}...")
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)


# Fit the model (find the clusters)
dbscan.fit(X_scaled)


# Get the cluster labels. Noise points (outliers) are labeled as -1.
labels = dbscan.labels_
df['Cluster'] = labels # Add labels back to the original DataFrame for plotting
print("Clustering complete.")
print("-" * 50)




# --- 3. Determine Performance parameters ---


# Number of clusters found (excluding noise, which is label -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)


print(f"Estimated Number of Clusters: {n_clusters}")
print(f"Estimated Number of Noise Points (Outliers): {n_noise}")


# Calculate Silhouette Score
# (Only if more than 1 cluster is found and not all points are noise)
if n_clusters > 1:
    # Use the subset of data that are not noise points for the score calculation
    score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
    print(f"Silhouette Score (on non-noise points): {score:.4f}")
else:
    print("Silhouette Score cannot be calculated (less than 2 clusters found).")


# Interpretation of Results
print("\n--- Model Evaluation Summary ---")
print("DBSCAN identifies clusters as dense regions, marking sparse points as noise (-1).")
print(f"Found {n_clusters} clusters and {n_noise} noise points.")




# --- 4. Visualize the Clusters ---
try:
    plt.figure(figsize=(10, 7))
   
    # Merge the original features needed for visualization with the new cluster labels
    plot_df = df[[VISUALIZATION_FEATURE_1, VISUALIZATION_FEATURE_2, 'Cluster']].copy()


    # Convert Cluster to categorical for plotting
    plot_df['Cluster'] = plot_df['Cluster'].astype('category')
   
    sns.scatterplot(
        x=VISUALIZATION_FEATURE_1,
        y=VISUALIZATION_FEATURE_2,
        hue='Cluster',
        data=plot_df,
        palette='Spectral', # Use a palette that highlights the different clusters and noise
        s=100,
        style='Cluster'
    )
    plt.title(f'DBSCAN Clustering: {n_clusters} Clusters (Eps={DBSCAN_EPS}, MinPts={DBSCAN_MIN_SAMPLES})')
    plt.xlabel(VISUALIZATION_FEATURE_1)
    plt.ylabel(VISUALIZATION_FEATURE_2)
    plt.grid(True)
   
    plt.legend(title='Cluster/Noise (-1)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("dbscan_cluster_visualization.png")
    print("\nDBSCAN cluster visualization saved as 'dbscan_cluster_visualization.png'")
   
except Exception as e:
    print(f"\nError during visualization: {e}")






