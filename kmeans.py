




import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Added for better visualization
from sklearn.impute import SimpleImputer


# --------------------------------------------------------------------------------
# --- CONFIGURATION: Change these variables for a new dataset ---
# --------------------------------------------------------------------------------


# SCENARIO 1: Iris Dataset (Predicting 3 clusters based on 4 features)
# DATASET_FILE = 'Iris.csv'
# FEATURES_TO_DROP = ['Id', 'Species'] # Drop ID and the known label
# N_CLUSTERS = 3
# VISUALIZATION_FEATURE_1 = 'SepalLengthCm' # Feature for X-axis in plot
# VISUALIZATION_FEATURE_2 = 'PetalLengthCm' # Feature for Y-axis in plot




# SCENARIO 2: Mall Customer Dataset (Predicting clusters based on Income/Score)
DATASET_FILE = 'Mall_Customers.csv'
FEATURES_TO_DROP = ['CustomerID', 'Gender', 'Age'] # Focus on Income and Score
N_CLUSTERS = 5 # Typically 5 is optimal for this data based on Elbow Method
VISUALIZATION_FEATURE_1 = 'Annual Income (k$)'
VISUALIZATION_FEATURE_2 = 'Spending Score (1-100)'




# --------------------------------------------------------------------------------
# --- END CONFIGURATION ---
# --------------------------------------------------------------------------------




# --- 1. Preprocess data (Scaling is essential for K-Means) ---


# Load the dataset
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Successfully loaded '{DATASET_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found. Make sure the file is in the same directory.")
    exit()


# Drop specified features and categorical columns not used in K-Means
X = df.drop(columns=[col for col in FEATURES_TO_DROP if col in df.columns], errors='ignore')


# Handle categorical features by encoding them (e.g., 'Gender' in Mall dataset if not dropped)
X = pd.get_dummies(X, drop_first=True)


# Impute missing values (just in case)
if X.isnull().sum().any():
    print(f"\nHandling {X.isnull().sum().sum()} missing values using SimpleImputer (mean strategy)...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X.values)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("Missing values imputation complete.")
else:
    print("No missing values found. Skipping imputation.")




# Feature Scaling (mandatory for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nData scaled using StandardScaler.")
print("-" * 50)




# --- 2. Build a Clustering model using the inbuilt library function ---


# Initialize K-Means.
print(f"Building K-Means model with n_clusters={N_CLUSTERS}...")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42, n_init='auto')


# Fit the model (find the clusters)
kmeans.fit(X_scaled)


# Get the cluster labels
labels = kmeans.labels_


# Add labels back to the original DataFrame for plotting
df['Cluster'] = labels
print("Clustering complete.")
print("-" * 50)




# --- 3. Determine Performance parameters ---


# Calculate Inertia (Within-Cluster Sum of Squares)
inertia = kmeans.inertia_
print(f"K-Means Inertia (WCSS): {inertia:.4f}")


# Calculate Silhouette Score (Requires > 1 cluster)
if N_CLUSTERS > 1:
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {score:.4f}")
else:
    print("Silhouette Score not calculated (n_clusters must be > 1).")


# Interpretation of Results
print("\n--- Model Evaluation Summary ---")
print("Inertia measures how tightly grouped the clusters are (lower is better).")
if N_CLUSTERS > 1:
    print(f"A Silhouette Score of {score:.4f} is considered a measure of cluster quality (closer to 1 is better).")




# --- 4. Visualize the Clusters ---
try:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=VISUALIZATION_FEATURE_1,
        y=VISUALIZATION_FEATURE_2,
        hue='Cluster',
        data=df,
        palette='viridis',
        s=100,
        style='Cluster'
    )
    plt.title(f'K-Means Clustering: {N_CLUSTERS} Clusters')
    plt.xlabel(VISUALIZATION_FEATURE_1)
    plt.ylabel(VISUALIZATION_FEATURE_2)
    plt.grid(True)
   
    # Plotting cluster centers (must transform back to original scale)
    # Note: This is an approximation since the centers are calculated in scaled space.
    # We find the indices for the visualization features in the original X columns
   
    # We need to get the centers in the scaled space
    centers_scaled = kmeans.cluster_centers_


    # To plot the centers on the unscaled scatter plot, we inverse transform the centers
    # NOTE: This inverse transformation is only strictly accurate if we used the same scaler
    # applied only to the visualization features, but we will approximate it here for display.
    # A more precise way would be to create a second scaler just for the visualization features.


    # Find the column indices for the visualization features in the scaled data X_scaled
    try:
        idx1 = X.columns.get_loc(VISUALIZATION_FEATURE_1)
        idx2 = X.columns.get_loc(VISUALIZATION_FEATURE_2)
       
        # Plot the centers from the scaled space (approximate)
        plt.scatter(
            X_scaled[:, idx1],
            X_scaled[:, idx2],
            marker='X',
            s=250,
            color='red',
            label='Centroids (Scaled approx)',
            edgecolors='black'
        )
    except KeyError:
        # Fallback if selected features are not in the final encoded X
        print("\nWarning: Could not plot centroids precisely as visualization features were dropped or encoded.")
        pass


    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("cluster_visualization.png")
    print("\nCluster visualization saved as 'cluster_visualization.png'")
   
except Exception as e:
    print(f"\nError during visualization: {e}")




