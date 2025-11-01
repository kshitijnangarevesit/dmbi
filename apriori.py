import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------------
# --- CONFIGURATION: Change these variables for a new transactional dataset ---
# --------------------------------------------------------------------------------


# SCENARIO 1: Groceries Dataset
DATASET_FILE = 'Groceries_dataset.csv'
# The transaction ID is a combination of these two columns for the Groceries data
TRANSACTION_ID_COLUMNS = ['Member_number', 'Date']
ITEM_COLUMN = 'ItemDescription'


# # SCENARIO 2: Hypothetical Online Retail Dataset (Uncomment to use)
# DATASET_FILE = 'online_retail.csv'
# TRANSACTION_ID_COLUMNS = ['InvoiceNo'] # Often a single column for the invoice/basket ID
# ITEM_COLUMN = 'Description'


# ALGORITHM PARAMETERS
MIN_SUPPORT = 0.01      # Minimum frequency (e.g., 1%) for an itemset to be considered "frequent"
# REDUCED MIN_CONFIDENCE: Changed from 0.5 to 0.25 to find more rules
MIN_CONFIDENCE = 0.25    # Minimum confidence (e.g., 25%) for a rule to be considered "strong"


# --------------------------------------------------------------------------------
# --- END CONFIGURATION ---
# --------------------------------------------------------------------------------




# --- 1. Preprocess Data: Prepare Transactional Data for Apriori ---


# Load the dataset
print(f"Loading {DATASET_FILE}...")
try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"Error: {DATASET_FILE} not found. Make sure the file is in the same directory.")
    exit()


# **FIX FOR KNOWN GROCERIES DATASET HEADER ISSUE**
# This fix attempts to handle missing or incorrectly cased headers for the Groceries dataset.
if DATASET_FILE == 'Groceries_dataset.csv':
    # Define the desired column names for the rest of the script
    desired_cols = ['Member_number', 'Date', 'ItemDescription']
   
    # Check if the columns match the common but incorrectly cased version: ['Member_number', 'Date', 'itemDescription']
    # This specifically targets the case-sensitivity issue in the 'itemDescription' column
    if 'itemDescription' in df.columns and 'ItemDescription' not in df.columns:
        df = df.rename(columns={'itemDescription': 'ItemDescription'})
        print("Note: Column 'itemDescription' renamed to 'ItemDescription' to resolve case-sensitivity.")
   
    # Secondary check: If headers were missing entirely (0, 1, 2 as column names)
    elif len(df.columns) == len(desired_cols) and df.columns[0] != desired_cols[0]:
        df.columns = desired_cols
        print("Note: Column headers for Groceries_dataset.csv were automatically set due to missing headers.")




# Ensure the required columns exist AFTER any fixes
if not all(col in df.columns for col in TRANSACTION_ID_COLUMNS + [ITEM_COLUMN]):
    print(f"Error: One or more required columns ({TRANSACTION_ID_COLUMNS + [ITEM_COLUMN]}) not found in {DATASET_FILE}. Please update the configuration.")
    exit()


# **CRITICAL FIX for ValueError: Index contains duplicate entries**
# Drop duplicate item entries within the same transaction ID before one-hot encoding.
# This ensures each unique item appears only once per basket, preventing the unstack error.
df = df.drop_duplicates(subset=TRANSACTION_ID_COLUMNS + [ITEM_COLUMN], keep='first')
print("Duplicate item entries within the same transaction have been removed.")


# Display initial data structure
print(f"Total Transactions (rows in long format): {len(df)}")
print(f"Total Unique Items: {df[ITEM_COLUMN].nunique()}")
print("-" * 70)




# Create a one-hot encoded matrix: each row is a basket, each column is an item
# 1. Group by the TRANSACTION_ID_COLUMNS and pivot using ITEM_COLUMN
print("Creating one-hot encoded basket matrix...")


# Use .apply(list) and then explode to handle duplicate item names within a row if necessary
# For Groceries_dataset, the original approach works best:
basket_sets = (df.groupby(TRANSACTION_ID_COLUMNS)[ITEM_COLUMN]
              .apply(lambda x: pd.Series(1, index=x))
              .unstack(fill_value=0))


# Clean up column and index names
basket_sets.columns.name = None
basket_sets.index.names = TRANSACTION_ID_COLUMNS


# Convert counts to boolean (1 if purchased, 0 if not) for mlxtend
# This handles cases where an item might appear multiple times in the source data for one transaction
# Suppress the FutureWarning from applymap by using .map() on individual columns or restructuring the code
basket_sets = basket_sets.applymap(lambda x: 1 if x > 0 else 0)


print(f"One-Hot Encoded Matrix Shape: {basket_sets.shape}")
print("-" * 70)




# --- 2. Run Apriori Algorithm to Find Frequent Itemsets ---


print(f"Running Apriori with minimum support = {MIN_SUPPORT}...")
frequent_itemsets = apriori(basket_sets, min_support=MIN_SUPPORT, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets.")
print("-" * 70)




# --- 3. Determine Association Rules (Performance parameters) ---


print(f"Generating rules with minimum confidence = {MIN_CONFIDENCE}...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)


# Sort the rules by Lift (a high lift indicates a strong, non-random association)
rules = rules.sort_values(by=['lift', 'confidence'], ascending=False).reset_index(drop=True)


print(f"Generated {len(rules)} strong association rules.")
print("\nTop 10 Association Rules:")
# Display only key columns for clarity
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
print("-" * 70)




# --- Visualization: Scatter Plot of Rules ---


if not rules.empty:
    plt.figure(figsize=(10, 6))
   
    # Scatter plot: Confidence vs Support, color-coded by Lift
    scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'],
                          s=rules['lift']*50, alpha=0.7, cmap='viridis')
   
    plt.xlabel("Support (Frequency)")
    plt.ylabel("Confidence (Prediction Power)")
    plt.title("Association Rules: Confidence vs Support (Color = Lift)")
   
    # Add color bar for Lift
    cbar = plt.colorbar(scatter)
    cbar.set_label('Lift Score (Stronger Association)')
   
    # Add horizontal and vertical lines for minimum thresholds
    plt.axhline(MIN_CONFIDENCE, color='r', linestyle='--', linewidth=1, label=f'Min Confidence ({MIN_CONFIDENCE})')
    plt.axvline(MIN_SUPPORT, color='b', linestyle='--', linewidth=1, label=f'Min Support ({MIN_SUPPORT})')
   
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    # Save the plot to a file
    plt.savefig("apriori_rules_visualization.png")


    print("\nAssociation rules plot saved as 'apriori_rules_visualization.png'")
   


# Final Interpretation
print("\n--- Apriori Analysis Summary ---")
print("High Lift scores (typically > 1) suggest a useful rule, meaning the items are bought together")
print("more frequently than expected by chance.")
if not rules.empty:
    print(f"The highest Lift achieved is: {rules['lift'].max():.4f}")
    print("Rules with high lift and high confidence are the most actionable for marketing or store layout.")



