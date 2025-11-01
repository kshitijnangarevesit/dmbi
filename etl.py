# ===============================================
# Retail Sales Data Warehouse with ETL + OLAP
# Author: Sameeksha Sankpal
# ===============================================

import pandas as pd
import sqlite3
import numpy as np

# -----------------------------
# Step 1: Extract - Create Raw Data
# -----------------------------

# Product data
products = pd.DataFrame({
    'product_id': [1, 2, 3, 4],
    'product_name': ['Laptop', 'Headphones', 'Smartphone', 'Keyboard'],
    'category': ['Electronics', 'Accessories', 'Electronics', 'Accessories'],
    'price': [80000, 2000, 30000, 1500]
})

# Branch data
branches = pd.DataFrame({
    'branch_id': [1, 2],
    'branch_name': ['Pune Central', 'Mumbai Mall'],
    'city': ['Pune', 'Mumbai'],
    'region': ['West', 'West']
})

# Customer data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'customer_name': ['Aarav', 'Isha', 'Kabir'],
    'gender': ['M', 'F', 'M'],
    'age': [28, 24, 35]
})

# Sales data (fact)
sales = pd.DataFrame({
    'sale_id': [101, 102, 103, 104, 105, 106],
    'date': pd.to_datetime([
        '2025-01-05', '2025-01-06', '2025-02-10',
        '2025-02-10', '2025-03-01', '2025-03-02'
    ]),
    'product_id': [1, 2, 3, 1, 4, 3],
    'branch_id': [1, 1, 2, 2, 1, 2],
    'customer_id': [1, 2, 3, 1, 2, 3],
    'quantity': [1, 2, 1, 1, 3, 1]
})

# -----------------------------
# Step 2: Transform
# -----------------------------

# Add calculated field total_amount
sales = sales.merge(products[['product_id', 'price']], on='product_id')
sales['total_amount'] = sales['quantity'] * sales['price']

# Extract date dimension
dim_date = pd.DataFrame({
    'date_id': range(1, len(sales['date'].unique()) + 1),
    'date': sales['date'].unique()
})
dim_date['day'] = dim_date['date'].dt.day
dim_date['month'] = dim_date['date'].dt.month
dim_date['year'] = dim_date['date'].dt.year

# Map date_id into sales (fact table)
sales = sales.merge(dim_date[['date', 'date_id']], on='date')

# Select only necessary fact columns
fact_sales = sales[['sale_id', 'date_id', 'product_id', 'branch_id', 'customer_id', 'quantity', 'total_amount']]

# -----------------------------
# Step 3: Load (into SQLite)
# -----------------------------

conn = sqlite3.connect('retail_dw.db')
cursor = conn.cursor()

# Create tables
products.to_sql('dim_product', conn, if_exists='replace', index=False)
branches.to_sql('dim_branch', conn, if_exists='replace', index=False)
customers.to_sql('dim_customer', conn, if_exists='replace', index=False)
dim_date.to_sql('dim_date', conn, if_exists='replace', index=False)
fact_sales.to_sql('fact_sales', conn, if_exists='replace', index=False)

print("✅ Data successfully loaded into Data Warehouse (SQLite).")