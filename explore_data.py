# explore_data.py
# Always understand your data before training anything

import pandas as pd
import matplotlib.pyplot as plt

# ── LOAD THE DATA ──────────────────────────────────────
# pd.read_csv reads a CSV file into a DataFrame
# A DataFrame is like an Excel spreadsheet inside Python
# rows = individual resumes, columns = fields

df = pd.read_csv("data/resume_dataset.csv")

# ── BASIC EXPLORATION ──────────────────────────────────

print("="*55)
print("DATASET OVERVIEW")
print("="*55)

# How many rows and columns?
print(f"\nShape: {df.shape}")
# Output: (2484, 2) means 2484 resumes, 2 columns

# What are the column names?
print(f"\nColumns: {df.columns.tolist()}")
# Output: ['Category', 'Resume']

# Show first 3 rows
print(f"\nFirst 3 rows:")
print(df.head(3))

# Check for missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# What categories exist and how many resumes per category?
print(f"\nJob Categories and their counts:")
category_counts = df['Category'].value_counts()
print(category_counts)

# Total unique categories
print(f"\nTotal categories: {df['Category'].nunique()}")

# Average resume length
df['resume_length'] = df['Resume'].apply(len)
print(f"\nAverage resume length: {df['resume_length'].mean():.0f} characters")
print(f"Shortest resume: {df['resume_length'].min()} characters")
print(f"Longest resume: {df['resume_length'].max()} characters")

# Sample one resume to see what it looks like
print("\n" + "="*55)
print("SAMPLE RESUME (first 500 chars):")
print("="*55)
print(df['Resume'].iloc[0][:500])

print("\n" + "="*55)
print(f"CATEGORY OF THIS RESUME: {df['Category'].iloc[0]}")
print("="*55)