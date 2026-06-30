# ============================================
# Exploratory Data Analysis (EDA)
# ============================================

# Import Required Libraries
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
warnings.filterwarnings("ignore")

# Create output folder automatically
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Dataset
try:
    file_path = "dataset/sample_dataset.csv"
    df = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!")

except FileNotFoundError:
    print("❌ Dataset not found.")
    exit()

except Exception as e:
    print("❌ Error:", e)
    exit()

# Display First Five Rows
print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

# Display Last Five Rows
print("\n========== LAST FIVE ROWS ==========")
print(df.tail())

# Display Dataset Information
print("\n========== DATASET INFORMATION ==========")
print(df.info())

# Display Dataset Shape
print("\nRows and Columns")
print(df.shape)

# Display Column Names
print("\nColumn Names")
print(df.columns)

# Check Missing Values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Check Missing Values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Handle Missing Values
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].mean())
    else:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\n✅ Missing values handled successfully.")

# Remove Duplicate Records
duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

df.drop_duplicates(inplace=True)

print("✅ Duplicate records removed.")
# ============================================
# SUMMARY STATISTICS
# ============================================

print("\n========== SUMMARY STATISTICS ==========")

summary = df.describe(include="all")

print(summary)

# Save summary statistics
summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "summary_statistics.csv"
    )
)

print("\n✅ Summary statistics saved successfully.")

# ============================================
# NUMERIC COLUMNS
# ============================================

numeric_columns = df.select_dtypes(include=np.number).columns

print("\nNumeric Columns:")
print(numeric_columns)

# ============================================
# MEAN
# ============================================

print("\n========== MEAN ==========")
print(df[numeric_columns].mean())

# ============================================
# MEDIAN
# ============================================

print("\n========== MEDIAN ==========")
print(df[numeric_columns].median())

# ============================================
# MODE
# ============================================

print("\n========== MODE ==========")
print(df[numeric_columns].mode())

# ============================================
# VARIANCE
# ============================================

print("\n========== VARIANCE ==========")
print(df[numeric_columns].var())

# ============================================
# STANDARD DEVIATION
# ============================================

print("\n========== STANDARD DEVIATION ==========")
print(df[numeric_columns].std())
# ============================================
# HISTOGRAMS
# ============================================

for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Histogram - {column}")
    plt.savefig(f"output/{column}_histogram.png")
    plt.close()

print("✅ Histograms saved.")
# ============================================
# CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(8, 6))

correlation = df[numeric_columns].corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("output/heatmap.png")

plt.close()

print("✅ Heatmap saved.")
# ============================================
# BOXPLOTS
# ============================================

for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot - {column}")
    plt.savefig(f"output/{column}_boxplot.png")
    plt.close()

print("✅ Boxplots saved.")
# ============================================
# PAIRPLOT
# ============================================

pair = sns.pairplot(df[numeric_columns])
pair.savefig("output/pairplot.png")
plt.close()

print("✅ Pairplot saved.")
# ============================================
# COUNTPLOTS
# ============================================

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[column])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"output/{column}_countplot.png")
    plt.close()

print("✅ Countplots saved.")
# ============================================
# OUTLIER DETECTION
# ============================================

print("\n========== OUTLIER DETECTION ==========")

for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"{column}: {len(outliers)} outlier(s)")
    # ============================================
# DATASET INSIGHTS
# ============================================

print("\n========== DATASET INSIGHTS ==========")

print(f"Total Rows : {df.shape[0]}")
print(f"Total Columns : {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

if "Salary" in df.columns:
    print(f"\nAverage Salary : {df['Salary'].mean():.2f}")
    print(f"Maximum Salary : {df['Salary'].max()}")
    print(f"Minimum Salary : {df['Salary'].min()}")

if "Age" in df.columns:
    print(f"\nAverage Age : {df['Age'].mean():.2f}")

if "Department" in df.columns:
    print("\nEmployees in Each Department:")
    print(df["Department"].value_counts())

print("\nEDA Analysis Completed Successfully!")
print("All graphs have been saved in the 'output' folder.")