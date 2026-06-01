# =============================================================================
# eda_utils.py
# Reusable EDA utility functions for Home Credit Risk Project
# Phase 3 — Exploratory Data Analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# 1. COLUMN LISTS
# -----------------------------------------------------------------------------
def get_column_lists(df):
    """
    Splits DataFrame columns into three reusable lists:
    - binary    : columns starting with FLAG_
    - numeric   : numeric columns excluding binary
    - categorical: string/category columns with more than 2 unique values

    Usage:
        numeric_data, categorical_data, binary_data = get_column_lists(df_train)
    """
    binary = [col for col in df.columns if col.startswith("FLAG_")]
    numeric = [col for col in df.select_dtypes(include=["number"]).columns
               if col not in binary]
    categorical = [col for col in df.select_dtypes(include=["str", "category"]).columns
                   if df[col].nunique() > 2]
    return numeric, categorical, binary


# -----------------------------------------------------------------------------
# 2. MISSING VALUE RATE
# -----------------------------------------------------------------------------
def missing_values(df):
    """
    Computes missing value count and rate per column, sorted descending.
    Only shows columns with at least one missing value.

    Usage:
        missing_values(df_train)
        missing_values(df_bureau)
    """
    total_missing = df.isna().sum().sort_values(ascending=False)
    missing_rate = (total_missing / len(df) * 100).round(2).sort_values(ascending=False)

    print(f"🟢 Percent missing:\n{missing_rate[missing_rate > 0]}\n")
    print(f"🟢 Number missing:\n{total_missing[total_missing > 0]}")


# -----------------------------------------------------------------------------
# 3. BAD RATE SUMMARY
# -----------------------------------------------------------------------------
def bad_rate_summary(df, target_col="TARGET"):
    """
    Prints a credit risk summary of the target variable:
    - Total records
    - Population PD (bad rate %)
    - Total bads and goods
    - Good:Bad ratio (used for scale_pos_weight in Phase 7)

    Usage:
        bad_rate_summary(df_train)
        bad_rate_summary(df_train[df_train["DAYS_EMPLOYED"] == 365243])
    """
    total = len(df)
    counts = df[target_col].value_counts()
    bad_rate = df[target_col].mean() * 100
    good_count = counts[0]
    bad_count = counts[1]

    print(f"--- Credit Risk Summary ---")
    print(f"Total records:     {total:,}")
    print(f"Population PD:     {bad_rate:.2f}%")
    print(f"Total bads:        {bad_count:,}")
    print(f"Total goods:       {good_count:,}")
    print(f"Good:Bad ratio:    {good_count / bad_count:.1f}:1")


# -----------------------------------------------------------------------------
# 4. SKEWNESS REPORT
# -----------------------------------------------------------------------------
def skewness_report(df, numeric_cols):
    """
    Prints skewness classification and descriptive stats for each numeric column.
    Classification bands:
        Fairly symmetrical    : -0.5 to 0.5
        Moderately skewed     : -1 to -0.5 or 0.5 to 1
        Highly skewed         : < -1 or > 1

    Usage:
        skewness_report(df_train, numeric_data)
        skewness_report(df_bureau, numeric_bureau)
    """
    for i in numeric_cols:
        if i in df.columns:
            skew = df[i].skew()

            if -0.5 <= skew <= 0.5:
                label = "🟢 Fairly symmetrical"
            elif -1 <= skew <= -0.5:
                label = "🟡 Moderately skewed LEFT"
            elif 0.5 <= skew <= 1:
                label = "🟡 Moderately skewed RIGHT"
            elif skew < -1:
                label = "🔴 Highly skewed LEFT"
            else:
                label = "🔴 Highly skewed RIGHT"

            print(f"{i}: {label}: {skew:.3f}")
            print(f"{df[i].describe().T}\n")


# -----------------------------------------------------------------------------
# 5. OUTLIER CHECK
# -----------------------------------------------------------------------------
def outlier_check(df, numeric_cols):
    """
    Prints min, max, 99th percentile, and skewness for each numeric column.
    Use to identify extreme outliers and sentinel values.

    Usage:
        outlier_check(df_train, numeric_data)
    """
    print(f"{'Column':<40} {'Min':>12} {'Max':>15} {'99th Pct':>12} {'Skew':>8}")
    print("─" * 90)
    for col in numeric_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            pct99   = df[col].quantile(0.99)
            skew    = df[col].skew()
            print(f"{col:<40} {min_val:>12,.2f} {max_val:>15,.2f} {pct99:>12,.2f} {skew:>8.3f}")


# -----------------------------------------------------------------------------
# 6. CHECK OVERLAP
# -----------------------------------------------------------------------------
def check_overlap(df1, df2, key_col):
    """
    Checks whether two DataFrames share any values in a key column.
    Used for train/test leakage detection.

    Usage:
        check_overlap(df_train, df_test, "SK_ID_CURR")
    """
    set1 = set(df1[key_col])
    set2 = set(df2[key_col])
    overlap = set1 & set2

    if overlap == set():
        print(f"✅ No overlap found in '{key_col}' between the two DataFrames")
    else:
        print(f"⚠️  Overlap found: {len(overlap):,} shared values in '{key_col}'")
        print(overlap)


# -----------------------------------------------------------------------------
# 7. CARDINALITY REPORT
# -----------------------------------------------------------------------------
def cardinality_report(df, categorical_cols):
    """
    Prints unique value count, top category, top category count,
    and missing % for each categorical column.
    Used to identify high cardinality and rare categories.

    Usage:
        cardinality_report(df_train, categorical_data)
    """
    print(f"{'Column':<20} {'Unique':>8} {'Top Value':<30} {'Top Count':>10} {'Missing %':>10}")
    print("─" * 100)
    for col in categorical_cols:
        if col in df.columns:
            n_unique    = df[col].nunique()
            top_val     = df[col].value_counts().index[0]
            top_count   = df[col].value_counts().iloc[0]
            missing_pct = df[col].isna().mean() * 100

            print(f"{col:<20} {n_unique:>8} {str(top_val):<30} {top_count:>10} {missing_pct:>9.2f}%")




# -----------------------------------------------------------------------------
# 8. SENTINEL BAD RATE CHECK
# -----------------------------------------------------------------------------
def sentinel_bad_rate(df, column, condition, target_col="TARGET"):
    """
    Computes bad rate for sentinel vs non-sentinel population.
    Condition is passed as a lambda function for flexibility.

    Usage:
        sentinel_bad_rate(df_train, "DAYS_EMPLOYED", lambda x: x == 365243)
        sentinel_bad_rate(df_train, "OWN_CAR_AGE",   lambda x: x == 64)
        sentinel_bad_rate(df_train, "EXT_SOURCE_1",  lambda x: x.isna())
    """
    sentinel_mask        = condition(df[column])
    sentinel_bad         = df[sentinel_mask][target_col].mean() * 100
    non_sentinel_bad     = df[~sentinel_mask & df[column].notna()][target_col].mean() * 100
    population_bad       = df[target_col].mean() * 100

    print(f"--- Sentinel Bad Rate Check: {column} ---")
    print(f"Sentinel population bad rate:     {sentinel_bad:.2f}%")
    print(f"Non-sentinel population bad rate: {non_sentinel_bad:.2f}%")
    print(f"Overall population bad rate:      {population_bad:.2f}%")
    print(f"Difference:                       {sentinel_bad - non_sentinel_bad:+.2f}pp")

# -----------------------------------------------------------------------------
# 9 . SENTINEL BAD RATE CHECK A MASK ON A DIFFERENT DATAFRAME
# -----------------------------------------------------------------------------

def coverage_bad_rate(df_main, df_child, key_col, target_col="TARGET"):
    """
    Compares bad rate between applicants who have records
    in a child table vs those who don't (thin-file population).

    Usage:
        coverage_bad_rate(df_train, df_bureau, "SK_ID_CURR")
    """
    covered     = set(df_child[key_col])
    has_records = df_main[key_col].isin(covered)

    covered_bad    = df_main[has_records][target_col].mean() * 100
    thin_file_bad  = df_main[~has_records][target_col].mean() * 100
    population_bad = df_main[target_col].mean() * 100

    print(f"--- Coverage Bad Rate Check: {key_col} ---")
    print(f"Applicants with records:    {has_records.sum():,} ({has_records.mean()*100:.1f}%)")
    print(f"Thin-file (no records):     {(~has_records).sum():,} ({(~has_records).mean()*100:.1f}%)")
    print(f"Bad rate — has records:     {covered_bad:.2f}%")
    print(f"Bad rate — thin file:       {thin_file_bad:.2f}%")
    print(f"Bad rate — population:      {population_bad:.2f}%")
    print(f"Difference:                 {thin_file_bad - covered_bad:+.2f}pp")

# -----------------------------------------------------------------------------
# 9 . CARDINAL TOTAL FREQUENCY
# -----------------------------------------------------------------------------

def cardinal_freq(df, categorical_columns ):
    for col in categorical_columns:
        if col in df:
            all_freq = df[col].value_counts()
            print(f"{all_freq}\n")