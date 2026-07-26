from scipy.stats import mstats

def classify_source(col_name):
    """Classify a column name into its source table group."""
    bureau_cols = [
        "COUNT_BUREAU_TOTAL", "COUNT_BUREAU_ACTIVE", "COUNT_BUREAU_CLOSED",
        "COUNT_BUREAU_BAD_DEBT", "COUNT_BUREAU_SOLD", "FLAG_HAS_BAD_DEBT_BUREAU",
        "FLAG_EVER_OVERDUE", "FLAG_HAS_MICROLOAN", "FLAG_HAS_MORTGAGE",
        "FLAG_EVER_PROLONGED", "MAX_CREDIT_DAY_OVERDUE", "MAX_AMT_CREDIT_MAX_OVERDUE",
        "SUM_AMT_CREDIT_SUM_OVERDUE", "SUM_AMT_CREDIT_SUM", "MEAN_AMT_CREDIT_SUM",
        "SUM_AMT_CREDIT_SUM_DEBT", "MEAN_AMT_CREDIT_SUM_DEBT",
        "DAYS_SINCE_LAST_CREDIT", "DAYS_SINCE_FIRST_CREDIT", "COUNT_CREDIT_PROLONG",
        "RATIO_ACTIVE_TO_TOTAL", "DEBT_TO_CREDIT_RATIO", "FLAG_NO_BUREAU_HISTORY"
    ]
    prev_cols = [
        "COUNT_PREV_TOTAL", "COUNT_PREV_APPROVED", "COUNT_PREV_REFUSED",
        "COUNT_PREV_CANCELED", "FLAG_EVER_REFUSED", "FLAG_REJECTED_SCOFR",
        "FLAG_REJECTED_HC", "FLAG_REJECTED_LIMIT", "FLAG_REVOLVING_LOAN",
        "FLAG_WALK_IN", "FLAG_CHANNEL_AP_PLUS", "FLAG_EVER_REFRESHED",
        "FLAG_PURPOSE_REFUSED_NAME", "FLAG_PURPOSE_DEBT_RELATED",
        "FLAG_HAD_DRAWING", "FLAG_EVER_INSURED", "FLAG_EVER_DOWNGRADED",
        "MEAN_AMT_CREDIT", "MAX_AMT_CREDIT", "MEAN_AMT_ANNUITY",
        "MEAN_CNT_PAYMENT", "MEAN_CREDIT_TO_APP_RATIO", "DAYS_SINCE_LAST_DECISION",
        "REFUSAL_RATE", "APPROVAL_RATE", "FLAG_NO_PREV_APPLICATION"
    ]

    if col_name.startswith("CC_") or col_name == "FLAG_HAS_CREDIT_CARD":
        return "credit_card"
    elif col_name.startswith("POS_"):
        return "pos_cash"
    elif col_name.startswith("INST_"):
        return "installments"
    elif col_name.startswith("BB_"):
        return "bureau_balance"
    elif col_name in bureau_cols:
        return "bureau"
    elif col_name in prev_cols:
        return "previous_application"
    else:
        return "application_train"


def group_missing_by_source(missing_series):
    """
    Groups a missing-rate Series by source table and summarises.

    Usage:
        missing_pct = (df_train.isna().sum() / len(df_train) * 100)
        group_missing_by_source(missing_pct)
    """
    df_missing = missing_series.reset_index()
    df_missing.columns = ["feature", "missing_pct"]
    df_missing["source"] = df_missing["feature"].apply(classify_source)

    summary = df_missing.groupby("source").agg(
        n_columns=("feature", "count"),
        n_with_missing=("missing_pct", lambda x: (x > 0).sum()),
        mean_missing_pct=("missing_pct", "mean"),
        max_missing_pct=("missing_pct", "max"),
    ).sort_values("mean_missing_pct", ascending=False)

    print(summary)
    return df_missing

# WINSORISE
WINSORISE_COLS = [
    # --- application_train raw columns ---
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",

    # --- derived ratios (inherit AMT tails) ---
    "CREDIT_TO_INCOME",
    "DEBT_TO_INCOME",
    "CREDIT_TO_GOODS",

    # --- bureau aggregations ---
    "MAX_AMT_CREDIT_MAX_OVERDUE",
    "SUM_AMT_CREDIT_SUM",
    "MEAN_AMT_CREDIT_SUM",
    "SUM_AMT_CREDIT_SUM_DEBT",
    "MEAN_AMT_CREDIT_SUM_DEBT",
    "SUM_AMT_CREDIT_SUM_OVERDUE",
    "MAX_CREDIT_DAY_OVERDUE",

    # --- previous_application aggregations ---
    "MEAN_AMT_CREDIT",
    "MAX_AMT_CREDIT",
    "MEAN_AMT_ANNUITY",

    # --- POS_CASH aggregations ---
    "POS_MAX_DPD",
    "POS_MAX_DPD_DEF",

    # --- installments aggregations ---
    "INST_MAX_DAYS_LATE",
    "INST_SUM_AMT_UNDERPAID",
    "INST_MEAN_AMT_INSTALMENT",

    # --- credit card aggregations ---
    "CC_MAX_DPD",
    "CC_MAX_UTILISATION",
    "CC_MEAN_UTILISATION",
    "CC_MAX_AMT_BALANCE",
    "CC_MEAN_AMT_BALANCE",
    "CC_SUM_AMT_DRAWINGS_ATM",
    "CC_MEAN_AMT_DRAWINGS",
]

def winsorise_columns(df, columns=WINSORISE_COLS, caps=None):
    """
    Cap right-tail outliers at the 99th percentile.

    LEAKAGE GUARD: caps must be computed on TRAIN only.
    - Train: call with caps=None -> computes caps (ignoring NaN) and returns them
    - Test:  call with caps=<the dict returned from the train call>

    Must run BEFORE zero-imputation — quantiles are computed on raw
    values with NaN excluded, so imputed zeros don't distort the caps.

    Usage:
        df_train, train_caps = winsorise_columns(df_train)
        df_test, _           = winsorise_columns(df_test, caps=train_caps)
    """
    df = df.copy()

    if caps is None:
        caps = {
            col: df[col].quantile(0.99)
            for col in columns
            if col in df.columns
        }

    for col, cap in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    return df, caps

# =============================================================================
# MISSING VALUE IMPUTATION — Phase 4
# Classification derived from EDA findings + Cell 1/2/3 analysis
# =============================================================================

# Bucket 1 — impute 0: child-table flags/counts/sums/max-DPD.
# "No records" genuinely means zero events.
IMPUTE_ZERO_PREFIXES = ("BB_", "POS_", "INST_", "CC_")

IMPUTE_ZERO_EXPLICIT = [
    # bureau aggregation (no shared prefix)
    "COUNT_BUREAU_TOTAL", "COUNT_BUREAU_ACTIVE", "COUNT_BUREAU_CLOSED",
    "COUNT_BUREAU_BAD_DEBT", "COUNT_BUREAU_SOLD", "FLAG_HAS_BAD_DEBT_BUREAU",
    "FLAG_EVER_OVERDUE", "FLAG_HAS_MICROLOAN", "FLAG_HAS_MORTGAGE",
    "FLAG_EVER_PROLONGED", "MAX_CREDIT_DAY_OVERDUE",
    "SUM_AMT_CREDIT_SUM_OVERDUE", "SUM_AMT_CREDIT_SUM",
    "SUM_AMT_CREDIT_SUM_DEBT", "COUNT_CREDIT_PROLONG",
    "MAX_AMT_CREDIT_MAX_OVERDUE",   # never overdue -> max overdue amount = 0
    # previous_application aggregation (no shared prefix)
    "COUNT_PREV_TOTAL", "COUNT_PREV_APPROVED", "COUNT_PREV_REFUSED",
    "COUNT_PREV_CANCELED", "FLAG_EVER_REFUSED", "FLAG_REJECTED_SCOFR",
    "FLAG_REJECTED_HC", "FLAG_REJECTED_LIMIT", "FLAG_REVOLVING_LOAN",
    "FLAG_WALK_IN", "FLAG_CHANNEL_AP_PLUS", "FLAG_EVER_REFRESHED",
    "FLAG_PURPOSE_REFUSED_NAME", "FLAG_PURPOSE_DEBT_RELATED",
    "FLAG_HAD_DRAWING", "FLAG_EVER_INSURED", "FLAG_EVER_DOWNGRADED",
]

# Exception: ratio/mean features stay NaN even with a bucket-1 prefix —
# imputing 0 would fabricate "worst behaviour" for no-history applicants.
KEEP_NAN_RATIOS = [
    "CC_MAX_UTILISATION", "CC_MEAN_UTILISATION", "CC_MEAN_PAYMENT_VS_MIN",
    "CC_ATM_TO_TOTAL_DRAWING_RATIO", "CC_MEAN_AMT_BALANCE",
    "CC_MAX_AMT_BALANCE", "CC_MEAN_CREDIT_LIMIT", "CC_MEAN_AMT_DRAWINGS",
    "POS_MEAN_COMPLETION_RATIO",
    "INST_MEAN_PAYMENT_RATIO", "INST_MEAN_DAYS_LATE", "INST_MAX_DAYS_LATE",
    "INST_MEAN_AMT_INSTALMENT",
    "BB_RATIO_GOOD_MONTHS",
    "RATIO_ACTIVE_TO_TOTAL", "DEBT_TO_CREDIT_RATIO",
    "MEAN_AMT_CREDIT_SUM", "MEAN_AMT_CREDIT_SUM_DEBT",
    "DAYS_SINCE_LAST_CREDIT", "DAYS_SINCE_FIRST_CREDIT",
    "MEAN_AMT_CREDIT", "MAX_AMT_CREDIT", "MEAN_AMT_ANNUITY",
    "MEAN_CNT_PAYMENT", "MEAN_CREDIT_TO_APP_RATIO",
    "REFUSAL_RATE", "APPROVAL_RATE", "DAYS_SINCE_LAST_DECISION",
]

# Bucket 2 — median imputation: tiny (<1%) random missingness in
# application_train numerics
IMPUTE_MEDIAN = [
    "EXT_SOURCE_2", "AMT_GOODS_PRICE", "CREDIT_TO_GOODS",
    "AMT_ANNUITY", "ANNUITY_TO_CREDIT", "DEBT_TO_INCOME",
    "CNT_FAM_MEMBERS", "POSITIVE_DAYS_LAST_PHONE_CHANGE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "EXT_1_2_MEAN", "EXT_1_2_MIN", "EXT_1_2_MAX",
    "EXT_2_3_MEAN", "EXT_2_3_MIN", "EXT_2_3_MAX",
    "EXT_1_2_3_MEAN", "EXT_1_2_3_MIN", "EXT_1_2_3_MAX",
]

# Bucket 3 — everything else stays NaN (informative/structural):
# EXT_SOURCE_1/3, EXT_*_STD, OWN_CAR_AGE, OCCUPATION_TYPE,
# building stats, YEARS_EMPLOYED, EMPLOYMENT_TO_AGE,
# AMT_REQ_CREDIT_BUREAU_*, NAME_TYPE_SUITE, CODE_GENDER,
# NAME_FAMILY_STATUS — no code needed, absence of action.


def impute_missing(df, medians=None):
    """
    Apply the three-bucket imputation strategy.

    LEAKAGE GUARD: medians must be computed on TRAIN only.
    - Train: call with medians=None -> computes and returns them
    - Test:  call with medians=<the dict returned from the train call>

    Usage:
        df_train, train_medians = impute_missing(df_train)
        df_test, _              = impute_missing(df_test, medians=train_medians)
    """
    df = df.copy()

    # --- Bucket 1: impute 0 ---
    zero_cols = [
        c for c in df.columns
        if (c.startswith(IMPUTE_ZERO_PREFIXES) or c in IMPUTE_ZERO_EXPLICIT)
        and c not in KEEP_NAN_RATIOS
    ]
    df[zero_cols] = df[zero_cols].fillna(0)

    # --- Bucket 2: median (train-fitted) ---
    if medians is None:
        medians = {c: df[c].median() for c in IMPUTE_MEDIAN if c in df.columns}

    for c, m in medians.items():
        if c in df.columns:
            df[c] = df[c].fillna(m)

    return df, medians

def get_binning_lists(df, target_col="TARGET", id_col="SK_ID_CURR"):
    """
    Split feature columns into numerical and categorical lists for optbinning.

    Excludes the target and ID columns — binning either would be
    meaningless (ID) or catastrophic leakage (target).

    Binary 0/1 columns are returned as numerical — optbinning handles
    them correctly and treats each value as its own bin.

    Usage:
        num_cols, cat_cols = get_binning_lists(df_train)
    """
    exclude = {target_col, id_col}
    features = [c for c in df.columns if c not in exclude]

    categorical = [
        c for c in features
        if df[c].dtype == "object"
        or df[c].dtype == "bool"
        or str(df[c].dtype) in ("string", "category")
    ]
    numerical = [c for c in features if c not in categorical]

    # Integrity check — nothing should fall through
    unclassified = set(features) - set(numerical) - set(categorical)
    if unclassified:
        print(f"⚠️  WARNING — unclassified columns: {unclassified}")

    print(f"Numerical:   {len(numerical)}")
    print(f"Categorical: {len(categorical)}")
    print(f"Excluded:    {sorted(exclude)}")
    print(f"Total:       {len(numerical) + len(categorical)} of {len(features)} features")

    return numerical, categorical


