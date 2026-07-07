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