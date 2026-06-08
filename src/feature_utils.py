import numpy as np

def create_sentinel_flags(df):
    """
    Create binary flags BEFORE replacing sentinel values.
    Must be called before replace_sentinels().

    Usage:
        df_train = create_sentinel_flags(df_train)
    """
    df = df.copy()
    df["FLAG_OWN_CAR_AGE_SENTINEL"] = (df["OWN_CAR_AGE"] == 64).astype(int)
    df["FLAG_EMPLOYED_SENTINEL"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    return df

def replace_sentinels(df):
    """
    Replace sentinel values with NaN across application_train.
    Must be called AFTER create_sentinel_flags().

    Usage:
        df_train = replace_sentinels(df_train)
    """
    df = df.copy()
    df.replace({365243: np.nan, "XNA": np.nan, "Unknown": np.nan}, inplace=True)
    df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].replace(64, np.nan)
    return df

def encode_binary_flags(df):
    """
    Convert Y/N string columns to 1/0 integers.

    Usage:
        df_train = encode_binary_flags(df_train)
    """
    df = df.copy()
    yn_cols = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
    for col in yn_cols:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0})
    return df

def convert_days_columns(df):
    """
    Convert negative DAYS columns to positive years or days.
    Creates new columns with YEARS_ and POSITIVE_DAYS_ prefixes.
    Drops original DAYS_ columns.

    Usage:
        df_train = convert_days_columns(df_train)
    """
    df = df.copy()
    year_cols = ["DAYS_BIRTH", "DAYS_EMPLOYED"]
    day_cols = ["DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]

    for col in year_cols:
        new_name = col.replace("DAYS_", "YEARS_")
        df[new_name] = df[col] * -1 / 365
        df.drop(col, axis=1, inplace=True)

    for col in day_cols:
        new_name = col.replace("DAYS_", "POSITIVE_DAYS_")
        df[new_name] = df[col] * -1
        df.drop(col, axis=1, inplace=True)

    return df

def create_domain_features(df):
    """
    Create domain ratio features and EXT_SOURCE interactions.
    Must be called AFTER convert_days_columns() — needs YEARS_ columns.

    Usage:
        df_train = create_domain_features(df_train)
    """
    df = df.copy()

    df["DEBT_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["CREDIT_TO_GOODS"]= df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["EMPLOYMENT_TO_AGE"] = df["YEARS_EMPLOYED"] / df["YEARS_BIRTH"].replace(0, np.nan)

    df["EXT_1_2_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2"]].mean(axis=1)
    df["EXT_1_2_MIN"]= df[["EXT_SOURCE_1", "EXT_SOURCE_2"]].min(axis=1)
    df["EXT_1_2_MAX"]= df[["EXT_SOURCE_1", "EXT_SOURCE_2"]].max(axis=1)
    df["EXT_1_2_STD"]= df[["EXT_SOURCE_1", "EXT_SOURCE_2"]].std(axis=1)

    df["EXT_2_3_MEAN"]=df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_2_3_MIN"]=df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["EXT_2_3_MAX"]=df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
    df["EXT_2_3_STD"]=df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)

    df["EXT_1_3_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_1_3_MIN"]= df[["EXT_SOURCE_1", "EXT_SOURCE_3"]].min(axis=1)
    df["EXT_1_3_MAX"]= df[["EXT_SOURCE_1", "EXT_SOURCE_3"]].max(axis=1)
    df["EXT_1_3_STD"]= df[["EXT_SOURCE_1", "EXT_SOURCE_3"]].std(axis=1)


    ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_1_2_3_MEAN"] = df[ext_sources].mean(axis=1)
    df["EXT_1_2_3_MIN"]= df[ext_sources].min(axis=1)
    df["EXT_1_2_3_MAX"]= df[ext_sources].max(axis=1)
    df["EXT_1_2_3_STD"]= df[ext_sources].std(axis=1)
    return df

def drop_low_variance(df):
    """
    Drop near-constant columns with no predictive signal.
    Identified in Phase 2 Step 5 variance check.

    Usage:
        df_train = drop_low_variance(df_train)
    """
    df = df.copy()
    drop_cols = [
        "FLAG_MOBIL",
        "FLAG_CONT_MOBILE",
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_20"
    ]
    df= df.drop(drop_cols, axis=1, errors='ignore')
    return df

# ------------------------------------------------------------------
# CHILD DF AGGREGATE
#-------------------------------------------------------------------
def aggregate_bureau(df_bureau):
    """
    Aggregate bureau table to SK_ID_CURR grain.
    Returns one row per applicant with bureau-derived features.

    Usage:
        bureau_agg = aggregate_bureau(df_bureau)
        df_train_fe = df_train_fe.merge(bureau_agg, on="SK_ID_CURR", how="left")
    """
    df = df_bureau.copy()

    # --- Data quality fixes ---
    df["AMT_CREDIT_SUM_DEBT"] = df["AMT_CREDIT_SUM_DEBT"].clip(lower=0)
    df["AMT_CREDIT_SUM_LIMIT"] = df["AMT_CREDIT_SUM_LIMIT"].clip(lower=0)
    df["DAYS_CREDIT_UPDATE"] = df["DAYS_CREDIT_UPDATE"].clip(upper=0)

    # --- Binary flags per credit record ---
    df["IS_ACTIVE"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)
    df["IS_CLOSED"] = (df["CREDIT_ACTIVE"] == "Closed").astype(int)
    df["IS_BAD_DEBT"] = (df["CREDIT_ACTIVE"] == "Bad debt").astype(int)
    df["IS_SOLD"] = (df["CREDIT_ACTIVE"] == "Sold").astype(int)
    df["IS_OVERDUE"] = (df["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    df["IS_MICROLOAN"] = (df["CREDIT_TYPE"] == "Microloan").astype(int)
    df["IS_MORTGAGE"] = (df["CREDIT_TYPE"] == "Mortgage").astype(int)
    df["IS_PROLONGED"] = (df["CNT_CREDIT_PROLONG"] > 0).astype(int)

    # --- Aggregation to SK_ID_CURR grain ---
    agg = df.groupby("SK_ID_CURR").agg(

        # Volume features
        COUNT_BUREAU_TOTAL        = ("SK_ID_BUREAU",            "count"),
        COUNT_BUREAU_ACTIVE       = ("IS_ACTIVE",               "sum"),
        COUNT_BUREAU_CLOSED       = ("IS_CLOSED",               "sum"),
        COUNT_BUREAU_BAD_DEBT     = ("IS_BAD_DEBT",             "sum"),
        COUNT_BUREAU_SOLD         = ("IS_SOLD",                 "sum"),

        # Risk flags
        FLAG_HAS_BAD_DEBT_BUREAU  = ("IS_BAD_DEBT",             "max"),
        FLAG_EVER_OVERDUE         = ("IS_OVERDUE",              "max"),
        FLAG_HAS_MICROLOAN        = ("IS_MICROLOAN",            "max"),
        FLAG_HAS_MORTGAGE         = ("IS_MORTGAGE",             "max"),
        FLAG_EVER_PROLONGED       = ("IS_PROLONGED",            "max"),

        # Overdue features
        MAX_CREDIT_DAY_OVERDUE    = ("CREDIT_DAY_OVERDUE",      "max"),
        MAX_AMT_CREDIT_MAX_OVERDUE= ("AMT_CREDIT_MAX_OVERDUE",  "max"),
        SUM_AMT_CREDIT_SUM_OVERDUE= ("AMT_CREDIT_SUM_OVERDUE",  "sum"),

        # Exposure features
        SUM_AMT_CREDIT_SUM        = ("AMT_CREDIT_SUM",          "sum"),
        MEAN_AMT_CREDIT_SUM       = ("AMT_CREDIT_SUM",          "mean"),
        SUM_AMT_CREDIT_SUM_DEBT   = ("AMT_CREDIT_SUM_DEBT",     "sum"),
        MEAN_AMT_CREDIT_SUM_DEBT  = ("AMT_CREDIT_SUM_DEBT",     "mean"),

        # Recency features
        DAYS_SINCE_LAST_CREDIT    = ("DAYS_CREDIT",             "max"),
        DAYS_SINCE_FIRST_CREDIT   = ("DAYS_CREDIT",             "min"),
        COUNT_CREDIT_PROLONG      = ("CNT_CREDIT_PROLONG",      "sum"),

    ).reset_index()

    # --- Post-aggregation engineered features ---
    agg["RATIO_ACTIVE_TO_TOTAL"] = (
        agg["COUNT_BUREAU_ACTIVE"] / agg["COUNT_BUREAU_TOTAL"]
    )
    agg["DEBT_TO_CREDIT_RATIO"] = (
        agg["SUM_AMT_CREDIT_SUM_DEBT"] /
        agg["SUM_AMT_CREDIT_SUM"].replace(0, np.nan)
    )

    # Convert recency to positive days
    agg["DAYS_SINCE_LAST_CREDIT"]  = agg["DAYS_SINCE_LAST_CREDIT"]  * -1
    agg["DAYS_SINCE_FIRST_CREDIT"] = agg["DAYS_SINCE_FIRST_CREDIT"] * -1

    return agg


def aggregate_bureau_balance(df_bureau_balance, df_bureau):
    """
    Aggregate bureau_balance to SK_ID_CURR grain via SK_ID_BUREAU.
    Two-level aggregation: bureau_balance → SK_ID_BUREAU → SK_ID_CURR.

    Usage:
        bb_agg = aggregate_bureau_balance(df_bureau_balance, df_bureau)
        df_train_fe = df_train_fe.merge(bb_agg, on="SK_ID_CURR", how="left")
    """
    df = df_bureau_balance.copy()

    # --- Encode STATUS numerically ---
    # C=closed, X=unknown treated as 0 (no DPD)
    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    df["STATUS_NUM"] = df["STATUS"].map(status_map).fillna(0)

    # --- Binary DPD flags ---
    df["IS_DPD"] = (df["STATUS_NUM"] >= 1).astype(int)
    df["IS_DPD_60"] = (df["STATUS_NUM"] >= 2).astype(int)
    df["IS_DPD_90"] = (df["STATUS_NUM"] >= 3).astype(int)
    df["IS_DPD_120"] = (df["STATUS_NUM"] >= 4).astype(int)

    # --- Recency filters ---
    df["IS_LAST_3M"] = (df["MONTHS_BALANCE"] >= -3).astype(int)
    df["IS_LAST_6M"] = (df["MONTHS_BALANCE"] >= -6).astype(int)
    df["IS_LAST_12M"] = (df["MONTHS_BALANCE"] >= -12).astype(int)
    df["IS_LAST_24M"] = (df["MONTHS_BALANCE"] >= -24).astype(int)

    # --- Aggregate to SK_ID_BUREAU grain ---
    bureau_agg = df.groupby("SK_ID_BUREAU").agg(

        # Lifetime DPD flags
        FLAG_EVER_DPD_BB=("IS_DPD", "max"),
        FLAG_EVER_DPD_60_BB=("IS_DPD_60", "max"),
        FLAG_EVER_DPD_90_BB=("IS_DPD_90", "max"),
        FLAG_EVER_DPD_120_BB=("IS_DPD_120", "max"),

        # DPD counts
        COUNT_MONTHS_DPD=("IS_DPD", "sum"),
        COUNT_MONTHS_DPD_60=("IS_DPD_60", "sum"),
        COUNT_MONTHS_DPD_90=("IS_DPD_90", "sum"),
        MAX_DPD_STATUS=("STATUS_NUM", "max"),

        # Recency DPD — last 12 months
        MAX_DPD_LAST_12M=("STATUS_NUM",
                          lambda x: x[df.loc[x.index, "IS_LAST_12M"] == 1].max()
                          if (df.loc[x.index, "IS_LAST_12M"] == 1).any()
                          else 0),

        # Total months observed
        COUNT_MONTHS_TOTAL=("MONTHS_BALANCE", "count"),

    ).reset_index()

    # --- Merge SK_ID_CURR onto bureau aggregation ---
    bureau_key = df_bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    bureau_agg = bureau_agg.merge(bureau_key, on="SK_ID_BUREAU", how="left")

    # --- Aggregate to SK_ID_CURR grain ---
    agg = bureau_agg.groupby("SK_ID_CURR").agg(

        BB_FLAG_EVER_DPD=("FLAG_EVER_DPD_BB", "max"),
        BB_FLAG_EVER_DPD_60=("FLAG_EVER_DPD_60_BB", "max"),
        BB_FLAG_EVER_DPD_90=("FLAG_EVER_DPD_90_BB", "max"),
        BB_FLAG_EVER_DPD_120=("FLAG_EVER_DPD_120_BB", "max"),
        BB_MAX_DPD_STATUS=("MAX_DPD_STATUS", "max"),
        BB_MAX_DPD_LAST_12M=("MAX_DPD_LAST_12M", "max"),
        BB_COUNT_MONTHS_DPD=("COUNT_MONTHS_DPD", "sum"),
        BB_COUNT_MONTHS_DPD_60=("COUNT_MONTHS_DPD_60", "sum"),
        BB_COUNT_MONTHS_DPD_90=("COUNT_MONTHS_DPD_90", "sum"),
        BB_COUNT_MONTHS_TOTAL=("COUNT_MONTHS_TOTAL", "sum"),

    ).reset_index()

    # --- Good month ratio ---
    agg["BB_RATIO_GOOD_MONTHS"] = (
            (agg["BB_COUNT_MONTHS_TOTAL"] - agg["BB_COUNT_MONTHS_DPD"]) /
            agg["BB_COUNT_MONTHS_TOTAL"].replace(0, np.nan)
    )

    return agg


def aggregate_previous_application(df_prev):
    """
    Aggregate previous_application to SK_ID_CURR grain.
    Filters to FLAG_LAST_APPL_PER_CONTRACT = Y before aggregating.

    Usage:
        prev_agg = aggregate_previous_application(df_prev)
        df_train_fe = df_train_fe.merge(prev_agg, on="SK_ID_CURR", how="left")
    """
    df = df_prev.copy()

    # --- Deduplicate — keep last application per contract ---
    df = df[df["FLAG_LAST_APPL_PER_CONTRACT"] == "Y"]

    # --- Data quality fixes ---
    df["AMT_APPLICATION"] = df["AMT_APPLICATION"].replace(0, np.nan)
    df["AMT_CREDIT"] = df["AMT_CREDIT"].replace(0, np.nan)
    df["AMT_DOWN_PAYMENT"] = df["AMT_DOWN_PAYMENT"].clip(lower=0)

    # --- Replace sentinel 365243 in DAYS columns ---
    days_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
                 "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    for col in days_cols:
        if col in df.columns:
            df[col] = df[col].replace(365243, np.nan)

    # --- Replace sentinel -1 in SELLERPLACE_AREA ---
    df["SELLERPLACE_AREA"] = df["SELLERPLACE_AREA"].replace(-1, np.nan)

    # --- Binary flags per application ---
    df["IS_APPROVED"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    df["IS_REFUSED"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    df["IS_CANCELED"] = (df["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
    df["IS_REVOLVING"] = (df["NAME_CONTRACT_TYPE"] == "Revolving loans").astype(int)
    df["IS_WALK_IN"] = (df["NAME_PRODUCT_TYPE"] == "walk-in").astype(int)
    df["IS_CHANNEL_AP"] = (df["CHANNEL_TYPE"] == "AP+ (Cash loan)").astype(int)
    df["IS_SCOFR"] = (df["CODE_REJECT_REASON"] == "SCOFR").astype(int)
    df["IS_REJECTED_HC"] = (df["CODE_REJECT_REASON"] == "HC").astype(int)
    df["IS_REJECTED_LIMIT"] = (df["CODE_REJECT_REASON"] == "LIMIT").astype(int)
    df["IS_REFRESHED"] = (df["NAME_CLIENT_TYPE"] == "Refreshed").astype(int)
    df["IS_PURPOSE_REFUSED"] = (df["NAME_CASH_LOAN_PURPOSE"] == "Refusal to name the goal").astype(int)
    df["IS_PURPOSE_DEBT"] = (df["NAME_CASH_LOAN_PURPOSE"].isin(
        ["Payments on other loans", "Car repairs",
         "Urgent needs", "Medicine"])).astype(int)
    df["IS_HAD_DRAWING"] = (df["DAYS_FIRST_DRAWING"].notna()).astype(int)
    df["IS_INSURED"] = (df["NFLAG_INSURED_ON_APPROVAL"] == 1).astype(int)

    # --- Credit to application ratio ---
    df["CREDIT_TO_APP_RATIO"] = df["AMT_CREDIT"] / df["AMT_APPLICATION"]
    df["IS_DOWNGRADED"] = (df["CREDIT_TO_APP_RATIO"] < 1).astype(int)

    # --- Aggregation to SK_ID_CURR grain ---
    agg = df.groupby("SK_ID_CURR").agg(

        # Coverage
        COUNT_PREV_TOTAL=("SK_ID_PREV", "count"),
        COUNT_PREV_APPROVED=("IS_APPROVED", "sum"),
        COUNT_PREV_REFUSED=("IS_REFUSED", "sum"),
        COUNT_PREV_CANCELED=("IS_CANCELED", "sum"),

        # Risk flags — mandatory
        FLAG_EVER_REFUSED=("IS_REFUSED", "max"),
        FLAG_REJECTED_SCOFR=("IS_SCOFR", "max"),
        FLAG_REJECTED_HC=("IS_REJECTED_HC", "max"),
        FLAG_REJECTED_LIMIT=("IS_REJECTED_LIMIT", "max"),
        FLAG_REVOLVING_LOAN=("IS_REVOLVING", "max"),
        FLAG_WALK_IN=("IS_WALK_IN", "max"),
        FLAG_CHANNEL_AP_PLUS=("IS_CHANNEL_AP", "max"),
        FLAG_EVER_REFRESHED=("IS_REFRESHED", "max"),
        FLAG_PURPOSE_REFUSED_NAME=("IS_PURPOSE_REFUSED", "max"),
        FLAG_PURPOSE_DEBT_RELATED=("IS_PURPOSE_DEBT", "max"),
        FLAG_HAD_DRAWING=("IS_HAD_DRAWING", "max"),
        FLAG_EVER_INSURED=("IS_INSURED", "max"),
        FLAG_EVER_DOWNGRADED=("IS_DOWNGRADED", "max"),

        # Amount features
        MEAN_AMT_CREDIT=("AMT_CREDIT", "mean"),
        MAX_AMT_CREDIT=("AMT_CREDIT", "max"),
        MEAN_AMT_ANNUITY=("AMT_ANNUITY", "mean"),
        MEAN_CNT_PAYMENT=("CNT_PAYMENT", "mean"),
        MEAN_CREDIT_TO_APP_RATIO=("CREDIT_TO_APP_RATIO", "mean"),

        # Recency
        DAYS_SINCE_LAST_DECISION=("DAYS_DECISION", "max"),

    ).reset_index()

    # --- Post-aggregation features ---
    agg["REFUSAL_RATE"] = (
            agg["COUNT_PREV_REFUSED"] /
            agg["COUNT_PREV_TOTAL"].replace(0, np.nan)
    )
    agg["APPROVAL_RATE"] = (
            agg["COUNT_PREV_APPROVED"] /
            agg["COUNT_PREV_TOTAL"].replace(0, np.nan)
    )

    # Convert recency to positive days
    agg["DAYS_SINCE_LAST_DECISION"] = agg["DAYS_SINCE_LAST_DECISION"] * -1

    return agg


def aggregate_pos_cash(df_pos):
    """
    Aggregate POS_CASH_balance to SK_ID_CURR grain.
    Aggregates SK_ID_PREV first then SK_ID_CURR.

    Usage:
        pos_agg = aggregate_pos_cash(df_pos)
        df_train_fe = df_train_fe.merge(pos_agg, on="SK_ID_CURR", how="left")
    """
    df = df_pos.copy()

    # --- Binary flags per monthly record ---
    df["IS_DPD"] = (df["SK_DPD"] > 0).astype(int)
    df["IS_DPD_DEF"] = (df["SK_DPD_DEF"] > 0).astype(int)
    df["IS_AMORTIZED"] = (df["NAME_CONTRACT_STATUS"] == "Amortized debt").astype(int)
    df["IS_DEMAND"] = (df["NAME_CONTRACT_STATUS"] == "Demand").astype(int)
    df["IS_RETURNED"] = (df["NAME_CONTRACT_STATUS"] == "Returned to the store").astype(int)
    df["IS_ACTIVE"] = (df["NAME_CONTRACT_STATUS"] == "Active").astype(int)
    df["IS_COMPLETED"] = (df["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

    # --- Recency filters ---
    df["IS_LAST_3M"] = (df["MONTHS_BALANCE"] >= -3).astype(int)
    df["IS_LAST_6M"] = (df["MONTHS_BALANCE"] >= -6).astype(int)
    df["IS_LAST_12M"] = (df["MONTHS_BALANCE"] >= -12).astype(int)

    # --- Completion ratio per contract ---
    df["COMPLETION_RATIO"] = (
            (df["CNT_INSTALMENT"] - df["CNT_INSTALMENT_FUTURE"]) /
            df["CNT_INSTALMENT"].replace(0, np.nan)
    )

    # --- Aggregate to SK_ID_PREV grain ---
    prev_agg = df.groupby("SK_ID_PREV").agg(

        POS_FLAG_EVER_DPD=("IS_DPD", "max"),
        POS_FLAG_EVER_DPD_DEF=("IS_DPD_DEF", "max"),
        POS_FLAG_EVER_AMORTIZED=("IS_AMORTIZED", "max"),
        POS_FLAG_EVER_DEMAND=("IS_DEMAND", "max"),
        POS_FLAG_EVER_RETURNED=("IS_RETURNED", "max"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
        POS_COUNT_MONTHS_DPD=("IS_DPD", "sum"),
        POS_COUNT_MONTHS_DPD_DEF=("IS_DPD_DEF", "sum"),
        POS_MEAN_COMPLETION_RATIO=("COMPLETION_RATIO", "mean"),
        POS_COUNT_ACTIVE=("IS_ACTIVE", "sum"),
        POS_COUNT_COMPLETED=("IS_COMPLETED", "sum"),
        POS_SK_ID_CURR=("SK_ID_CURR", "first"),

    ).reset_index()

    # --- Aggregate to SK_ID_CURR grain ---
    agg = prev_agg.groupby("POS_SK_ID_CURR").agg(

        POS_FLAG_EVER_DPD=("POS_FLAG_EVER_DPD", "max"),
        POS_FLAG_EVER_DPD_DEF=("POS_FLAG_EVER_DPD_DEF", "max"),
        POS_FLAG_EVER_AMORTIZED=("POS_FLAG_EVER_AMORTIZED", "max"),
        POS_FLAG_EVER_DEMAND=("POS_FLAG_EVER_DEMAND", "max"),
        POS_FLAG_EVER_RETURNED=("POS_FLAG_EVER_RETURNED", "max"),
        POS_MAX_DPD=("POS_MAX_DPD", "max"),
        POS_MAX_DPD_DEF=("POS_MAX_DPD_DEF", "max"),
        POS_SUM_MONTHS_DPD=("POS_COUNT_MONTHS_DPD", "sum"),
        POS_SUM_MONTHS_DPD_DEF=("POS_COUNT_MONTHS_DPD_DEF", "sum"),
        POS_MEAN_COMPLETION_RATIO=("POS_MEAN_COMPLETION_RATIO", "mean"),
        POS_COUNT_LOANS=("POS_SK_ID_CURR", "count"),

    ).reset_index().rename(columns={"POS_SK_ID_CURR": "SK_ID_CURR"})

    return agg


def aggregate_installments(df_inst):
    """
    Aggregate installments_payments to SK_ID_CURR grain.
    Most behaviourally rich table — captures payment timing and amounts.

    Usage:
        inst_agg = aggregate_installments(df_inst)
        df_train_fe = df_train_fe.merge(inst_agg, on="SK_ID_CURR", how="left")
    """
    df = df_inst.copy()

    # --- Data quality fixes ---
    df["DAYS_ENTRY_PAYMENT"] = df["DAYS_ENTRY_PAYMENT"].clip(lower=-2915)

    # --- Engineered payment behaviour columns ---
    df["DAYS_LATE"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["AMT_UNDERPAID"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)

    # --- Binary flags per instalment ---
    df["IS_LATE"] = (df["DAYS_LATE"] > 0).astype(int)
    df["IS_LATE_30"] = (df["DAYS_LATE"] > 30).astype(int)
    df["IS_LATE_60"] = (df["DAYS_LATE"] > 60).astype(int)
    df["IS_LATE_90"] = (df["DAYS_LATE"] > 90).astype(int)
    df["IS_UNDERPAID"] = (df["AMT_UNDERPAID"] > 0).astype(int)
    df["IS_RESCHEDULED"] = (df["NUM_INSTALMENT_VERSION"] > 1).astype(int)

    # --- Recency filters (using DAYS_INSTALMENT as time reference) ---
    df["IS_LAST_3M"] = (df["DAYS_INSTALMENT"] >= -90).astype(int)
    df["IS_LAST_6M"] = (df["DAYS_INSTALMENT"] >= -180).astype(int)
    df["IS_LAST_12M"] = (df["DAYS_INSTALMENT"] >= -365).astype(int)

    # --- Aggregate to SK_ID_CURR grain ---
    agg = df.groupby("SK_ID_CURR").agg(

        # Late payment flags
        INST_FLAG_EVER_LATE=("IS_LATE", "max"),
        INST_FLAG_EVER_LATE_30=("IS_LATE_30", "max"),
        INST_FLAG_EVER_LATE_60=("IS_LATE_60", "max"),
        INST_FLAG_EVER_LATE_90=("IS_LATE_90", "max"),
        INST_COUNT_LATE=("IS_LATE", "sum"),
        INST_MAX_DAYS_LATE=("DAYS_LATE", "max"),
        INST_MEAN_DAYS_LATE=("DAYS_LATE", "mean"),

        # Underpayment flags
        INST_FLAG_EVER_UNDERPAID=("IS_UNDERPAID", "max"),
        INST_COUNT_UNDERPAID=("IS_UNDERPAID", "sum"),
        INST_SUM_AMT_UNDERPAID=("AMT_UNDERPAID", "sum"),
        INST_MEAN_PAYMENT_RATIO=("PAYMENT_RATIO", "mean"),

        # Rescheduling
        INST_FLAG_RESCHEDULED=("IS_RESCHEDULED", "max"),
        INST_COUNT_RESCHEDULES=("IS_RESCHEDULED", "sum"),

        # Recency — last 12 months
        INST_FLAG_LATE_LAST_12M=("IS_LATE",
                                 lambda x: x[df.loc[x.index, "IS_LAST_12M"] == 1].max()
                                 if (df.loc[x.index, "IS_LAST_12M"] == 1).any()
                                 else 0),
        INST_COUNT_LATE_LAST_12M=("IS_LATE",
                                  lambda x: x[df.loc[x.index, "IS_LAST_12M"] == 1].sum()
                                  if (df.loc[x.index, "IS_LAST_12M"] == 1).any()
                                  else 0),

        # Volume
        INST_COUNT_TOTAL=("SK_ID_PREV", "count"),
        INST_MEAN_AMT_INSTALMENT=("AMT_INSTALMENT", "mean"),

    ).reset_index()

    return agg


def aggregate_credit_card(df_cc):
    """
    Aggregate credit_card_balance to SK_ID_CURR grain.
    Key signals: utilisation, over-limit, ATM drawings, minimum payment.

    Usage:
        cc_agg = aggregate_credit_card(df_cc)
        df_train_fe = df_train_fe.merge(cc_agg, on="SK_ID_CURR", how="left")
    """
    df = df_cc.copy()

    # --- Data quality fixes ---
    for col in ["AMT_BALANCE", "AMT_RECEIVABLE_PRINCIPAL",
                "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # --- Key engineered features per monthly record ---
    df["UTILISATION"] = (
            df["AMT_BALANCE"] /
            df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    )
    df["PAYMENT_VS_MIN"] = (
            df["AMT_PAYMENT_CURRENT"] /
            df["AMT_INST_MIN_REGULARITY"].replace(0, np.nan)
    )

    # --- Binary flags per monthly record ---
    df["IS_DPD"] = (df["SK_DPD"] > 0).astype(int)
    df["IS_DPD_DEF"] = (df["SK_DPD_DEF"] > 0).astype(int)
    df["IS_OVER_LIMIT"] = (df["UTILISATION"] > 1.0).astype(int)
    df["IS_HIGH_UTIL"] = (df["UTILISATION"] > 0.9).astype(int)
    df["IS_MIN_PAYMENT"] = (df["PAYMENT_VS_MIN"] <= 1.1).astype(int)
    df["IS_ATM_DRAWING"] = (df["AMT_DRAWINGS_ATM_CURRENT"] > 0).astype(int)

    # --- Recency filters ---
    df["IS_LAST_3M"] = (df["MONTHS_BALANCE"] >= -3).astype(int)
    df["IS_LAST_6M"] = (df["MONTHS_BALANCE"] >= -6).astype(int)
    df["IS_LAST_12M"] = (df["MONTHS_BALANCE"] >= -12).astype(int)

    # --- Aggregate to SK_ID_CURR grain ---
    agg = df.groupby("SK_ID_CURR").agg(

        # Utilisation features — mandatory
        CC_FLAG_EVER_OVER_LIMIT=("IS_OVER_LIMIT", "max"),
        CC_FLAG_EVER_HIGH_UTIL=("IS_HIGH_UTIL", "max"),
        CC_MAX_UTILISATION=("UTILISATION", "max"),
        CC_MEAN_UTILISATION=("UTILISATION", "mean"),
        CC_COUNT_MONTHS_OVER_LIMIT=("IS_OVER_LIMIT", "sum"),
        CC_COUNT_MONTHS_HIGH_UTIL=("IS_HIGH_UTIL", "sum"),

        # Payment behaviour — mandatory
        CC_FLAG_MIN_PAYMENT_ONLY=("IS_MIN_PAYMENT", "max"),
        CC_COUNT_MONTHS_MIN_PAYMENT=("IS_MIN_PAYMENT", "sum"),
        CC_MEAN_PAYMENT_VS_MIN=("PAYMENT_VS_MIN", "mean"),

        # ATM drawing features — mandatory
        CC_FLAG_EVER_ATM_DRAWING=("IS_ATM_DRAWING", "max"),
        CC_COUNT_MONTHS_ATM_DRAWING=("IS_ATM_DRAWING", "sum"),
        CC_SUM_AMT_DRAWINGS_ATM=("AMT_DRAWINGS_ATM_CURRENT", "sum"),

        # DPD features — mandatory
        CC_FLAG_EVER_DPD=("IS_DPD", "max"),
        CC_FLAG_EVER_DPD_DEF=("IS_DPD_DEF", "max"),
        CC_MAX_DPD=("SK_DPD", "max"),
        CC_COUNT_MONTHS_DPD=("IS_DPD", "sum"),

        # Balance features
        CC_MEAN_AMT_BALANCE=("AMT_BALANCE", "mean"),
        CC_MAX_AMT_BALANCE=("AMT_BALANCE", "max"),
        CC_MEAN_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),

        # Drawing behaviour
        CC_MEAN_AMT_DRAWINGS=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_COUNT_MONTHS_WITH_DRAWINGS=("AMT_DRAWINGS_CURRENT",
                                       lambda x: (x > 0).sum()),

    ).reset_index()

    # --- Post-aggregation features ---
    agg["CC_ATM_TO_TOTAL_DRAWING_RATIO"] = (
            agg["CC_SUM_AMT_DRAWINGS_ATM"] /
            (agg["CC_MEAN_AMT_DRAWINGS"] * agg["CC_COUNT_MONTHS_WITH_DRAWINGS"])
            .replace(0, np.nan)
    )

    # Coverage flag
    agg["FLAG_HAS_CREDIT_CARD"] = 1

    return agg


def build_feature_matrix(df_main, bureau_agg, bb_agg, prev_agg,
                         pos_agg, inst_agg, cc_agg):
    """
    Merge all aggregated child table features onto the main application table.
    Uses left join — applicants with no child records get NaN.

    Usage:
        df_features = build_feature_matrix(
            df_train_fe, bureau_agg, bb_agg, prev_agg,
            pos_agg, inst_agg, cc_agg
        )
    """
    df = df_main.copy()

    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(bb_agg, on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    df = df.merge(pos_agg, on="SK_ID_CURR", how="left")
    df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
    df = df.merge(cc_agg, on="SK_ID_CURR", how="left")

    # FLAG_NO_BUREAU_HISTORY — thin file applicants
    df["FLAG_NO_BUREAU_HISTORY"] = df["COUNT_BUREAU_TOTAL"].isna().astype(int)

    # FLAG_NO_PREV_APPLICATION
    df["FLAG_NO_PREV_APPLICATION"] = df["COUNT_PREV_TOTAL"].isna().astype(int)

    # FLAG_HAS_CREDIT_CARD — fill NaN with 0 for applicants with no card
    df["FLAG_HAS_CREDIT_CARD"] = df["FLAG_HAS_CREDIT_CARD"].fillna(0).astype(int)

    print(f"Feature matrix shape: {df.shape}")
    print(f"Total features: {df.shape[1] - 2}")  # minus SK_ID_CURR and TARGET

    return df




