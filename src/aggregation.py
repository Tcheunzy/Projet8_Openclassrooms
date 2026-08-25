import pandas as pd


def aggregate_table(df: pd.DataFrame, group_key: str, prefix: str, exclude_cols: list = None) -> pd.DataFrame:
    """
    Agrège une table annexe générique (credit_card_balance, installments_payments,
    POS_CASH_balance) au niveau `group_key` (typiquement SK_ID_CURR).
    Colonnes one-hot -> moyenne (proportion). Colonnes numériques -> min/max/mean/sum.
    """
    df = df.copy()
    exclude_cols = exclude_cols or []

    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)
        df_encoded = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    else:
        dummies = pd.DataFrame()
        df_encoded = df

    numeric_cols = df_encoded.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols + [group_key]]

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean'] if col in dummies.columns else ['min', 'max', 'mean', 'sum']

    agg_result = df_encoded.groupby(group_key).agg(agg_dict)
    agg_result.columns = [prefix + '_' + '_'.join(col).strip() for col in agg_result.columns.values]
    agg_result = agg_result.reset_index()
    agg_result[f'{prefix}_COUNT'] = df_encoded.groupby(group_key).size().values

    return agg_result


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège bureau_balance (historique mensuel par crédit) dans bureau, puis
    agrège bureau au niveau SK_ID_CURR.
    """
    # Historique mensuel -> résumé par crédit (SK_ID_BUREAU)
    bureau_balance_dummies = pd.get_dummies(bureau_balance['STATUS'], prefix='STATUS')
    bureau_balance_encoded = pd.concat(
        [bureau_balance[['SK_ID_BUREAU', 'MONTHS_BALANCE']], bureau_balance_dummies], axis=1
    )
    bureau_balance_agg = bureau_balance_encoded.groupby('SK_ID_BUREAU').agg(
        MONTHS_BALANCE_min=('MONTHS_BALANCE', 'min'),
        MONTHS_BALANCE_max=('MONTHS_BALANCE', 'max'),
        MONTHS_BALANCE_count=('MONTHS_BALANCE', 'count'),
        **{f'{col}_mean': (col, 'mean') for col in bureau_balance_dummies.columns}
    ).reset_index()

    # Fusion dans bureau
    bureau_enriched = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')

    # Encodage des catégorielles de bureau
    bureau_cat_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
    bureau_dummies = pd.get_dummies(bureau_enriched[bureau_cat_cols], prefix=bureau_cat_cols, dtype=int)
    bureau_enriched_encoded = pd.concat(
        [bureau_enriched.drop(columns=bureau_cat_cols), bureau_dummies], axis=1
    )

    numeric_cols_bureau = bureau_enriched_encoded.select_dtypes(include=['number']).columns.tolist()
    numeric_cols_bureau = [c for c in numeric_cols_bureau if c not in ['SK_ID_CURR', 'SK_ID_BUREAU']]

    agg_dict = {}
    for col in numeric_cols_bureau:
        agg_dict[col] = ['mean'] if col in bureau_dummies.columns else ['min', 'max', 'mean', 'sum']

    bureau_agg = bureau_enriched_encoded.groupby('SK_ID_CURR').agg(agg_dict)
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg = bureau_agg.reset_index()
    bureau_agg['BUREAU_CREDIT_COUNT'] = bureau_enriched_encoded.groupby('SK_ID_CURR').size().values

    return bureau_agg


def aggregate_previous_application(previous_app: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège previous_application (demandes de crédit précédentes chez Prêt à dépenser)
    au niveau SK_ID_CURR.
    """
    previous_app = previous_app.copy()

    anom_cols_previous = [c for c in previous_app.columns if c.endswith('_ANOM')]
    previous_app[anom_cols_previous] = previous_app[anom_cols_previous].astype(int)

    previous_cat_cols = previous_app.select_dtypes(include=['object']).columns.tolist()
    previous_dummies = pd.get_dummies(previous_app[previous_cat_cols], prefix=previous_cat_cols, dtype=int)
    previous_encoded = pd.concat(
        [previous_app.drop(columns=previous_cat_cols), previous_dummies], axis=1
    )

    numeric_cols_previous = previous_encoded.select_dtypes(include=['number']).columns.tolist()
    numeric_cols_previous = [c for c in numeric_cols_previous if c not in ['SK_ID_CURR', 'SK_ID_PREV']]

    agg_dict_previous = {}
    for col in numeric_cols_previous:
        agg_dict_previous[col] = ['mean'] if col in previous_dummies.columns else ['min', 'max', 'mean', 'sum']

    previous_agg = previous_encoded.groupby('SK_ID_CURR').agg(agg_dict_previous)
    previous_agg.columns = ['PREV_' + '_'.join(col).strip() for col in previous_agg.columns.values]
    previous_agg = previous_agg.reset_index()
    previous_agg['PREV_APPLICATION_COUNT'] = previous_encoded.groupby('SK_ID_CURR').size().values

    return previous_agg

if __name__ == "__main__":
    test_df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [10, 11, 12],
        "AMT_BALANCE": [100, 200, 50],
    })
    result = aggregate_table(test_df, group_key="SK_ID_CURR", prefix="CC", exclude_cols=["SK_ID_PREV"])
    print(result)
    # Attendu : pour SK_ID_CURR=1 -> CC_AMT_BALANCE_sum=300, mean=150, min=100, max=200, CC_COUNT=2
    #           pour SK_ID_CURR=2 -> CC_AMT_BALANCE_sum=50, ..., CC_COUNT=1