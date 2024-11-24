import pandas as pd

def process_adult():
    df = pd.read_csv('./data/adult.csv')

    selected_features = [
        "age",
        "education-num",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "Probability",
    ]

    df_processed = df[selected_features]

    df_processed.loc[:, 'race'] = df_processed['race'].apply(lambda x: 1 if x == ' White' else 0)
    # 'male' -> 1, 'female' -> 0
    df_processed.loc[:, 'sex'] = df_processed['sex'].map({' Male': 1, ' Female': 0})
    df_processed.loc[:, 'Probability'] = df_processed['Probability'].map({' >50K': 1, ' <=50K': 0})

    df_processed.to_csv('./data/adult_processed.csv', index=False)

def process_compas():
    df = pd.read_csv('./data/compas.csv')

    selected_features = [
        "sex",
        "age_cat",
        "race",
        "priors_count",
        "c_charge_degree",
        "decile_score",
        "priors_count",
        "two_year_recid",
    ]

    df_processed = df[selected_features]

    df_processed.loc[:, 'sex'] = df_processed['sex'].map({'Male': 1, 'Female': 0})
    df_processed.loc[:, 'age_cat'] = df_processed['age_cat'].map({"Greater than 45": 45, "25 - 45": 25, "Less than 25": 0})
    df_processed.loc[:, 'race'] = df_processed['race'].apply(lambda x: 1 if x == 'Caucasian' else 0)
    df_processed.loc[:, 'c_charge_degree'] = df_processed['c_charge_degree'].map({"F": 1, "M": 0})

    df_processed = df_processed.rename(columns={"two_year_recid": "Probability"})

    df_processed.to_csv('./data/compas_processed.csv', index=False)

def process_default():
    df = pd.read_csv('./data/default.csv', header=[0, 1])

    selected_features = [
        "X1",   # "LIMIT_BAL" - Credit limit
        "X2",   # "SEX" - Gender
        "X5",   # "AGE" - Age of the client
        "X6",   # "PAY_0" - Repayment status for the most recent month
        "X7",   # "PAY_2" - Repayment status for the second most recent month
        "X12",  # "BILL_AMT1" - Bill amount for the most recent month
        "X18",  # "PAY_AMT1" - Payment amount for the most recent month
        "Y",    # "default.payment.next.month" - Binary Classification Label
    ]

    df_processed = df[selected_features]

    # lowercase column names
    df_processed.columns = [f"{col[1].lower()}" for col in df_processed.columns]

    df_processed = df_processed.rename(columns={"default payment next month": "Probability"})

    # 'male' : 1 -> 1, 'female' : 2 -> 0
    df_processed.loc[:, 'sex'] = df_processed['sex'].map({1: 1, 2: 0})

    # reindex 'age' column to ensure it fits into the range [0, num_unique_ages]
    unique_ages = df_processed['age'].unique()
    age_to_index = {age: idx for idx, age in enumerate(sorted(unique_ages))}
    df_processed.loc[:, 'age'] = df_processed['age'].map(age_to_index)

    df_processed.to_csv('./data/default_processed.csv', index=False)

if __name__ == "__main__":
    process_adult()
    process_compas()
    process_default()
