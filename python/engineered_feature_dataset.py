import inspect

import numpy as np
import pandas as pd
import math

NUM_ROWS = 50000
SEED = 42
OUTLIER_THRESH = 2.5

NAME = 'name'
FN = 'fn'

GENERATED_FEATURES = [
    {NAME: 'ratio_diff', FN: lambda a, b, c, d: (a - b) / (c - d)},
    {NAME: 'diff', FN: lambda a, b: (a - b)},
    {NAME: 'ratio', FN: lambda a, b: (a / b)},
    {NAME: 'ratio_poly2', FN: lambda a, b: 1.0 / ((5 * (a ** 2) * (b ** 2)) + (4 * a * b) + 2)},
    {NAME: 'coef_ratio', FN: lambda a, b: (a / b)},
    {NAME: 'poly2', FN: lambda a, b: (5 * (a ** 2) * (b ** 2)) + (4 * a * b) + 2},
    {NAME: 'ratio_poly', FN: lambda x: 1 / (5 * x + 8 * x ** 2)},
    {NAME: 'poly', FN: lambda x: 1 + 5 * x + 8 * x ** 2},
    {NAME: 'sqrt', FN: lambda x: np.sqrt(x)},
    {NAME: 'log', FN: lambda x: np.log(x)},
    {NAME: 'pow', FN: lambda x: x ** 2}
]


def generate_dataset(rows):
    df = pd.DataFrame(index=range(1, rows + 1))

    predictor_columns = []
    y_columns = []

    for f in GENERATED_FEATURES:
        arg_count = len(inspect.signature(f[FN]).parameters)
        for arg_idx in range(arg_count):
            col_name = "{}-x{}".format(f[NAME], arg_idx)
            predictor_columns.append(col_name)
            df[col_name] = (2 * np.random.random(rows)) - 1
        y_columns.append("{}-y0".format(f[NAME]))

    idx = 1
    generated_columns = []

    for f in GENERATED_FEATURES:
        col_name = "{}-y0".format(f[NAME])
        generated_columns.append(col_name)
        arg_count = len(inspect.signature(f[FN]).parameters)
        a = [df.iloc[:, idx + x] for x in range(arg_count)]
        df[col_name] = f[FN](*a)
    return df, predictor_columns, generated_columns


def generate_report(df, generated_columns):
    report = pd.DataFrame(index=range(len(generated_columns)))
    y = df.loc[:, generated_columns]
    report['name'] = generated_columns
    report['max'] = np.amax(y, axis=0).tolist()
    report['min'] = np.amin(y, axis=0).tolist()
    report['range'] = report['max'] - report['min']
    report['mean'] = np.mean(y, axis=0).tolist()
    report['std'] = np.std(y, axis=0).tolist()
    return report


def generate_predictor_report(df, predictor_columns):
    report = pd.DataFrame(index=range(len(predictor_columns)))
    y = df.loc[:, predictor_columns]
    report['name'] = predictor_columns
    report['max'] = np.amax(y, axis=0).tolist()
    report['min'] = np.amin(y, axis=0).tolist()
    return report


def remove_outliers(df, target_columns, sdev):
    for col in target_columns:
        df = df[np.abs(df[col] - df[col].mean()) <= (sdev * df[col].std())]
    return df


def main():
    np.random.seed(SEED)

    df, predictor_columns, generated_columns = generate_dataset(int(NUM_ROWS * 4))

    len1 = len(df)
    df = remove_outliers(df, generated_columns, OUTLIER_THRESH)
    len2 = len(df)

    print("Removed {}({}%) outliers.".format(len1 - len2, 100.0 * ((len1 - len2) / len1)))

    df = df.head(NUM_ROWS)
    df = df.reset_index(drop=True)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv("/Users/jeff/temp/feature_eng.csv", index_label='id')

    report = generate_report(df, generated_columns)

    print(report)


main()
