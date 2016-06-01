import numpy as np
import scipy as sp
import scipy.optimize
import itertools
import codecs
import csv

# foo.__code__.co_argcount

GENERATED_FEATURES = [
    ('ratio_diff', lambda a, b, c, d: (a - b) / (c - d)),
    ('diff', lambda a, b: (a - b)),
    ('ratio', lambda a, b: (a / b)),
    ('2-poly', lambda a, b: (5 * (a ** 2) * (b ** 2)) + (4 * a * b) + 2)
]

NUM_PARAMS = 6
SAMPLE_COUNT = 1000  # 10000
max_args = 0
best_args = {}


def random_range(range):
    return (np.random.ranf(len(range)) * range * 2) - range


def score_expression(score, count, value):
    if value < -5 or value > 5:
        score += 1
    count += 1
    return (score, count)


def sample_score(f, x):
    score = 0
    count = 0
    for i in range(SAMPLE_COUNT):
        pvec = random_range(x)
        score, count = score_expression(score, count, f(*pvec))

    result = float(score) / float(count)
    return result


def permutations_score(f, x):
    arg_count = f[1].__code__.co_argcount
    pool = [i for i in range(NUM_PARAMS)]

    all_perm = itertools.permutations(pool, arg_count)
    result = 1
    for perm in all_perm:
        x2 = [x[idx] for idx in perm]
        s = sample_score(f[1], x2)
        if s < result:
            result = s
            best_args[f[0]] = perm

    return result


def objective_function(x):
    sum = 0
    for f in GENERATED_FEATURES:
        sum += permutations_score(f, x)
    result = sum / len(GENERATED_FEATURES)
    print("{}:{}".format(result, x))
    return result


def generate(filename, x):
    with codecs.open(filename, "w", "utf-8") as fp:
        writer = csv.writer(fp)
        header = ["x{}".format(x) for x in range(NUM_PARAMS)]
        header += [x[0] for x in GENERATED_FEATURES]
        writer.writerow(header)

        for i in range(SAMPLE_COUNT):
            pvec = random_range(x).tolist()
            row = pvec[:]
            for f in GENERATED_FEATURES:
                args = best_args[f[0]]
                x2 = [pvec[idx] for idx in args]
                row += [f[1](*x2)]
            writer.writerow(row)


def main():
    global max_args
    for f in GENERATED_FEATURES:
        c = f[1].__code__.co_argcount
        max_args = max(c, max_args)

    x0 = np.random.ranf(NUM_PARAMS) * 10
    res = sp.optimize.minimize(objective_function, x0, method='nelder-mead',
                               options={'xtol': 1e-1, 'disp': True, 'maxiter': 10})
    print(res)
    print(res.x)

    objective_function(res.x)

    for key in best_args:
        args = best_args[key]
        print("{} ::: {}".format(key, args))

    generate("/Users/jeff/temp/dataset.csv", res.x)


# Allow windows to multi-thread (unneeded on advanced OS's)
# See: https://docs.python.org/2/library/multiprocessing.html
if __name__ == '__main__':
    main()
