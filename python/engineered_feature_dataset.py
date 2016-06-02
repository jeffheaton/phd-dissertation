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

#OUTPUT_FILE = "/Users/jeff/temp/dataset.csv"
OUTPUT_FILE = "c:\\temp\\dataset.csv"
SAMPLE_COUNT = 10000
TARGET_RANGE = 200

x_count = 0
y_count = 0

def random_range(range):
    return (np.random.ranf(len(range)) * range * 2) - range

def sample_score(f,ranges):
    sample = []
    for i in range(SAMPLE_COUNT):
        x2 = random_range(ranges)
        sample.append( f(*x2) )

    min = np.min(sample)
    max = np.max(sample)
    return abs(TARGET_RANGE - (max-min))


def objective_function(x):
    sum = 0

    idx = 0
    for f in GENERATED_FEATURES:
        c = f[1].__code__.co_argcount
        slice = x[idx:idx+c]
        sum += sample_score(f[1], slice)
        idx+=c

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
    x_count = 0
    for f in GENERATED_FEATURES:
        c = f[1].__code__.co_argcount
        x_count += c

    x0 = (np.random.ranf(x_count) * 10)
    res = sp.optimize.minimize(objective_function, x0, method='nelder-mead',
                               options={'xtol': 1e-8, 'disp': True})
    print(res)
    print(res.x)

    #generate(OUTPUT_FILE, res.x)


# Allow windows to multi-thread (unneeded on advanced OS's)
# See: https://docs.python.org/2/library/multiprocessing.html
if __name__ == '__main__':
    main()
