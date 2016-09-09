import codecs
import os
import csv


encoding = 'utf-8'

c = 0

with codecs.open("/Users/jeff/downloads/kdd99.csv", "r", encoding) as infile, \
    codecs.open("/Users/jeff/downloads/kdd99-2.csv", "w", encoding) as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Generate header index using comprehension.
    # Comprehension is cool, but not necessarily a beginners feature of Python.
    headers = next(reader)
    print(headers)
    header_idx = {key: value for (value, key) in enumerate(headers)}
    writer.writerow(headers)
    idx = header_idx['outcome']

    for row in reader:
        if row[idx] != 'normal.':
            row[idx] = "threat."
        writer.writerow(row)