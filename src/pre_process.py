#
# This script is to process all of the data into a single large csv file
#
import os

root_dir = '../res'

data = []
for folder, subs, files in os.walk(root_dir):
    for filename in files:
        if filename != 'Summary.txt':
            with open(os.path.join(folder, filename), 'r') as src:
                line = 'spam' if ('spam' in filename) else 'ham'
                line = line + '\t' + src.readline().replace('Subject: ', '').decode('utf-8', 'ignore').encode('utf-8')
                data.append(line)

with open('../res/all.csv', 'w') as fp:
    fp.write(''.join(data))

