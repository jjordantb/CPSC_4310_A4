#
# This script is to process all of the data into a single large csv file
#
import os
import re

root_dir = '../res'
output_file = '../res/mass.csv'

for folder, subs, files in os.walk(root_dir):
    with open(output_file, 'w') as dest:
        for filename in files:
            if filename != 'Summary.txt':
                with open(os.path.join(folder, filename), 'r') as src:
                    if 'spam' in filename:
                        dest.write('spam,' + src.readline().replace("Subject: ", ""))
                    else:
                        dest.write('ham,' + src.readline().replace("Subject: ", ""))
