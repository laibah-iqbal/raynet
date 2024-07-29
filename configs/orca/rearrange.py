import re
from collections import defaultdict

# Input and output file paths
input_file = '/home/laibah/raynet/configs/orca/results/General-#0.vec'
output_file = '/home/laibah/raynet/configs/orca/results/rearranged_results.vec'

# Regular expressions to match vector definitions and data lines
version_re = re.compile(r'^version')
run_re = re.compile(r'^run')
attr_re = re.compile(r'^attr')
vector_re = re.compile(r'^vector (\d+)')
data_re = re.compile(r'^(\d+)')

# Data structures to hold the entire content
version_line = None
runs = []
current_run = None
attributes = []
vector_definitions = []
vector_data = defaultdict(list)

# Read the input file
with open(input_file, 'r') as infile:
    for line in infile:
        if version_re.match(line):
            version_line = line
        elif run_re.match(line):
            if current_run:
                runs.append((current_run, attributes, vector_definitions, vector_data))
            current_run = line
            attributes = []
            vector_definitions = []
            vector_data = defaultdict(list)
        elif attr_re.match(line):
            attributes.append(line)
        elif vector_re.match(line):
            vector_definitions.append(line)
        elif data_re.match(line):
            vector_id = data_re.match(line).group(1)
            vector_data[vector_id].append(line)
    if current_run:
        runs.append((current_run, attributes, vector_definitions, vector_data))

# Write the rearranged data to the output file
with open(output_file, 'w') as outfile:
    if version_line:
        outfile.write(version_line)
        outfile.write("\n")
    for run, attrs, vectors, data in runs:
        outfile.write(run)
        for attr in attrs:
            outfile.write(attr)
        for vector in vectors:
            outfile.write(vector)
            vector_id = vector_re.match(vector).group(1)
            if vector_id in data:
                outfile.writelines(data[vector_id])
        outfile.write("\n")

print(f'Rearranged data written to {output_file}')