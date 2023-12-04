import pandas as pd

# Load in csv
first = pd.read_csv('data/Test_labels.csv')
second = pd.read_csv('data/first submission.csv')

# Find the rows that are different
diff = first[first['class'] != second['class']]

# Print the rows that are different
print(diff)