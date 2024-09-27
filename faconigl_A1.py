import pandas as pd

data = pd.read_csv('email.tsv', sep='\t')
print(data.head())

"""
Select/develop at least 6 features (generally going to be 6 Python functions). Yes, you could
use some of these values directly – go ahead and try them out! However, using a combination
of the data for a single feature might make it perform better. You can get creative here. There
are no necessarily wrong answ
"""

# Feature 1: Getting the number of characters in the email (in thousands)

def email_length(row):
    return row['num_char']

# Feature 2: Determine whether the email is sent to multiple recipients or has more than 10 cc's

def multiple_recipients(row):
    # Returns 1 for true, 0 for false
    return int(row['to_multiple'] or (row['cc'] > 10))

# Feature 3: Whether the email subject contains an exclamation mark or the word 'urgent'

def excited_subject(row):
    return int(row['exclaim_subj'] or row['urgent_subj'])

# Feature 4: Getting the number of attached files

def num_attachments(row):
    return int(row['attach'])

# Feature 5: Whether the email contains HTMl code

def contains_html(row):
    return int(row['format'])

# Feature 6: The number of times the email contains one or more of the following words or characters: 'dollar', 'winner', 'inherit', 'viagra', 'password'

def spam_words(row):
    return sum([row['dollar'], row['winner'], row['inherit'], row['viagra'], row['password']])