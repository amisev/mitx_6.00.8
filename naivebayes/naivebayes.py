import sys
import os.path
import numpy as np

import util

from collections import Counter
from collections import defaultdict

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError
    c = Counter()
    for file_name in file_list:
        c = c + Counter(dict.fromkeys(set(util.get_words_in_file(file_name)), 1))
    return dict(c)


def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError
    words_counts = get_counts(file_list)
    files_number = len(file_list)
    words_number = len(words_counts)

    c = Counter()
    c = defaultdict(lambda: np.log(1/(files_number + words_number + 1)))
    d = {k: np.log((v + 1)/(files_number + words_number + 1)) for k, v in words_counts.items()}
    c.update(d)
    return c

def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files,
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError
    #initialize spam/ham counters
    n_spam = len(file_lists_by_category[0])
    n_ham = len(file_lists_by_category[1])
    #
    log_probability_spam = get_log_probabilities(file_lists_by_category[0])
    log_probability_ham = get_log_probabilities(file_lists_by_category[1])
    return ([log_probability_spam, log_probability_ham], [np.log(n_spam/(n_spam+n_ham)), np.log(n_ham/(n_spam+n_ham))])

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here
    #return 'spam'
    n_ratio = 0
    email_words_list = list(set(util.get_words_in_file(email_filename)))
    n_ratio = log_prior_by_category[0] - log_prior_by_category[1] + np.sum(log_probabilities_by_category[0][w] for w in email_words_list) + np.sum(np.log(1 - np.exp(log_probabilities_by_category[0][w])) for w in set(log_probabilities_by_category[0].keys()) if w not in set(email_words_list)) - np.sum(log_probabilities_by_category[1][w] for w in email_words_list) - np.sum(np.log(1 - np.exp(log_probabilities_by_category[1][w])) for w in set(log_probabilities_by_category[1].keys()) if w not in set(email_words_list))

    return 'spam' if n_ratio >=0 else 'ham'

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
