import numpy as np

def convert_tree_as_set_to_adjacencies(tree):
    """
    This snippet of code converts between two representations we use for
    edges (namely, with Chow-Liu it suffices to just store edges as a set of
    pairs (i, j) with i < j), whereas when we deal with learning tree
    parameters and code Sum-Product it will be convenient to have an
    "adjacency list" representation, where we can query and find out the list
    of neighbors for any node. We store this "adjacency list" as a Python
    dictionary.

    Input
    -----
    - tree: a Python set of edges (where (i, j) being in the set means that we
        don't have to have (j, i) also stored in this set)

    Output
    ------
    - edges: a Python dictionary where `edges[i]` gives you a list of neighbors
        of node `i`
    """
    edges = {}
    for i, j in tree:
        if i not in edges:
            edges[i] = [j]
        else:
            edges[i].append(j)
        if j not in edges:
            edges[j] = [i]
        else:
            edges[j].append(i)
    return edges


class UnionFind():
    def __init__(self, nodes):
        """
        Union-Find data structure initialization sets each node to be its own
        parent (so that each node is in its own set/connected component), and
        to also have rank 0.

        Input
        -----
        - nodes: list of nodes
        """
        self.parents = {}
        self.ranks = {}

        for node in nodes:
            self.parents[node] = node
            self.ranks[node] = 0

    def find(self, node):
        """
        Finds which set/connected component that a node belongs to by returning
        the root node within that set.

        Technical remark: The code here implements path compression.

        Input
        -----
        - node: the node that we want to figure out which set/connected
            component it belongs to

        Output
        ------
        the root node for the set/connected component that `node` is in
        """
        if self.parents[node] != node:
            # path compression
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]

    def union(self, node1, node2):
        """
        Merges the connected components of two nodes.

        Inputs
        ------
        - node1: first node
        - node2: second node
        """
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:  # only merge if the connected components differ
            if self.ranks[root1] > self.ranks[root2]:
                self.parents[root2] = root1
            else:
                self.parents[root1] = root2
                if self.ranks[root1] == self.ranks[root2]:
                    self.ranks[root2] += 1


# compute mutual inf between two RVs (in case they are jointly Gaussian)
def compute_mutual_inf(x1, x2):
    N = len(x1)
    return -1/2*np.log(1 - (1/N*np.sum((x1 - x1.mean())*(x2 - x2.mean()))/x1.var()**(1/2)/x2.var()**(1/2))**2)


def compute_empirical_mutual_info_nats(var1_values, var2_values):
    """
    Compute the empirical mutual information for two random variables given a
    pair of observed sequences of those two random variables.

    Inputs
    ------
    - var1_values: observed sequence of values for the first random variable
    - var2_values: observed sequence of values for the second random variable
        where it is assumed that the i-th entries of `var1_values` and
        `var2_values` co-occur

    Output
    ------
    The empirical mutual information *in nats* (not bits)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    empirical_mutual_info_nats = compute_mutual_inf(var1_values, var2_values)
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return empirical_mutual_info_nats


def chow_liu(observations):
    """
    Run the Chow-Liu algorithm.

    Input
    -----
    - observations: a 2D NumPy array where the i-th row corresponds to the
        i-th training data point

        *IMPORTANT*: it is assumed that the nodes in the graphical model are
        numbered 0, 1, ..., up to the number of variables minus 1, where the
        number of variables in the graph is determined from `observations` by
        looking at `observations.shape[1]`

    Output
    ------
    - best_tree: a Python set consisting of edges that are in a Chow-Liu tree
        (note that if edge (i, j) is in this set, then edge (j, i) should not
        be in the set; also, for grading purposes, please present the edges
        so that for an edge (i, j) in this set, i < j
    """
    best_tree = set()  # we will add in edges to this set
    num_obs, num_vars = observations.shape
    union_find = UnionFind(range(num_vars))

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #

    # define matrix to hold mutual cross information between vertices
    A = np.zeros(shape=(num_vars, num_vars))
    # compute mutual information between vertices
    for index, value in np.ndenumerate(A):
        if index[0] < index[1]:
            A[index] = compute_empirical_mutual_info_nats(observations[:, index[0]], observations[:, index[1]])
    # iterate over matrix A while graph whill be fully connected
    connected = False
    while not connected:
        index = np.unravel_index(A.flatten().argmax(), (A.shape))
        A[index] = 0
        if union_find.find(index[0]) != union_find.find(index[1]):
            union_find.union(index[0], index[1])
            best_tree.add((np.int(index[0]), np.int(index[1])))
        else:
            connected = True
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return best_tree


def get_labeled_windowed_data(observations, window_size=7):
    """
    Split up the observations into windowed chunks. Each windowed chunk of
    observations is associated with a label vector of what the price change is
    per market *immediately after* the windowed chunk (+1 for price goes up,
    0 for no change, and -1 for price goes down). Thus, a classifier's task
    for the data is given a windowed chunk, to predict what its label is
    (i.e., given recent percent changes in all the markets, predict the
    directions of the next price changes per market).

    Inputs
    ------
    - observations: 2D array; each column is a percent-change time series data
        for a specific market
    - window_size: how large the window is (in number of time points)

    Outputs
    -------
    - windows: 3D array; each element of the outermost array is a 2D array
        of the same format as `observations` except where the number of time
        points is exactly `window_size`
    - window_labels: 2D array; `window_labels[i]` is a 1D vector of labels
        corresponding to the time point *after* the window specified by
        `windows[i]`; `window_labels[i]` says what the price change is for
        each market (+1 for going up, 0 for staying the same, and -1 for going
        down)

    *WARNING*: Note that the training data produced here is inherently not
    i.i.d. in that `windows[0]` and `windows[1]`, for instance, will largely
    overlap!
    """
    num_time_points, num_markets = observations.shape
    windows = []
    window_labels = []
    for start_idx in range(num_time_points-window_size):
        windows.append(observations[start_idx:start_idx+window_size])
        window_labels.append(1*(observations[start_idx+window_size] > 0)
                             -1*(observations[start_idx+window_size] < 0))
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    return windows, window_labels


class norm:
    def __init__(self, mu = 0, sigma = 1e-6):
        self.mu = mu
        self.sigma = sigma
    def pdf(self, x):
        d = len(self.mu)
        if d == 1:
            return 1/((2*np.pi)**(1/2)*self.sigma)*np.exp(-1/2*(x-self.mu)**2/self.sigma**2)
        else:
            return 1/((2*np.pi)**(d/2)*np.linalg.det(self.sigma)**(1/2))*np.exp(-1/2*np.dot(np.dot((x-self.mu).T, np.linalg.inv(self.sigma)), (x-self.mu)))
# global variables to be saved for the trained classifier
guess = None


def train(windows, window_labels):
    """
    Your training procedure goes here! It should train a classifier where you
    store whatever you want to store for the trained classifier as *global*
    variables. `train` will get called exactly once on the exact same training
    data you have access to. However, you will not get access to the mystery
    test data.

    Inputs
    ------
    - windows, window_labels: see the documentation for the output of
        `get_labeled_windowed_data`
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #

    # The autograder wants you to explicitly state which variables are global
    # and are supposed to thus be saved after training for use with prediction.
    global guess

    tree = chow_liu(windows.reshape((windows.shape[0], windows.shape[1]*windows.shape[2])))
    guess = 0

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------


def forecast(window):
    """
    Your forecasting method goes here! You may assume that `train` has already
    been called on training data and so any global variables you stored as a
    result of running `train` are available to you here for prediction
    purposes.

    Input
    -----
    - window: 2D array; each column is 7 days worth of percent changes in
        price for a specific market

    Output
    ------
    1D array; the i-th entry is a prediction for whether the percentage
    return will go up (+1), stay the same (0), or go down (-1) for the i-th
    market
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #

    predicted_labels = np.array([guess for idx in range(window.shape[1])])

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return predicted_labels


def main():
    # get coconut oil challenge training data
    observations = []
    with open('coconut_challenge.csv', 'r') as f:
        for line in f.readlines():
            pieces = line.split(',')
            if len(pieces) == 5:
                observations.append([float(pieces[1]),
                                     float(pieces[2]),
                                     float(pieces[3]),
                                     float(pieces[4])])
    observations = np.array(observations)
    train_windows, train_window_labels = \
        get_labeled_windowed_data(observations, window_size=7)

    train(train_windows, train_window_labels)

    # figure out accuracy of the trained classifier on predicting labels for
    # the training data
    train_predictions = []
    for window, window_label in zip(train_windows, train_window_labels):
        train_predictions.append(forecast(window))
    train_predictions = np.array(train_predictions)

    train_prediction_accuracy_plus1 = \
        np.mean(train_predictions[train_window_labels == 1]
                == train_window_labels[train_window_labels == 1])
    train_prediction_accuracy_minus1 = \
        np.mean(train_predictions[train_window_labels == -1]
                == train_window_labels[train_window_labels == -1])
    train_prediction_accuracy_0 = \
        np.mean(train_predictions[train_window_labels == 0]
                == train_window_labels[train_window_labels == 0])
    print('Training accuracy for prediction +1:',
          train_prediction_accuracy_plus1)
    print('Training accuracy for prediction -1:',
          train_prediction_accuracy_minus1)
    print('Training accuracy for prediction 0:',
          train_prediction_accuracy_0)


if __name__ == '__main__':
    main()
