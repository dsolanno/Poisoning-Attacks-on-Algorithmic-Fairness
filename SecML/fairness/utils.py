import numpy as np

import math
import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal  # generating synthetic data

# SEC_ML imports
from secml.data.c_dataset import CDataset


SEED = 999
seed(
    SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def generate_synthetic_data(plot_data=False, distrib_distance=np.array([5, 5]),
                            ax=None, title=""):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

    n_samples = 2000  # generate these many data points per class

    disc_factor = math.pi / 4.0  # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv, X, y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = np.array([2, 2]), [[5, 1], [1, 5]]
    mu2, sigma2 = np.array(mu1 - distrib_distance), [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, 0)  # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0, n_samples * 2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]

    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)],
                              [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    """ Generate the sensitive feature here """
    x_control = []  # this array holds the sensitive feature value
    for i in range(0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s

        r = np.random.uniform()  # generate a random number from 0 to 1

        if r < p1:  # the first cluster is the positive class
            x_control.append(1.0)  # 1.0 means its male
        else:
            x_control.append(0.0)  # 0.0 -> female

    x_control = np.array(x_control)

    """ Show the data """
    if plot_data:
        num_to_draw = 200  # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]

        if ax is not None:
            ax.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1],
                       color='green', marker='x', s=30, linewidth=1.5,
                       label="Unprivileged favorable")
            ax.scatter(X_s_0[y_s_0 == 0.0][:, 0], X_s_0[y_s_0 == 0.0][:, 1],
                       color='red', marker='x', s=30, linewidth=1.5,
                       label="Unprivileged unfavorable")
            ax.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1],
                       color='green', marker='o', facecolors='none', s=30,
                       label="Privileged favorable")
            ax.scatter(X_s_1[y_s_1 == 0.0][:, 0], X_s_1[y_s_1 == 0.0][:, 1],
                       color='red', marker='o', facecolors='none', s=30,
                       label="Privileged unfavorable")

            ax.tick_params(axis='x', which='both', bottom='off', top='off',
                           labelbottom='off')  # dont need the ticks to see the data distribution
            ax.tick_params(axis='y', which='both', left='off', right='off',
                           labelleft='off')
            # plt.legend(loc=2, fontsize=15)
            ax.set_title(title)

            # plt.xlim((-15,10))
            # plt.ylim((-10,15))
            # plt.savefig("img/data.png")
            # plt.show()
        else:
            plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1],
                        color='green', marker='x', s=30, linewidth=1.5,
                        label="Unprivileged favorable")
            plt.scatter(X_s_0[y_s_0 == 0.0][:, 0], X_s_0[y_s_0 == 0.0][:, 1],
                        color='red', marker='x', s=30, linewidth=1.5,
                        label="Unprivileged unfavorable")
            plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1],
                        color='green', marker='o', facecolors='none', s=30,
                        label="Privileged favorable")
            plt.scatter(X_s_1[y_s_1 == 0.0][:, 0], X_s_1[y_s_1 == 0.0][:, 1],
                        color='red', marker='o', facecolors='none', s=30,
                        label="Privileged unfavorable")

            plt.tick_params(axis='x', which='both', bottom='off', top='off',
                            labelbottom='off')  # dont need the ticks to see the data distribution
            plt.tick_params(axis='y', which='both', left='off', right='off',
                            labelleft='off')
            # plt.legend(loc=2, fontsize=15)

            plt.legend(bbox_to_anchor=(1.01, 1.05), fontsize=15)
            # plt.xlim((-15,10))
            # plt.ylim((-10,15))
            # plt.savefig("img/data.png")
            plt.show()

    x_control = {
        "s1": x_control}  # all the sensitive features are stored in a dictionary
    return X, y, x_control


def calculate_disparate_impact(y, sensible_att_vals, privileged_classes=1,
                               favorable_output=1, verbose=False):
    privileged = y[sensible_att_vals == privileged_classes]
    unprivileged = y[sensible_att_vals != privileged_classes]

    unprivileged_favorable = unprivileged[unprivileged == favorable_output]
    privileged_favorable = privileged[privileged == favorable_output]

    n1 = (len(unprivileged_favorable) / len(unprivileged))
    n2 = (len(privileged_favorable) / len(privileged))

    if verbose:
        print("\tN1: ", n1)
        print("\tN2: ", n2)

    disparate_impact = n1 / (max(n2, 0.1))
    return disparate_impact


def get_false_positive_rate(y_true, y_pred, favorable_output):
    _tmp = y_pred[y_true != favorable_output]

    fp = _tmp[_tmp == favorable_output]
    if len(_tmp) == 0:
        return 0

    return len(fp) / len(_tmp)


def get_false_negative_rate(y_true, y_pred, favorable_output):
    _tmp = y_pred[y_true == favorable_output]

    fn = _tmp[_tmp != favorable_output]

    if len(_tmp) == 0:
        return 0

    return len(fn) / len(_tmp)


from sklearn.model_selection import train_test_split


def load_data():
    random_state = 999

    N = 9  # Max euclidean distance between average of distributions

    dimp_in_data = []
    euc_distances = []
    dimp_scenarios = []

    n = 13

    ## Generating data
    euc_dist = n
    i = np.sqrt((euc_dist ** 2) / 2)
    X, y, X_control = generate_synthetic_data(False, np.array([i, i]))
    formatted_X = np.array([X[:, 0], X[:, 1], X_control[
        's1']]).T  ## Concatenating X with sensible att

    sec_ml_dataset_all = CDataset(X, y)
    sensible_att_all = X_control['s1']

    euc_distances.append(n)
    dimp_in_data.append(calculate_disparate_impact(sec_ml_dataset_all.Y.get_data(),
                                                   sensible_att_all))

    ## Splitting data.
    X_train_val, X_test, y_train_val, y_test = train_test_split(formatted_X, y,
                                                                test_size=0.2,
                                                                random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.5,
                                                      random_state=random_state)

    training = CDataset(X_train[:, :2], y_train)
    training_sensible_att = X_train[:, 2]

    validation = CDataset(X_val[:, :2], y_val)
    validation_sensible_att = X_val[:, 2]
    val_lambda = np.zeros(validation.num_samples)

    test = CDataset(X_test[:, :2], y_test)
    test_sensible_att = X_test[:, 2]

    ## GENERATING DATA FOR WHITE BOX ATTACK
    X2, y2, X_control2 = generate_synthetic_data(False, np.array([i, i]))
    formatted_X2 = np.array([X2[:, 0], X2[:, 1], X_control2[
        's1']]).T  ## Concatenating X with sensible att

    sec_ml_dataset_all2 = CDataset(X2, y2)
    sensible_att_all2 = X_control2['s1']

    ## Splitting data.
    X_train_val2, X_test2, y_train_val2, y_test2 = train_test_split(formatted_X2,
                                                                    y2,
                                                                    test_size=0.2,
                                                                    random_state=random_state)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train_val2,
                                                          y_train_val2,
                                                          test_size=0.5,
                                                          random_state=random_state)

    training2 = CDataset(X_train2[:, :2], y_train2)
    training_sensible_att2 = X_train2[:, 2]

    validation2 = CDataset(X_val2[:, :2], y_val2)
    validation_sensible_att2 = X_val2[:, 2]
    val_lambda2 = np.zeros(validation2.num_samples)

    test2 = CDataset(X_test2[:, :2], y_test)
    test_sensible_att2 = X_test2[:, 2]

    scenario = {
        "name": "Use case 4 - {}".format(n),
        "description": "Disparate impact attack. \n Euclidean distance between group averages: {}\n".format(
            n),
        "training": training,
        "training_sensible_att": training_sensible_att,
        "validation": validation,
        "validation_sensible_att": validation_sensible_att,
        "lambda_validation": val_lambda,
        "test": test,
        "test_sensible_att": test_sensible_att,
        "all_data": sec_ml_dataset_all,
        "all_sensible_att": sensible_att_all,
        "black_box_training": training2,
        "black_box_training_sensible_att": training_sensible_att2,
        "black_box_validation": validation2,
        "black_box_validation_sensible_att": validation_sensible_att2,
        "black_box_lambda_validation": val_lambda2,
        "black_box_test": test2,
        "black_box_test_sensible_att": test_sensible_att2,
        "black_box_all_data": sec_ml_dataset_all2,
        "black_box_all_sensible_att": sensible_att_all2,
    }



    dimp_scenarios.append(scenario)

    scenario = dimp_scenarios[0]

    training_set = training#scenario['training']
    validation_set = validation#scenario['validation']
    test_set = test#scenario['test']
    privileged_condition_validation = validation_sensible_att

    # x0, y0 = test_set[5, :].X, test_set[5, :].Y
    return training_set, validation_set, test_set, privileged_condition_validation