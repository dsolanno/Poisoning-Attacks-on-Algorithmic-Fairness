#!/usr/bin/env python
# coding: utf-8


from secml.ml.classifiers import CClassifierLogistic
from secml.ml.peval.metrics import CMetricAccuracy

from secml.adv.attacks import CAttackPoisoningLogisticRegression

from secml.data import CDataset

from secml.figure import CFigure
from secml.ml.classifiers.loss import CLossClassification, CLossLogistic
from secml.array import CArray

from fairness.utils import load_data


def plot_data(fig, data):
    x = data.X
    y = data.Y
    unpriv = disp_impact.unprivileged()
    pos_unpriv = (y == 1) * (unpriv == 1)
    pos_priv = (y == 1) * (unpriv == 0)
    neg_unpriv = (y == 0) * (unpriv == 1)
    neg_priv = (y == 0) * (unpriv == 0)
    size_x = 12
    size_o = 8
    fig.sp.scatter(x[pos_unpriv, 0].ravel(), x[pos_unpriv, 1].ravel(),
                   c='r', marker='x', s=size_x,
                   label='y=+1, unprivileged', alpha=1.0)
    fig.sp.scatter(x[pos_priv, 0].ravel(), x[pos_priv, 1].ravel(),
                   c='r', marker='.', s=size_o,
                   label='y=+1, privileged', alpha=1.0)
    fig.sp.scatter(x[neg_unpriv, 0].ravel(), x[neg_unpriv, 1].ravel(),
                   c='b', marker='x', s=size_x,
                   label='y=-1, unprivileged', alpha=1.0)
    fig.sp.scatter(x[neg_priv, 0].ravel(), x[neg_priv, 1].ravel(),
                   c='b', marker='.', s=size_o,
                   label='y=-1, privileged', alpha=1.0)


class CLossDisparateImpact(CLossClassification):
    """Surrogate function of disparate impact.

    Attributes
    ----------
    class_type : 'log'
    suitable_for : 'classification'

    """
    __class_type = 'dimp_log'

    def __init__(self, _privileged_condition):
        self._privileged_condition = CArray(_privileged_condition)

    def unprivileged(self):
        """Give 1 to unprivileged, 0 to privileged"""
        y = CArray.zeros(self._privileged_condition.size)
        y[self._privileged_condition == 0] = 1
        return y

    def loss(self, y_true, score, pos_label=1):
        """Computes loss_priv-loss_unpriv, which is what we aim to max"""
        # give 1 to unpriv, 0 to priv
        y = self.unprivileged()
        p_priv = (y == 0).sum() / y.size
        p_unpriv = (y == 1).sum() / y.size
        # loss = (score >= 0) != y  # zero-one loss
        loss = CLossLogistic().loss(y_true=y, score=score)  # smoothed version
        loss[y == 1] *= -p_priv / p_unpriv  # rebalance class weights
        return loss

    def dloss(self, y_true, score, pos_label=1):
        """Computes the derivative of the loss vs score."""
        y = self.unprivileged()
        p_priv = (y == 0).sum() / y.size
        p_unpriv = (y == 1).sum() / y.size
        grad = CLossLogistic().dloss(y, score, pos_label)
        grad[y == 1] *= -p_priv / p_unpriv  # rebalance class weights
        return grad


class CDisparateImpact:
    """disparate impact."""

    def __init__(self, _privileged_condition, tr, ts, clf):
        self._privileged_condition = CArray(_privileged_condition)
        self.tr = tr
        self.ts = ts
        self.clf = clf

    def unprivileged(self):
        """Give 1 to unprivileged, 0 to privileged"""
        y = CArray.zeros(self._privileged_condition.size)
        y[self._privileged_condition == 0] = 1
        return y

    def objective_function(self, xc, yc):
        # retrain clf on poisoned data
        clf = self.clf.deepcopy()
        tr = self.tr.append(CDataset(xc, yc))
        clf.fit(tr)
        y_pred = clf.predict(self.ts.X)
        unpriv = self.unprivileged()
        return y_pred[unpriv == 1].mean() / y_pred[unpriv == 0].mean()


random_state = 0

training_set, validation_set, test_set, \
privileged_condition_validation = load_data()

clf = CClassifierLogistic()
clf.fit(training_set)
print("Training of classifier complete!")

# Compute predictions on a test set
y_pred, scores = clf.predict(test_set.X, return_decision_function=True)

# Evaluate the accuracy of the classifier
metric = CMetricAccuracy()
acc = metric.performance_score(y_true=test_set.Y, y_pred=y_pred)
print("Accuracy on test set: {:.2%}".format(acc))

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.05,
    'max_iter': 1000,
    'eps': 1e-6
}

pois_attack = CAttackPoisoningLogisticRegression(
    classifier=clf,
    training_data=training_set,
    surrogate_classifier=clf,
    surrogate_data=validation_set,
    val=validation_set,
    lb=-20,
    ub=20,
    solver_type='pgd',
    solver_params=solver_params,
    random_seed=random_state,
    init_type="random")

attack_loss = CLossDisparateImpact(privileged_condition_validation)
pois_attack._attacker_loss = attack_loss
pois_attack.n_points = 1
pois_attack.verbose = 1

pois_y_pred, pois_scores, pois_ds, f_opt = \
    pois_attack.run(x=validation_set.X, y=validation_set.Y)

# this class plots the disparate impact as a function of xc
disp_impact = CDisparateImpact(
    privileged_condition_validation,
    tr=training_set,
    ts=validation_set,
    clf=clf)

#%%

width = 6
height = 5
fig = CFigure(width=width, height=height)
fig.sp.plot_fun(pois_attack._objective_function, plot_levels=False,
                multipoint=False, n_grid_points=200,
                grid_limits=[(-20, 20), (-20, 20)])
# fig.sp.plot_fgrads(pois_attack._objective_function_gradient,
#                   n_grid_points=20, grid_limits=[(-20, 20), (-20, 20)])
fig.sp.plot_decision_regions(clf, plot_background=False,
                             n_grid_points=500,
                             grid_limits=[(-20, 20), (-20, 20)])
fig.sp.grid(grid_on=True, linestyle=':', linewidth=0.5)
plot_data(fig, training_set)
fig.sp.plot_path(pois_attack.x_seq)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect('equal')
fig.sp.title("Attacker's Loss")
fig.sp.legend()
fig.savefig("loss.pdf")
fig.close()

fig = CFigure(width=width, height=height)
fig.sp.plot_fun(disp_impact.objective_function, plot_levels=False,
                multipoint=False, n_grid_points=200, cmap='jet',
                vmin=0.158, vmax=0.178,
                grid_limits=[(-20, 20), (-20, 20)], yc=1)
fig.sp.plot_decision_regions(clf, plot_background=False,
                             n_grid_points=500,
                             grid_limits=[(-20, 20), (-20, 20)])
fig.sp.grid(grid_on=True, linestyle=':', linewidth=0.5)
plot_data(fig, training_set)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect('equal')
fig.sp.title("Disparate Impact")
fig.savefig("disp_impact.pdf")
fig.close()


fig = CFigure(width=width, height=height)

plot_data(fig, training_set)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect('equal')
fig.savefig("test.pdf")
fig.close()

#%%

width = 5
fig = CFigure(width=width, height=height)
fig.sp.plot_decision_regions(clf, plot_background=True,
                             n_grid_points=500,
                             grid_limits=[(-20, 20), (-20, 20)])
fig.sp.grid(grid_on=True, linestyle=':', linewidth=0.5)
plot_data(fig, training_set)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect('equal')
fig.sp.legend()
fig.savefig("dr.pdf")
fig.close()

pois_attack.n_points = 80
pois_y_pred, pois_scores, pois_ds, f_opt = \
    pois_attack.run(x=validation_set.X, y=validation_set.Y)

poisoned_clf = clf.deepcopy()
poisoned_clf.fit(training_set.append(pois_ds))

#%% cell

fig = CFigure(width=width, height=height)
fig.sp.plot_decision_regions(
    pois_attack._poisoned_clf, plot_background=True,
    n_grid_points=500, grid_limits=[(-20, 20), (-20, 20)])
fig.sp.grid(grid_on=True, linestyle=':', linewidth=0.5)
plot_data(fig, training_set)
# fig.sp.plot_ds(pois_ds, markers='*', edgecolors='k')
fig.sp.scatter(pois_ds.X[pois_ds.Y == 0, 0].ravel(),
               pois_ds.X[pois_ds.Y == 0, 1].ravel(),
               s=100, c='b', marker='*', edgecolors='w', linewidths=0.9)
fig.sp.scatter(pois_ds.X[pois_ds.Y == 1, 0].ravel(),
               pois_ds.X[pois_ds.Y == 1, 1].ravel(),
               s=100, c='r', marker='*', edgecolors='k', linewidths=0.9)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect('equal')
fig.savefig("dr_pois.pdf")
fig.close()


fig = CFigure(width=width, height=height)
fig.sp.plot_decision_regions(
    pois_attack._poisoned_clf, plot_background=True,
    n_grid_points=500, grid_limits=[(-20, 20), (-20, 20)])
fig.sp.grid(grid_on=True, linestyle=':', linewidth=0.5)
plot_data(fig, training_set)
fig.sp.set_axisbelow(True)
fig.sp._sp.set_aspect('equal')
fig.savefig("dr_pois2.pdf")
fig.close()
