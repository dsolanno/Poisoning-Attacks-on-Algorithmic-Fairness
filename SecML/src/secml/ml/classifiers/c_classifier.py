"""
.. module:: CClassifier
   :synopsis: Interface and common functions for classification

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.ml import CModule
from secml.array import CArray
from secml.data import CDataset
from secml.ml.features import CPreProcess
from secml.utils.mixed_utils import check_is_fitted
from secml.core.exceptions import NotFittedError
from secml.core.decorators import deprecated


class CClassifier(CModule, metaclass=ABCMeta):
    """Abstract class that defines basic methods for Classifiers.

    A classifier assign a label (class) to new patterns using the
    informations learned from training set.

    This interface implements a set of generic methods for training
    and classification that can be used for every algorithms. However,
    all of them can be reimplemented if specific routines are needed.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    """
    __super__ = 'CClassifier'

    def __init__(self, preprocess=None):
        # List of classes on which training has been performed
        self._classes = None
        # Number of features of the training dataset
        self._n_features = None

        # TODO: CModule.__init__ should handle the call to create.
        preprocess = preprocess if preprocess is None \
            else CPreProcess.create(preprocess)
        CModule.__init__(self, preprocess=preprocess)

    @property
    def classes(self):
        """Return the list of classes on which training has been performed."""
        return self._classes

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        return self.classes.size if self.classes is not None else None

    @property
    def n_features(self):
        """Number of features (before preprocessing)."""
        return self._n_features

    def is_fitted(self):
        """Return True if the classifier is trained (fitted).

        Returns
        -------
        bool
            True or False depending on the result of the
            call to `check_is_fitted`.

        """
        try:
            self._check_is_fitted()
        except NotFittedError:
            return False
        return True

    def _check_is_fitted(self):
        """Check if the classifier is trained (fitted).

        Raises
        ------
        NotFittedError
            If the classifier is not fitted.

        """
        check_is_fitted(self, ['classes', 'n_features'])

    @abstractmethod
    def _fit(self, dataset):
        """Private method that trains the One-Vs-All classifier.
        Must be reimplemented by subclasses.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        raise NotImplementedError()

    def fit(self, dataset, n_jobs=1):
        """Trains the classifier.

        If a preprocess has been specified,
        input is normalized before training.

        For multiclass case see `.CClassifierMulticlass`.

        Parameters
        ----------
        dataset : CDataset
            Training set. Must be a :class:`.CDataset` instance with
            patterns data and corresponding labels.
        n_jobs : int
            Number of parallel workers to use for training the classifier.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        trained_cls : CClassifier
            Instance of the classifier trained using input dataset.

        """
        if not isinstance(dataset, CDataset):
            raise TypeError(
                "training set should be provided as a CDataset object.")

        # Storing dataset classes
        self._classes = dataset.classes
        self._n_features = dataset.num_features

        data_x = dataset.X
        # Transform data if a preprocess is defined
        if self.preprocess is not None:
            data_x = self.preprocess.fit_transform(dataset.X)

        # Data is ready: fit the classifier
        try:  # Try to use parallelization
            self._fit(CDataset(data_x, dataset.Y), n_jobs=n_jobs)
        except TypeError:  # Parallelization is probably not supported
            self._fit(CDataset(data_x, dataset.Y))

        return self

    def decision_function(self, x, y=None):
        """Computes the decision function for each pattern in x.

        If a preprocess has been specified, input is normalized
        before computing the decision function.

        .. note::

            The actual decision function should be implemented
            inside :meth:`_decision_function` method.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        y : int or None, optional
            The label of the class wrt the function should be calculated.
            If None, return the output for all classes.

        Returns
        -------
        score : CArray
            Value of the decision function for each test pattern.
            Dense flat array of shape (n_samples,) if y is not None,
            otherwise a (n_samples, n_classes) array.

        """
        scores = self.forward(x, caching=False)
        return scores if y is None else scores[:, y].ravel()

    def _check_clf_index(self, y):
        """Raise error if index y is outside [-1, n_classes) range.

        Parameters
        ----------
        y : int
            class label index.

        """
        if y < 0 or y >= self.n_classes:
            raise ValueError(
                "class label {:} is out of range".format(y))

    def grad_f_x(self, x, y):
        """Computes the gradient of the classifier's decision function wrt x.

        Parameters
        ----------
        x : CArray or None, optional
            The input point. The gradient will be computed at x.
        y : int
            Binary index of the class wrt the gradient must be computed.

        Returns
        -------
        gradient : CArray
            The gradient of the linear classifier's decision function
            wrt decision function input. Vector-like array.

        """
        self._check_clf_index(y)

        # check that x is a single point
        if CArray(x).is_vector_like is False:
            raise ValueError("Classifier gradient can be computed only on"
                             " a single input sample.")

        w = CArray.zeros(self.n_classes)
        w[y] = 1  # one-hot encoding of y
        return self.gradient(x, w)

    def predict(self, x, return_decision_function=False):
        """Perform classification of each pattern in x.

        If preprocess has been specified,
        input is normalized before classification.

        Parameters
        ----------
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the `decision_function` value along
            with predictions. Default False.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
            to each test pattern. The classification label is the label of
            the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
            score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        """
        scores = self.decision_function(x, y=None)

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1).ravel()

        return (labels, scores) if return_decision_function is True else labels

    def estimate_parameters(self, dataset, parameters, splitter, metric,
                            pick='first', perf_evaluator='xval', n_jobs=1):
        """Estimate parameter that give better result respect a chose metric.

        Parameters
        ----------
        dataset : CDataset
            Dataset to be used for evaluating parameters.
        parameters : dict
            Dictionary with each item as `{parameter: list of values to test}`.
            Example:
            `{'C': [1, 10, 100], 'gamma': list(10.0 ** CArray.arange(-4, 4))}`
        splitter : CDataSplitter or str
            Object to use for splitting the dataset into train and validation.
            A splitter type can be passed as string, in this case all
            default parameters will be used. For data splitters, num_folds
            is set to 3 by default.
            See CDataSplitter docs for more informations.
        metric : CMetric or str
            Object with the metric to use while evaluating the performance.
            A metric type can be passed as string, in this case all
            default parameters will be used.
            See CMetric docs for more informations.
        pick : {'first', 'last', 'random'}, optional
            Defines which of the best parameters set pick.
            Usually, 'first' correspond to the smallest parameters while
            'last' correspond to the biggest. The order is consistent
            to the parameters dict passed as input.
        perf_evaluator : CPerfEvaluator or str, optional
            Performance Evaluator to use. Default 'xval'.
        n_jobs : int, optional
            Number of parallel workers to use for performance evaluation.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        best_parameters : dict
            Dictionary of best parameters found through performance evaluation.

        """
        # Import here as is only needed if this function is called
        from secml.ml.peval import CPerfEvaluator

        # Initialize the evaluator
        perf_eval = CPerfEvaluator.create(perf_evaluator, splitter, metric)
        # Set verbosity level to be the same of classifier
        # Classifier verbosity will be set to 0 wile estimating params
        perf_eval.verbose = self.verbose

        # Evaluate the best parameters for the classifier (self)
        best_params = perf_eval.evaluate_params(
            self, dataset, parameters, pick=pick, n_jobs=n_jobs)[0]

        # Set the best parameters in classifier
        self.set_params(best_params)

        return best_params
