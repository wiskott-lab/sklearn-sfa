"""
This module contains the core implementation of linear SFA.
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils.validation import check_array
from scipy.sparse import issparse
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class SFA(TransformerMixin, BaseEstimator):
    """ Slow Feature Analysis (SFA)

    Linear dimensionality reduction and feature extraction method to be
    trained on time-series data. The data is decorrelated by whitening
    and linearly projected into the most slowly changing subspace.
    Slowness is measured by the average of squared one-step differences
    - thus, the most slowly changing subspacecorresponds to the directions
    with minimum variance for the dataset of one-step differences.
    It can be found using PCA.

    After training, the reduction can be applied to non-timeseries data as
    well. Read more in the :ref:`User Guide <SFA>`

    Parameters
    ----------
    n_components : int, float, None or str
        Number of components to keep.
        If n_components is not set all components are kept.

    batch_size : int or None
        Batches of the provided sequence that should be considered individual
        time-series. If batch_size is not set, the whole sequence will be
        considered connected.

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    svd_solver : str {'auto', 'full', 'arpack', 'randomized'}
        The solver used by the internal PCA transformers.
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'
        of the internal PCA transformers.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance, default=None
        Used when ``svd_solver`` == 'arpack' or 'randomized'. Pass an int
        for reproducible results across multiple function calls.

    robustness_cutoff : float, default=1e-15
        Applied after whitening. Features that have less explained
        variance in the input signal than robustness_cutoff will be
        considered trivial features and will be replaced in a way
        specified by fill_mode to ensure consistent output-dimensionality.

    fill_mode : str {'zero', 'slowest', 'fastest', 'noise'} or\
            None (default 'noise')
        The signal by which to replace output components that are
        artificial dimensions with almost no explained variance in
        input signal.
        If zero :
            Output components will be replace with constant zero signals.
            Subsequent applications of SFA will pick this up as a trivial
            signal again.
        If slowest :
            Output components will be replaced with copies of the slowest
            signals. Subsequent applications of SFA will pick this up as
            linearly dependent input dimension.
        If fastest :
            Output components will be replaced with copies of the fastest
            signals. Subsequent applications of SFA will pick this up as
            linearly dependent input dimension Since it is faster than
            any other signal, it would not be extracted in any case.
        If noise :
            Output components will be replaced with independent streams of
            Gaussian noise. The streams are typically not heavily correlated,
            but are very fast and thus will not be extracted.
        If fill_mode is specifically set to None :
            An exception will be thrown should trivial features are present
            in the input data.


    Attributes
    ----------
    input_dim_ : int
        The number of input features.

    delta_values_ : array, shape (n_components,)
        The estimated delta values (mean squared time-difference) of
        the different components.

    n_nontrivial_components_ : int
        The number of components that did not fall under the threshold
        defined by robustness_cutoff. Read a: effective dimension of
            input data.


    Examples
    --------
    >>> from sksfa import SFA
    >>> import numpy as np
    >>>
    >>> t = np.linspace(0, 8*np.pi, 1000).reshape(1000, 1)
    >>> t = t * np.arange(1, 6)
    >>>
    >>> ordered_cosines = np.cos(t)
    >>> mixed_cosines = np.dot(ordered_cosines, np.random.normal(0, 1, (5, 5)))
    >>>
    >>> sfa = SFA(n_components=2)
    >>> unmixed_cosines = sfa.fit_transform(mixed_cosines)
    """
    def __init__(self, n_components=None, *, batch_size=None, copy=True,
                 svd_solver="full", tol=.0, iterated_power="auto",
                 random_state=None, robustness_cutoff=1e-15,
                 fill_mode="noise"):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.copy = copy
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.batch_size = batch_size
        self.robustness_cutoff = robustness_cutoff
        self.fill_mode = fill_mode

    def _initialise_pca(self):
        """ Initialises internally used PCA transformers.
        """
        # initialize internal pca methods
        self.pca_whiten_ = PCA(svd_solver=self.svd_solver,
                               tol=self.tol, whiten=True)
        # initialize internal pca methods
        if self.batch_size is None:
            self.pca_diff_ = PCA(svd_solver=self.svd_solver, tol=self.tol)
        else:
            self.pca_diff_ = IncrementalPCA()

    def fit(self, X, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.

        y : None or {array-like}, shape (n_samples, 1)
            This does nothing and is only in here for compliance with
            sklearn API.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy, ensure_min_features=1)
        # check_estimators test expects feature warnings before sample warnings
        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy,
                        ensure_min_samples=10 if self.batch_size is None
                        else self.batch_size)
        self.input_dim_ = X.shape[1]
        if self.n_components is None:
            self.n_components_ = self.input_dim_
        else:
            self.n_components_ = self.n_components
        self._initialise_pca()
        if issparse(X):
            raise TypeError('SFA does not support sparse input.')
        if (X.shape[1] < 2):
            raise ValueError(f"At least two dimensional data is needed, \
                    n_features={X.shape[1]} is too small.")
        self._fit(X)
        self.is_fitted_ = True
        return self

    def _fit(self, X):
        """Fit the model to X either by using minor component extraction
        on the difference time-series of the whitened data or other
        (not yet implemented) method.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.
        """
        self._fit_standard_method(X)

    def _fit_standard_method(self, X):
        """ Fit the model to X either by first whitening the data, calculating
        the one-step differences and then extract their minor components.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples. If batch_size was set on init,
            this is assumed to be composed of concatenated time-series
            of batch_size length.
        """
        n_samples, _ = X.shape
        self.pca_whiten_.fit(X)

        X_whitened = self.pca_whiten_.transform(X)

        # Find non-trivial components
        input_evr = self.pca_whiten_.explained_variance_ratio_
        nontrivial_indices = np.argwhere(input_evr > self.robustness_cutoff)
        self.nontrivial_indices_ = nontrivial_indices.reshape((-1,))
        self.n_nontrivial_components_ = self.nontrivial_indices_.shape[0]
        n_trivial = self.n_components_ - self.n_nontrivial_components_
        if self.n_nontrivial_components_ == 0:
            raise ValueError(f"While whitening, only trivial components were \
                    found. This can be caused by passing 0-only input. ")
        if n_trivial > 0:
            if self.fill_mode is not None:
                warning_string = f"While whitening, {n_trivial} trivial \
                        components were found. Those are likely caused \
                        by a low effective dimension of the input data."
                if self.fill_mode == "zero":
                    warning_string += " Trivial components will be replaced \
                            by a 0 signal."
                if self.fill_mode == "slowest":
                    warning_string += " Trivial components will be replaced \
                            by copies of the slowest signal."
                if self.fill_mode == "fastest":
                    warning_string += " Trivial components will be replaced \
                            by copies of the fastest signal."
                if self.fill_mode == "noise":
                    warning_string += " Trivial components will be replaced \
                            by decorrelated white noise."
                warnings.warn(warning_string, RuntimeWarning)
            else:
                raise ValueError(f"While whitening, {n_trivial} trivial \
                        components were found. Those are likely caused \
                        by a low effective dimension of the input data. \
                        Set 'fill_mode' parameter to replace trivial \
                        features with blind signals.")
        X_whitened = X_whitened[:, self.nontrivial_indices_]

        X_diff = X_whitened[1:] - X_whitened[:-1]
        if self.batch_size is None:
            self.pca_diff_.fit(X_diff)
        else:
            n_batches = int(n_samples/self.batch_size)
            current_slowness = 0
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = (batch_idx + 1) * self.batch_size
                current_batch = X_whitened[batch_start:batch_end]
                batch_diff = current_batch[1:] - current_batch[:-1]
                self.pca_diff_.partial_fit(batch_diff)
        self._compute_delta_values()
        #self._compute_parameters()

    def _compute_delta_values(self):
        """ Computes the delta values, but in compliance with the method
            chosen for handling trivial components.
        """
        output_ev = self.pca_diff_.explained_variance_
        delta_values_ = output_ev[self.nontrivial_indices_][::-1]
        n_trivial = self.n_components_ - self.n_nontrivial_components_
        if n_trivial > 0:
            if self.fill_mode == "zero":
                delta_values_ = np.pad(delta_values_,
                                       (0, n_trivial), "constant",
                                       constant_values=np.nan)
            if self.fill_mode == "slowest":
                delta_values_ = np.pad(delta_values_,
                                       (n_trivial, 0),
                                       "constant",
                                       constant_values=delta_values_[0])
            if self.fill_mode == "fastest":
                delta_values_ = np.pad(delta_values_,
                                       (0, n_trivial),
                                       "constant",
                                       constant_values=delta_values_[-1])
        self.delta_values_ = delta_values_

    def transform(self, X):
        """ Use the trained model to apply dimensionality reduction to X.
        First, it is whitened using a trained PCA model. Afterwards, it
        is projected onto the previously extracted slow subspace using a
        second trained PCA model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The samples to be transformed.

        Returns
        -------
        y : ndarray, shape (n_samples, n_components)
            The slow features extracted from X.
        """
        X = check_array(X, dtype=[np.float64, np.float32],
                        ensure_2d=True, copy=self.copy)
        y = self.pca_whiten_.transform(X)
        y = y[:, self.nontrivial_indices_]
        y = self.pca_diff_.transform(y)
        n_missing_components = max(self.n_components_ - y.shape[1], 0)
        if n_missing_components > 0:
            if self.fill_mode == "zero":
                y = np.pad(y, ((0, 0), (n_missing_components, 0)))
            if self.fill_mode == "fastest":
                y = np.pad(y, ((0, 0), (n_missing_components, 0)), "edge")
            if self.fill_mode == "slowest":
                y = np.pad(y, ((0, 0), (0, n_missing_components)), "edge")
        y = y[:, -self.n_components_:][:, ::-1]
        return y

    def _compute_parameters(self):
        """ Collapse the parameters of the two linear PCA reductions into
        one set of mean and components for easier model inspection.
        TODO:
            make this deal with trivial components.
        """
        W_whiten = self.pca_whiten_.components_
        W_diff = self.pca_diff_.components_
        self.mean_ = self.pca_whiten_.mean_
        self.components_ = np.dot(np.dot(W_diff, np.diag(1/np.sqrt(self.pca_whiten_.explained_variance_))), W_whiten)[-self.n_components_:][::-1]

