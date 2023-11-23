from sklearn.feature_extraction.text import CountVectorizer
import warnings

from abc import ABCMeta, abstractmethod


import numpy as np
from scipy.special import logsumexp
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.utils import deprecated
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.utils.validation import _check_sample_weight
import math
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pickle

_ALPHA_MIN = 1e-10


class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape (n_samples, n_classes).

        predict, predict_proba, and predict_log_proba pass the input through
        _check_X and handle it over to _joint_log_likelihood.
        """

    @abstractmethod
    def _check_X(self, X):
        """To be overridden in subclasses with the actual checks.

        Only used in predict* methods.
        """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        # print(jll[0])
        # print(transform_to_sentiment_score(*jll[0]))
        data = []
        for i in range(len(jll)):
            transform_to_sentiment_score(*jll[i], data)
        print(data)
        # print(self.classes_)

        answers = self.classes_[np.argmax(jll, axis=1)]
        final = []
        for i in range(len(answers)):
            #result = [max(data[i][0]) - min(data[i]), answers[i]]
            result = [data[i][0] - data[i][1], answers[i]]

            final.append(result)

        return final

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.exp(self.predict_log_proba(X))


class _BaseDiscreteNB(_BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per _BaseNB
    _update_feature_log_prob(alpha)
    _count(X, Y)
    """

    @abstractmethod
    def _count(self, X, Y):
        """Update counts that are used to calculate probabilities.

        The counts make up a sufficient statistic extracted from the data.
        Accordingly, this method is called each time `fit` or `partial_fit`
        update the model. `class_count_` and `feature_count_` must be updated
        here along with any model specific counts.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Y : ndarray of shape (n_samples, n_classes)
            Binarized class labels.
        """

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        """Update feature log probabilities based on counts.

        This method is called each time `fit` or `partial_fit` update the
        model.

        Parameters
        ----------
        alpha : float
            smoothing parameter. See :meth:`_check_alpha`.
        """

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return self._validate_data(X, accept_sparse="csr", reset=False)

    def _check_X_y(self, X, y, reset=True):
        """Validate X and y in fit methods."""
        return self._validate_data(X, y, accept_sparse="csr", reset=reset)

    def _update_class_log_prior(self, class_prior=None):
        """Update class log priors.

        The class log priors are based on `class_prior`, class count or the
        number of classes. This method is called each time `fit` or
        `partial_fit` update the model.
        """
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(self.class_count_)

            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError(
                "Smoothing parameter alpha = %.1e. alpha should be > 0."
                % np.min(self.alpha)
            )
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.n_features_in_:
                raise ValueError(
                    "alpha should be a scalar or a numpy array with shape [n_features]"
                )
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha = %.1e"
                % _ALPHA_MIN
            )
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        first_call = not hasattr(self, "classes_")
        X, y = self._check_X_y(X, y, reset=first_call)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_classes = len(classes)
            self._init_counters(n_classes, n_features)

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently
        Y = Y.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def _more_tags(self):
        return {"poor_score": True}

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `n_features_` was deprecated in version 1.0 and will be "
        "removed in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_


class MultinomialNB(_BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models.

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    class_log_prior_ : ndarray of shape (n_classes,)
        Smoothed empirical log probability for each class.

    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier

    feature_count_ : ndarray of shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    n_features_ : int
        Number of features of each sample.

        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0"""

    def __init__(self, *, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        check_non_negative(X, "MultinomialNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_


def transform_to_sentiment_score(ln, lp, data=[]):
    # ln = i[0]
    # lp = i[1]
    mi = min(ln, lp)
    ln += abs(int(mi))
    lp += abs(int(mi))
    log_likelihood_negative = ln
    log_likelihood_positive = lp
    # Transform log likelihood to score using the sigmoid function
    score_negative = 1 / (1 + math.exp(-log_likelihood_negative))
    score_positive = 1 / (1 + math.exp(-log_likelihood_positive))

    data.append([score_negative, score_positive])
    # return (
    #     score_negative,
    #     score_positive,
    # )
    return data


stop_words = set(stopwords.words("english"))


def conv_score(score):
    if str(score) == "1":
        return "Negative"
    elif str(score) == "2":
        return "Negative"
    elif str(score) == "3":
        return "Negative"
    elif str(score) == "4":
        return "Positive"
    elif str(score) == "5":
        return "Positive"


def process_review(review):
    global stop_words

    # Tokenize Into A List
    tokenized_text = word_tokenize(review)
    filtered_sent = []

    # Removing Stop Words
    for w in tokenized_text:
        if w not in stop_words:
            filtered_sent.append(w)

    # #Stemming
    # ps = PorterStemmer()
    # stemmed_words=[]
    # for w in filtered_sent:
    #     stemmed_words.append(ps.stem(w))

    # Lemmatization
    lem = WordNetLemmatizer()
    filtered_sent_2 = []
    for i in filtered_sent:
        word = lem.lemmatize(i, "v")
        filtered_sent_2.append(word)

    final = []

    for i in filtered_sent_2:
        if i.isalpha() == True:
            final.append(i)

    return " ".join(final)


def clean_data(dataset):
    df = pd.read_csv("dataset2.csv")
    source_reviews = list(df["Text"])
    source_scores = list(df["Score"])
    reviews = []
    scores = []

    for i in source_reviews:
        review = process_review(i)
        reviews.append(review)

    for i in source_scores:
        score = conv_score(i)
        scores.append(score)

    modrev = []
    posno = 0
    modscore = []
    for i in range(len(reviews)):
        if scores[i] == "Positive":
            if posno < 729:
                posno += 1
                modrev.append(reviews[i])
                modscore.append(scores[i])
        else:
            modrev.append(reviews[i])
            modscore.append(scores[i])

    d = {}
    for i in modscore:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    #print(d)
    reviews = modrev
    scores = modscore
    return (reviews, scores)


def train_with_data(test_data, dataset="dataset2.csv"):
    try:
        reviews, scores = clean_data(dataset)
        X_test = test_data
        sample = []
        for i in X_test:
            sample.append(process_review(i))

        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(reviews)
        X_test_vec = vectorizer.transform(sample)
        clf = MultinomialNB()
        clf.fit(X_train_vec, scores)
        y_pred = clf.predict(X_test_vec)
        # print("Y predictions:")
        # print(y_pred)   
        score = 0

        for i in range(len(y_pred)):
            t_score = y_pred[i][0]
            score += t_score
        # for i in range(len(y_pred)):
        #     t_score = y_pred[i][0]
        #     result = y_pred[i][1]
        #     if result == "Positive":
        #         score += t_score
        #     else:
        #         score -= t_score

        return score / len(y_pred)

    except ValueError as e:
        # Handle the exception here, e.g., by returning an error message or taking appropriate action.
        return -1


def process_with_model(test_data, modelpath):
    
        with open(modelpath, "rb") as file:
            loaded_model = pickle.load(file)
            vectorizer = CountVectorizer()
            X_test_vec = vectorizer.transform(test_data)
            y_pred = loaded_model.predict(X_test_vec)

            score = 0

            for i in range(len(y_pred)):
                t_score = y_pred[i][0]
                result = y_pred[i][1]
                if result == "Positive":
                    score += t_score
                else:
                    score -= t_score

            return score / len(y_pred)

    # except Exception:
    #     print("Encountered exception, Error code 404")


def save_training_model(test_data, dataset="dataset2.csv"):

    reviews, scores = clean_data(dataset)
    X_test = test_data
    sample = []
    for i in X_test:
        sample.append(process_review(i))

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(reviews)
    X_test_vec = vectorizer.transform(sample)
    clf = MultinomialNB()
    clf.fit(X_train_vec, scores)
    y_pred = clf.predict(X_test_vec)
    # Save the model to a file
    filename = "sentimentclassificationmodel.pkl"
    with open(filename, "wb") as file:
        pickle.dump(clf, file)

    score = 0

    for i in range(len(y_pred)):
        t_score = y_pred[i][0]
        result = y_pred[i][1]
        if result == "Positive":
            score += t_score
        else:
            score -= t_score

    return score / len(y_pred)

   