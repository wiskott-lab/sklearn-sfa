.. -*- mode: rst -*-

sklearn-sfa - An implementation of Slow Feature Analysis compatible with scikit-learn
=====================================================================================

.. _scikit-learn: https://scikit-learn.org

.. _documentation: https://sklearn-sfa.readthedocs.io/en/latest/index.html

**sklearn-sfa** or **sksfa** is an implementation of Slow Feature Analysis for scikit-learn_.

It is meant as a standalone transformer for dimensionality reduction or as a building block
for more complex representation learning pipelines utilizing scikit-learn's extensive collection
of machine learning methods.

The package contains a solver for linear SFA and some auxiliary functions. The documentation_ 
provides an explanation of the algorithm, different use-cases, as well as pointers how to 
fully utilize SFA's potential, e.g., by employing non-linear basis functions or more sophisticated 
architectures.

Installation 
------------

The package can be installed via *pip*:

.. code-block:: bash

  pip install --user sklearn-sfa
  

Basic usage
-----------

In Python 3.6+, the package can then be imported as 

.. code-block:: python

  import sksfa 
  
The package comes with an SFA transformer. Below you see an example of initializing a transformer that
extracts 2-dimensional features:

.. code-block:: python

  sfa_transformer = sksfa.SFA(n_components=2)
  
The transformer implements sklearn's typical interface by providing ``fit``, ``fit_transform``, and ``transform`` methods.
