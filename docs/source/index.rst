.. Project X documentation master file, created by
   sphinx-quickstart on Mon Dec  1 19:28:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Project X documentation
=======================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

Overview
--------

This site hosts the developer documentation for Project X.

API reference
-------------

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :caption: Public functions
   :nosignatures:

   load_data
   print_data_info
   analyze_class_distribution
   preserve_letters
   generate_wordcloud

.. toctree::
   :maxdepth: 2
   :caption: Additional guides

