# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../../")) 

project = 'Project X'
copyright = '2025, Alvaro Espinoza Chonlon, Marta Torres Sanchez, Maria de las Mercedes Ramos Santana, Ruben Agustin Gonzalez'
author = 'Alvaro Espinoza Chonlon, Marta Torres Sanchez, Maria de las Mercedes Ramos Santana, Ruben Agustin Gonzalez'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # docstrings estilo Google / NumPy
    "sphinx.ext.autosummary",       # genera tablas de resumen
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

# Mock optional third-party packages so autodoc can import the project without
# requiring the full ML stack during docs builds (e.g., on CI).
autodoc_mock_imports = [
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "seaborn",
    "matplotlib",
    "wordcloud",
    "nltk",
    "unidecode",
    "torch",
    "torchvision",
    "transformers",
    "regex",
    "sentence_transformers",
    "gensim",
    "TextVectorRepresentation",
    "nn_pytorch",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True

typehints_fully_qualified = True
typehints_document_rtype = True

# Make compilation strict with references:
nitpicky = True

nitpick_ignore = [
    ("py:class", "pd.DataFrame"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "torch.nn.Module"),
    ("py:class", "torch.Tensor"),
    ("py:class", "model"),
    ("py:class", "nn.Module"),
    ("py:class", "WordCloud"),
    ("py:class", "array-like"),
    ("py:class", "sparse matrix"),
]

# Inter-sphinx (references to external documentation)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "shibuya"

html_static_path = ["_static"]

html_favicon = "_static/logo_html.png"

# Ignore warnings about mocked imports so doctree builds stay clean when using
# autodoc_mock_imports to avoid heavy ML dependencies during docs generation.
suppress_warnings = ["autodoc.mocked_object"]