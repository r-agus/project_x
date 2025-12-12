"""
Stopwords Management Module
===========================

This module centralizes the loading and customization of Spanish stopwords for
text preprocessing tasks across the project.

It provides:

- **Robust loading of NLTK Spanish stopwords**
  The function automatically downloads the required NLTK resource if it is not
  already available in the environment, ensuring smooth execution on any machine.

- **Extension of the base stopword list**
  A small collection of additional tokens commonly found in social-media text
  (e.g., “rt”, URLs, repeated laughter tokens such as “jaja”) is added to the
  standard Spanish stopwords to improve noise reduction during text cleaning.

- **Caching for efficiency**
  The stopword set is cached using ``functools.lru_cache`` so it is loaded only
  once, even if accessed repeatedly by other modules.

- **Convenient module-level constant**
  The variable ``stopwords`` exposes the final stopword set for easy import
  elsewhere in the project.

Overall, this module ensures a consistent, efficient, and domain-aware stopword
handling strategy for all text-processing components.
"""

import functools

import nltk  # INSTALAR
from nltk.corpus import stopwords as sw

_EXTRAS = {
    'rt', 'https', 'http',
    'jaja', 'jajaja', 'jajajaja', 'jajajajaja',
    'mas', 'hace', 'user', 'hashtag', 'politician', 'political_party'
}

def _load_spanish_stopwords() -> set[str]:
    """Return the base Spanish stopwords set, downloading resources if needed."""
    try:
        return set(sw.words('spanish'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        return set(sw.words('spanish'))


@functools.lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Return the reusable stopwords set (base + extras)."""
    return _load_spanish_stopwords().union(_EXTRAS)


# Keep a module-level constant for convenient imports, but populate lazily once.
stopwords = get_stopwords()
