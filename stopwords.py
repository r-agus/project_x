import functools

import nltk  # INSTALAR
from nltk.corpus import stopwords as sw

_EXTRAS = {
    'rt', 'https', 'http',
    'jaja', 'jajaja', 'jajajaja', 'jajajajaja',
    'mas', 'hace',
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
