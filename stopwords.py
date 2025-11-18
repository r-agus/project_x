from nltk.corpus import stopwords as sw

spanish_sw = set(sw.words('spanish'))

# Extras 
extras = {
    'rt', 'https', 'http',
    'jaja', 'jajaja', 'jajajaja', 'jajajajaja',
    'mas', 'hace'
}

stopwords = spanish_sw.union(extras)
