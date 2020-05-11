from sklearn.feature_extraction.text import CountVectorizer


def create_two_gram_vocabulary(raw_documents):
    count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    count_vectorizer.fit(raw_documents=raw_documents)

    return count_vectorizer.vocabulary_


def calculate_containment_value(ngram_array, answer_index=0, source_index=1):
    answer_representation = ngram_array[answer_index]
    source_representation = ngram_array[source_index]

    common_ngrams = 0.0
    answer_ngrams = 0.0
    for ngram_index in range(len(answer_representation)):
        if answer_representation[ngram_index] == 1:
            answer_ngrams += 1

            if source_representation[ngram_index] == 1:
                common_ngrams += 1

    return common_ngrams / answer_ngrams
