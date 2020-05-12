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
        answer_count = answer_representation[ngram_index]
        source_count = source_representation[ngram_index]

        if answer_count > 0:
            answer_ngrams += answer_count

            if source_count > 0:
                common_ngrams += min(answer_count, source_count)

    return common_ngrams / answer_ngrams
