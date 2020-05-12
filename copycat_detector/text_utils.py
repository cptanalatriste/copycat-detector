from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


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


def get_longest_common_subsequence(answer_text, source_text, normalized=True):
    answer_word_list = answer_text.split()
    source_word_list = source_text.split()

    rows = len(answer_word_list) + 1
    columns = len(source_word_list) + 1

    lcs_matrix = np.zeros(shape=(rows, columns))

    for row_index in range(1, rows):
        row_word = answer_word_list[row_index - 1]
        for column_index in range(1, columns):
            column_word = source_word_list[column_index - 1]

            if row_word == column_word:
                top_left_value = lcs_matrix[row_index - 1, column_index - 1]
                lcs_matrix[row_index, column_index] = top_left_value + 1
            else:
                left_value = lcs_matrix[row_index, column_index - 1]
                top_value = lcs_matrix[row_index - 1, column_index]
                lcs_matrix[row_index, column_index] = max(left_value, top_value)

    result = lcs_matrix[-1, -1]
    if normalized:
        result = result / len(answer_word_list)
    return result
