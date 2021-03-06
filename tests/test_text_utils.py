from unittest import TestCase

from sklearn.feature_extraction.text import CountVectorizer

from copycat_detector import create_two_gram_vocabulary, pre_process_data_file, get_answer_and_source, \
    calculate_containment_value, get_longest_common_subsequence
from helpers import process_file


class TestTextUtils(TestCase):
    def test_create_two_gram_vocabulary(self):
        raw_documents = ["This is an answer text", "This is a source text"]

        output = create_two_gram_vocabulary(raw_documents=raw_documents)
        expected_vocabulary = set(["this is", "is an", "an answer", "answer text", "is source", "source text"])

        self.assertEqual(expected_vocabulary, output.keys())

    def test_calculate_containment_value(self):
        csv_file = '../data/file_information.csv'
        plagiarism_df = pre_process_data_file(csv_file=csv_file)

        target_file = 'g0pA_taskb.txt'
        answer_index, source_index = get_answer_and_source(plagiarism_df, target_file=target_file)

        self.assertEqual(target_file, plagiarism_df.at[answer_index, 'File'])

        answer_file = '../data/' + target_file
        with open(answer_file, 'r', encoding='utf-8', errors='ignore') as file:
            answer_string = process_file(file)

        source_file = plagiarism_df.at[source_index, 'File']
        source_file = '../data/' + source_file
        with open(source_file, 'r', encoding='utf-8', errors='ignore') as file:
            source_string = process_file(file)

        ngram_size = 1
        count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(ngram_size, ngram_size))
        ngram_array = count_vectorizer.fit_transform(raw_documents=[answer_string, source_string]).toarray()

        containment_value = calculate_containment_value(ngram_array)
        self.assertAlmostEqual(1.0, containment_value)

    def test_get_longest_common_subsequence(self):
        answer_text = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached " \
                      "to each element of a hyperlinked set of documents"
        source_text = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical " \
                      "weighting to each element of a hyperlinked set of documents"

        raw_lcs = get_longest_common_subsequence(answer_text=answer_text, source_text=source_text, normalized=False)
        self.assertAlmostEqual(20, raw_lcs)

        normalized_lcs = get_longest_common_subsequence(answer_text=answer_text, source_text=source_text,
                                                        normalized=True)
        self.assertAlmostEqual(0.7408, normalized_lcs, places=3)
