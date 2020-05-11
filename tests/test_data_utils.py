from unittest import TestCase

from copycat_detector import pre_process_data_file, get_answer_and_source


class TestDataUtils(TestCase):

    def test_pre_process_data_file(self):
        csv_file = '../data/file_information.csv'

        plagiarism_df = pre_process_data_file(csv_file=csv_file)
        expected_categories = set([-1, 0, 1, 2, 3])
        self.assertEqual(expected_categories, set(plagiarism_df['Category'].unique()))

        expected_classes = set([-1, 0, 1])
        self.assertEqual(expected_classes, set(plagiarism_df['Class'].unique()))

    def test_get_original_for_task(self):
        csv_file = '../data/file_information.csv'

        plagiarism_df = pre_process_data_file(csv_file=csv_file)
        target_file = 'g0pB_taske.txt'
        answer_index, source_index = get_answer_and_source(plagiarism_df, target_file=target_file)

        expected_answer_index = 9
        self.assertEqual(expected_answer_index, answer_index)

        source_row = plagiarism_df.iloc[source_index]
        expected_task = 'e'
        self.assertEqual(expected_task, source_row['Task'])
        expected_class = -1
        self.assertEqual(expected_class, source_row['Class'])
