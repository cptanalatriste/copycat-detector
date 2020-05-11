from unittest import TestCase

from copycat_detector import pre_process_data_file


class TestDataUtils(TestCase):

    def test_pre_process_data_file(self):
        csv_file = '../data/file_information.csv'

        plagiarism_df = pre_process_data_file(csv_file=csv_file)
        expected_categories = set([-1, 0, 1, 2, 3])
        self.assertEqual(expected_categories, set(plagiarism_df['Category'].unique()))

        expected_classes = set([-1, 0, 1])
        self.assertEqual(expected_classes, set(plagiarism_df['Class'].unique()))
