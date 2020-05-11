from unittest import TestCase
from copycat_detector import create_two_gram_vocabulary


class TestTextUtils(TestCase):
    def test_create_two_gram_vocabulary(self):
        raw_documents = ["This is an answer text", "This is a source text"]

        output = create_two_gram_vocabulary(raw_documents=raw_documents)
        expected_vocabulary = set(["this is", "is an", "an answer", "answer text", "is source", "source text"])

        self.assertEqual(expected_vocabulary, output.keys())
