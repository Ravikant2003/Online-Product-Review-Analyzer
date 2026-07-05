import unittest

from text_sources import find_text_column, read_texts_from_csv


class TextSourcesTest(unittest.TestCase):
    def test_finds_xquik_tweet_text_column(self) -> None:
        column, texts = read_texts_from_csv('Tweet Text,Status\n"Great update",ok\n"",skip\n')

        self.assertEqual(column, "Tweet Text")
        self.assertEqual(texts, ["Great update"])

    def test_accepts_normalized_aliases(self) -> None:
        column, texts = read_texts_from_csv("tweet_text,score\nUseful   feature,1\nSecond row,0\n", limit=1)

        self.assertEqual(column, "tweet_text")
        self.assertEqual(texts, ["Useful feature"])

    def test_reports_missing_text_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "text column"):
            read_texts_from_csv("rating,value\n5,10\n")

    def test_finds_existing_alias(self) -> None:
        self.assertEqual(find_text_column(["id", "Comments", "score"]), "Comments")


if __name__ == "__main__":
    unittest.main()
