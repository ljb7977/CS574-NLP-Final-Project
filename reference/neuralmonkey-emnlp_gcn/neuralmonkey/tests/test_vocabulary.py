#!/usr/bin/env python3.5

import unittest

from neuralmonkey.vocabulary import Vocabulary

CORPUS = [
    "the colorless ideas slept furiously",
    "pooh slept all night",
    "working class hero is something to be",
    "I am the working class walrus",
    "walrus for president"
]

TOKENIZED_CORPUS = [s.split(" ") for s in CORPUS]

VOCABULARY = Vocabulary()

for s in TOKENIZED_CORPUS:
    VOCABULARY.add_tokenized_text(s)


class TestVocabulary(unittest.TestCase):

    def test_all_words_in(self):
        for sentence in TOKENIZED_CORPUS:
            for word in sentence:
                self.assertTrue(word in VOCABULARY)

    def test_unknown_word(self):
        self.assertFalse("jindrisek" in VOCABULARY)

    def test_padding(self):
        pass

    def test_weights(self):
        pass

    def test_there_and_back_self(self):
        vectors, _ = VOCABULARY.sentences_to_tensor(TOKENIZED_CORPUS, 20,
                                                    add_start_symbol=True,
                                                    add_end_symbol=True)
        senteces_again = VOCABULARY.vectors_to_sentences(vectors[1:])

        for orig_sentence, reconstructed_sentence in \
                zip(TOKENIZED_CORPUS, senteces_again):
            self.assertSequenceEqual(orig_sentence, reconstructed_sentence)

    def test_min_freq(self):

        vocabulary = Vocabulary()

        for sentence in TOKENIZED_CORPUS:
            vocabulary.add_tokenized_text(sentence)

        vocabulary.truncate_by_min_freq(2)

        self.assertTrue("walrus" in vocabulary)
        self.assertFalse("colorless" in vocabulary)


if __name__ == "__main__":
    unittest.main()
