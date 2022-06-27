import os
import re
import sys
from nltk.stem.snowball import SnowballStemmer
from spacy.tokenizer import Tokenizer
import unicodedata
import assistant_skill_analysis


SUPPORTED_LANGUAGE = ["en", "fr", "de", "cs", "es", "it", "pt"]
PUNCTUATION = [
    "\\" + chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
]


class LanguageUtility:
    def __init__(self, language_code):
        if language_code not in SUPPORTED_LANGUAGE:
            raise Exception(
                "language_code %s not in supported languages %s",
                language_code,
                SUPPORTED_LANGUAGE,
            )
        self.language_code = language_code
        self.init_resources()

    def tokenize(self, sentences):
        tokens = list(map(lambda x: x.text, self.tokenizer(sentences)))
        if self.stemmer:
            tokens = [self.stemmer.stem(str(token)).strip() for token in tokens]
        tokens = [token for token in tokens if len(token) > 0]
        return tokens

    def init_resources(self):
        self.punctuation_pattern = re.compile("|".join(PUNCTUATION))
        self.stemmer = None
        stopwords_path = os.path.join(
            os.path.dirname(assistant_skill_analysis.__file__),
            "resources",
            self.language_code,
            "stopwords",
        )
        if self.language_code == "en":
            from spacy.lang.en import English

            self.tokenizer = Tokenizer(English().vocab)
            self.stemmer = SnowballStemmer(language="english")
            self.stop_words = self.load_stop_words(stopwords_path)

        elif self.language_code == "fr":
            from spacy.lang.fr import French

            self.tokenizer = Tokenizer(French().vocab)
            self.stemmer = SnowballStemmer(language="french")
            self.stop_words = self.load_stop_words(stopwords_path)

        elif self.language_code == "de":
            from spacy.lang.de import German

            self.tokenizer = Tokenizer(German().vocab)
            self.stemmer = SnowballStemmer(language="german")
            self.stop_words = self.load_stop_words(stopwords_path)

        elif self.language_code == "it":
            from spacy.lang.it import Italian

            self.tokenizer = Tokenizer(Italian().vocab)
            self.stemmer = SnowballStemmer(language="italian")
            self.stop_words = self.load_stop_words(stopwords_path)

        elif self.language_code == "cs":
            from spacy.lang.cs import Czech

            self.tokenizer = Tokenizer(Czech().vocab)
            self.stop_words = self.load_stop_words(stopwords_path)

        elif self.language_code == "pt":
            from spacy.lang.pt import Portuguese

            self.tokenizer = Tokenizer(Portuguese().vocab)
            self.stemmer = SnowballStemmer(language="portuguese")
            self.stop_words = self.load_stop_words(stopwords_path)

        elif self.language_code == "es":
            from spacy.lang.es import Spanish

            self.tokenizer = Tokenizer(Spanish().vocab)
            self.stemmer = SnowballStemmer(language="spanish")
            self.stop_words = self.load_stop_words(stopwords_path)
        else:
            raise Exception("language code %s is not supported", self.language_code)

    def preprocess(self, sentence):
        sentence = sentence.lower()
        sentence = self.contraction_normalization(sentence)
        sentence = self.strip_punctuations(sentence)
        if self.language_code in ["fr", "es", "cs", "es", "pt"]:
            sentence = self.accent_removal(sentence)
        return sentence

    def load_stop_words(self, path):
        stopwords = set()
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                stopwords.add(line)
        return stopwords

    def strip_punctuations(self, sentence):
        """
        function to strip punctuations from the utternace
        :param utterance:
        :return:
        """
        sentence = re.sub(self.punctuation_pattern, " ", sentence)
        return sentence

    def contraction_normalization(self, sentence):
        """
        common contraction normalization for english
        :param sentence:
        :return:
        """
        sentence = sentence.replace("'s", " is ")
        sentence = sentence.replace("n't", " not ")
        sentence = sentence.replace("'ll", " will ")
        sentence = sentence.replace("'m", " am ")
        return sentence

    def accent_removal(self, sentence):
        """origin from facebook research xlm preprocessing
        https://github.com/facebookresearch/XLM"""

        return "".join(
            [
                ch
                for ch in unicodedata.normalize("NFD", sentence)
                if unicodedata.category(ch) != "Mn"
            ]
        )
