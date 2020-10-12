from nltk.stem.snowball import SnowballStemmer
import unicodedata
import re
from spacy.tokenizer import Tokenizer


SUPPORTED_LANGUAGE = ["en", "fr", "de", "cs", "es", "it", "pt"]

PUNCTUATION = [
    ";",
    ":",
    ",",
    "\.",
    '"',
    "'",
    "\?",
    "\(",
    "\)",
    "!",
    "？",
    "！",
    "；",
    "：",
    "。",
    "、",
    "《",
    "》",
    "，",
    "¿",
    "¡",
    "؟",
    "،",
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
        return tokens

    def init_resources(self):
        self.punctuation_pattern = re.compile("|".join(PUNCTUATION))
        self.stemmer = None
        if self.language_code == "en":
            from spacy.lang.en import English

            self.tokenizer = Tokenizer(English().vocab)
            self.stemmer = SnowballStemmer(language="english")
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/en/stopwords"
            )

        if self.language_code == "fr":
            from spacy.lang.fr import French

            self.tokenizer = Tokenizer(French().vocab)
            self.stemmer = SnowballStemmer(language="french")
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/fr/stopwords"
            )

        if self.language_code == "de":
            from spacy.lang.de import German

            self.tokenizer = Tokenizer(German().vocab)
            self.stemmer = SnowballStemmer(language="german")
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/de/stopwords"
            )

        if self.language_code == "it":
            from spacy.lang.it import Italian

            self.tokenizer = Tokenizer(Italian().vocab)
            self.stemmer = SnowballStemmer(language="italian")
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/it/stopwords"
            )

        if self.language_code == "cs":
            from spacy.lang.cs import Czech

            self.tokenizer = Tokenizer(Czech().vocab)
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/cs/stopwords"
            )

        if self.language_code == "pt":
            from spacy.lang.pt import Portuguese

            self.tokenizer = Tokenizer(Portuguese().vocab)
            self.stemmer = SnowballStemmer(language="portuguese")
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/pt/stopwords"
            )

        if self.language_code == "es":
            from spacy.lang.es import Spanish

            self.tokenizer = Tokenizer(Spanish().vocab)
            self.stemmer = SnowballStemmer(language="spanish")
            self.stop_words = self.load_stop_words(
                "assistant_dialog_skill_analysis/resources/es/stopwords"
            )

    def preprocess(self, sentence):
        sentence = sentence.lower()
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

        sentence = sentence.replace("'s", " is ")
        sentence = re.sub(self.punctuation_pattern, " ", sentence)
        return sentence

    def accent_removal(self, sentence):
        sentence = (
            unicodedata.normalize("NFD", sentence)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        return sentence
