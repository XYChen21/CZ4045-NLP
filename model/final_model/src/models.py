import re
import string

from textblob import TextBlob
import nltk
import numpy as np
import joblib

# Data cleaning
import emot
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
from cleantext import clean
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

# nn
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# BERT huggingFace
from transformers import pipeline

# Model accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sb

import warnings

warnings.filterwarnings('ignore')

try:
    nltk.data.find('words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class BaseSentimentModel:
    name = ""

    def __init__(self):
        pass

    def predict(self, text: list):
        """
        Predict the result for text
        :params text: List of input text 
        :returns (list of labels, list of conf)
        """
        raise NotImplementedError


class TFIDF(BaseSentimentModel):
    name = "TF-IDF model with Logistic Regression"

    def __init__(self, model_path, text_transformer_path):
        super().__init__()
        self.model = joblib.load(model_path)
        self.text_transformer = joblib.load(text_transformer_path)

    def predict(self, text: list) -> (list, list):
        text = self.text_transformer.transform(text)
        pred_labels = self.model.predict(text)
        pred_confs = self.model.predict_proba(text)  # [:,1]

        conf = []
        for i in range(len(pred_confs)):
            conf.append(max(pred_confs[i][0], pred_confs[i][1]))

        return list(pred_labels), conf


class BERT(BaseSentimentModel):
    name = "BERT model using HuggingFace"

    def __init__(self):
        super().__init__()
        # load model from remote pretrained model on the huggingHub
        self.model = pipeline(model="gohbwj/sentiment-fine-tuned-yelp-2L", framework="pt")

    @staticmethod
    def __preprocess(text: list):

        temp = []
        for item in text:
            try:
                item = item.replace(r'\n', '')
                if len(item) > 512:
                    item = item[:512]
                temp.append(item)
            except:
                temp.append("")
                continue

        return temp

    def predict(self, text: list) -> (list, list):

        text = self.__preprocess(text)
        res = self.model(text)

        pred_labels = []
        conf = []
        for item in res:
            label = item["label"]
            score = item["score"]

            p = 0 if label == "LABEL_0" else 1

            pred_labels.append(p)
            conf.append(score)

        return pred_labels, conf


class NN_NO_POS(BaseSentimentModel):
    name = "Neural Network model without POS features"

    def __init__(self, model_path, tokenizer_path):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)

    @classmethod
    def seq_pad_and_trunc(self, sentences, tokenizer, padding, truncating, maxlen):
        """
        Generates an array of token sequences and pads them to the same length

        Args:
            sentences (list of string): list of sentences to tokenize and pad
            tokenizer (object): Tokenizer instance containing the word-index dictionary
            padding (string): type of padding to use
            truncating (string): type of truncating to use
            maxlen (int): maximum length of the token sequence

        Returns:
            pad_trunc_sequences (array of int): tokenized sentences padded to the same length
        """
        # Convert sentences to sequences
        sequences = tokenizer.texts_to_sequences(sentences)

        # Pad the sequences using the correct padding, truncating and maxlen
        pad_trunc_sequences = pad_sequences(sequences, padding=padding, truncating=truncating, maxlen=maxlen)

        return pad_trunc_sequences

    def predict(self, text: list) -> (list, list):
        MAXLEN = 256
        TRUNCATING = 'post'
        PADDING = 'post'

        seq = self.seq_pad_and_trunc(text, self.tokenizer, PADDING, TRUNCATING, MAXLEN)
        preds = self.model.predict(seq, verbose=0)
        conf = [p[0] if p[0] > p[1] else p[1] for p in preds]
        pre_labels = np.argmax(preds, 1)

        return list(pre_labels), conf


class NN_POS(NN_NO_POS):
    name = "Neural Network model with POS features"

    def __init__(self, model_path, tokenizer_path, pos_tokenizer_path):
        super().__init__(model_path, tokenizer_path)
        self.pos_tokenizer = joblib.load(pos_tokenizer_path)
        self.words = set(nltk.corpus.words.words())

    def __tokenize(self, text: str) -> str:

        tokenized_text = [word_tokenize(t) for t in sent_tokenize(text) if t not in self.words]

        # Pos tagging for each sentence in row['Tokenized']
        for i in range(len(tokenized_text)):
            # i refers to each of the tokenized sentence
            tokenized_text[i] = pos_tag(tokenized_text[i])

        return str(tokenized_text)

    @staticmethod
    def __process(token_col):
        """
        Process pos
        """
        stop_words = set(stopwords.words('english'))
        all_pairs = re.findall(r'\(\'.+?\', \'.+?\'\)', token_col)
        new_pairs = []
        final_pairs = []
        for pair in all_pairs:
            word_pos = pair.strip('()').split(', ')
            if '$' in word_pos[1]:
                word_pos[1] = word_pos[1].replace('$', 'S')

            new_pairs.append((word_pos[0][1:-1], word_pos[1][1:-1]))
        final_pairs = [pair for pair in new_pairs if pair[0].lower() not in stop_words and pair[1] != '\'\'"']
        text = ' '.join([pair[0].lower() for pair in final_pairs])
        text = str(''.join([t if ord(t) < 128 else ' ' for t in text]))
        pos = ' '.join([pair[1] for pair in final_pairs])
        return (text, pos)

    def predict(self, text: list) -> (list, list):
        MAXLEN = 256
        TRUNCATING = 'post'
        PADDING = 'post'

        text_list = []
        pos_list = []
        for item in text:
            t, p = self.__process(self.__tokenize(item))
            text_list.append(t)
            pos_list.append(p)

        seq = NN_NO_POS.seq_pad_and_trunc(text_list, self.tokenizer, PADDING, TRUNCATING, MAXLEN)
        seq_pos = NN_NO_POS.seq_pad_and_trunc(pos_list, self.pos_tokenizer, PADDING, TRUNCATING, MAXLEN)

        preds = self.model.predict((seq, seq_pos), verbose=0)
        conf = [p[0] if p[0] > p[1] else p[1] for p in preds]
        pre_labels = np.argmax(preds, 1)

        return list(pre_labels), conf


class Ensemble(BaseSentimentModel):
    name = "Stacked Ensemble model"

    def __init__(self, model_path: str, base_models: list):
        """
        Model order might matter, we use the following order for training:
        1. BERT
        2. TFIDF
        3. NN w/o pos
        4. NN w pos
        """
        super().__init__()
        assert len(base_models) == 4, "Must provide 4 models"

        self.model = joblib.load(model_path)
        self.base_models = base_models

    @staticmethod
    def __get_score(pred, conf) -> float:
        # score is the conf predicting 0
        if pred == 0:
            return conf
        else:
            return 1 - conf

    def predict(self, text: list) -> list:

        scores = []
        temp_confs = []
        for model in self.base_models:
            labels, confs = model.predict(text)  # [0, 1, 0, 2, 1], [0.9, 0.8, 0.9]

            temp_s = []
            # cal the score for each label-conf pair
            # score is the conf predicting 0
            for i in range(len(labels)):
                s = self.__get_score(labels[i], confs[i])
                temp_s.append(s)

            temp_confs.append(temp_s)

        for i, item in enumerate(text):
            score = []
            for j in range(4):
                score.append(temp_confs[j][i])

            scores.append(score)

        return list(self.model.predict(scores))


class GRP3Model(BaseSentimentModel):
    """
    Proposed model for CZ4045 Project
    """
    name = "CZ4045 GRP3 Sentiment Analysis Model with Stacked Ensemble"

    def __init__(self, models_dir):
        super().__init__()

        # init all the base models
        nn_pos = NN_POS(
            model_path=f"{models_dir}/nnpos.h5",
            tokenizer_path=f"{models_dir}/nnpos_tokenizer.sav",
            pos_tokenizer_path=f"{models_dir}/nnpos_pos_tokenizer.sav",
        )

        nn_no_pos = NN_NO_POS(
            model_path=f"{models_dir}/nn.h5",
            tokenizer_path=f"{models_dir}/nn_tokenizer.sav",
        )

        tfidf = TFIDF(
            model_path=f"{models_dir}/tfidf.sav",
            text_transformer_path=f"{models_dir}/tfidf_text_transformer.sav",
        )

        bert = BERT()

        self.base_models = [
            bert,
            tfidf,
            nn_no_pos,
            nn_pos
        ]

        self.model = Ensemble(
            model_path=f"{models_dir}/en_model.sav",
            base_models=self.base_models
        )

    @staticmethod
    def __get_subjectivity(text):
        subj = TextBlob(text).sentiment.subjectivity
        return subj

    @staticmethod
    def __preprocess(text: list):

        text_res = []

        words = set(nltk.corpus.words.words())
        lemmatizer = WordNetLemmatizer()

        # Cleaning Content with Emoji Removal, Lemmatizer and Non English words
        ## formatting
        all_emoji_emoticons = {**EMOTICONS_EMO, **UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS}
        all_emoji_emoticons = {k: v.replace(":", "").replace("_", " ").strip() for k, v in all_emoji_emoticons.items()}

        kp_all_emoji_emoticons = KeywordProcessor()
        for k, v in all_emoji_emoticons.items():
            kp_all_emoji_emoticons.add_keyword(k, v)

        for item in text:
            # remove punctuation
            item = item.translate(str.maketrans('', '', string.punctuation))

            # Lemmatize the words in sentence
            tokenized_text = word_tokenize(item)
            lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]

            """ 
            Removes non english words by:
            Joining English words w.lower() in words and joins with symbols/punctation --> w.alpha()
            Limitations :
            Removes some words:
            1. NER nouns (teriyaki chicken becomes chicken)
            2. Mispelled
            3. Split sort forms like can't , i've (i've become i 've )
            """
            # Remove Non English word in nltk.corpus
            cleaned_text = " ".join(w for w in lemmatized_text if w.lower() in words or not w.isalpha())

            res = ""
            temp_sentence = ""
            for i in cleaned_text:
                if ord(i) > 127:
                    temp_sentence += f" {i}"
                else:
                    temp_sentence += i
            # print(temp_sentence)
            cleaned_text = temp_sentence

            # Replacing emoji with words instead, done after because i want to retain the full text of emoji
            cleaned_text = kp_all_emoji_emoticons.replace_keywords(cleaned_text)
            cleaned_text = re.sub('â€™', '', cleaned_text)  # bug

            text_res.append(cleaned_text)

        return text_res

    def full_predict(self, text: list) -> list:
        """
        Predict using subjectivity and ensemble of 4 models
        :returns list of predictions
        0 -> Neg, 1 -> Pos, 2 -> Neu
        """

        y_labels = []
        for item in text:
            # First phase, subjectivity detection, 
            # Do not predict with ensemble if it's objective
            subj = self.__get_subjectivity(item)
            if subj == 0:
                y_labels.append(2)
                continue

            # Second phase, predict with stacked ensemble
            # todo: slower predicting one by one, maybe can bulk predict, lazy implement
            y_labels.append(self.model.predict([item])[0])

        return y_labels

    def __model_selection(self, model_class: str):
        """
        :param model_class: The model class, choices ["bert", "tfidf", "nn", "nn_pos"]
        """
        if model_class == "bert":
            model_index = 0
        elif model_class == "tfidf":
            model_index = 1
        elif model_class == "nn":
            model_index = 2
        else:
            model_index = 3

        return self.base_models[model_index]

    def predict(self, text: list, model_class: str = None) -> list:
        """
        Predict given a list of text
        :param model_class: The model class, choices ["bert", "tfidf", "nn", "nn_pos"]
        """

        # Preprocess phase, clean text, remove emoji etc.
        text = self.__preprocess(text)

        if not model_class:
            # run everything
            return self.full_predict(text)

        model = self.__model_selection(model_class)
        labels, confs = model.predict(text)
        return labels

    def model_accuracy(self, text: list, truth_labels: list, model_class: str = None):
        assert len(truth_labels) == len(text), f"Label size mismatch, must have {len(truth_labels)} ground truth labels"

        y_labels = self.predict(text=text, model_class=model_class)

        cm = confusion_matrix(truth_labels, y_labels)
        # False Postive = Actually False , Predict True --> Top right
        sb.heatmap( cm ,
                annot = True, fmt=".0f", annot_kws={"size": 18}
                , xticklabels=["Predict Negative" , "Predict Positive", "Predict Neutral"]
                , yticklabels=["Actual Negative" , "Actual Positive", "Actual Neutral"])


        print('Accuracy: %.3f' % accuracy_score(truth_labels, y_labels))
        print('Precision: %.3f' % precision_score(truth_labels, y_labels, average="weighted"))
        print('Recall: %.3f' % recall_score(truth_labels, y_labels, average="weighted"))
        print('F1: %.3f' % f1_score(truth_labels, y_labels, average="weighted"))
