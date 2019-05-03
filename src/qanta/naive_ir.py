from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import pickle
import json
from os import path
import numpy as np

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset


MODEL_PATH = 'naive_ir.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.1


def guess_and_buzz(model, question_text, evidence) -> Tuple[str, bool]:
    guesses = model.guess([question_text], [evidence], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    #print('INSIDE guess and buzz - ', guesses[0][0], buzz)
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions, evidences) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, evidences, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs


class NaiveIRGuesser:
    def __init__(self):
        # self.tfidf_vectorizer = None
        # self.tfidf_matrix = None
        self.i_to_ans = None

    def ir_based_answering(self, sent_evidences):
        guesses_dic = {}
        sent_idx = {}
        i = 1
        for l in sent_evidences:
            for d in l:
                page = d['page']
                score = d['score']
                #multiplier = 1.0
                if i<=10:
                    multiplier = np.exp(i/25)
                else:
                    multiplier = np.exp(0.1)
                if page in guesses_dic.keys():
                    if i not in sent_idx[page]:
                        guesses_dic[page].append(multiplier*score)
                        sent_idx[page].append(i)
                else:
                    guesses_dic[page] = [multiplier*score]
                    sent_idx[page] = [i]
            i+=1
        guesses = []
        for x in guesses_dic.keys():
            l = guesses_dic[x]
            #guesses.append((x, sum(l)/len(l)))
            guesses.append((x, len(l), sum(l)))
        guesses.sort(key = lambda x:x[1]*x[2], reverse=True)
        guesses = list(map(lambda x:(x[0], x[2]), guesses))
        return guesses


    #This NAIVE IR BASED GUESSER does NOT require any training, so the below function is just a DUMMY in order to work with the established pipeline.
    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        # self.tfidf_vectorizer = TfidfVectorizer(
        #     ngram_range=(1, 3), min_df=2, max_df=.9
        # ).fit(x_array)
        # self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], evidences: List[List[List[Dict]]], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:

        guesses = [self.ir_based_answering(evidence)[:max_n_guesses] for evidence in evidences]

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans#,
                # 'tfidf_vectorizer': self.tfidf_vectorizer,
                # 'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = NaiveIRGuesser()
            # guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            # guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


def create_app(enable_batch=True):
    naive_ir_guesser = NaiveIRGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        evidence = request.json['wiki_paragraphs']    #because guessing here involves the evidence files.
        guess, buzz = guess_and_buzz(naive_ir_guesser, question, evidence)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = []
        evidences = []  #because guessing here involves the evidence files.
        for q in request.json['questions']:
            questions.append(q['text'])
            evidences.append(q['wiki_paragraphs'])
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(naive_ir_guesser, questions, evidences)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    While this Naive IR model does not require any training, we call it any way to maintain consistency with the pipeline,
    requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    naive_ir_guesser = NaiveIRGuesser()
    naive_ir_guesser.train(dataset.training_data()) 
    naive_ir_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
