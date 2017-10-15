import psycopg2
import re
import tensorflow as tf
from tensorflow.python.platform import gfile

_WORD_SPLIT = re.compile('([,!?\":;)(])')

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

_PAD_ID = 0
_GO_ID = 1
_EOS_ID = 2

DB_CONN = 'postgresql://localhost:5432/briansomes'


class DrinkDataReader(object):


    def __init__(self, connection):
        self.connection_string = connection


    def all_drink_names(self):
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute("SELECT name from drinks")
        names = [result[0] for result in cur]
        cur.close()
        conn.close()
        return names


    def all_drink_data(self):
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        drinkdata = dict()
        cur.execute("SELECT d.name, c.baseid FROM drinks d JOIN contains c on d.id = c.drinkid")
        for result in cur:
            if result[0] in drinkdata:
                drinkdata[result[0]].append(result[-1])
            else:
                drinkdata[result[0]] = [result[-1]]
        cur.close()
        conn.close()
        return drinkdata


    def name_vocabulary(self):
        words = []
        for name in self.all_drink_data().keys():
            words.extend(tokenizer(name))
        return list(set(words))


    def all_ingredients(self):
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute("SELECT id from baseingredients")
        ids = [id for id in cur]
        cur.close()
        conn.close()
        return ids

    def count_all_ingredients(self):
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(id) from baseingredients")
        count = [res[1] for res in cur]
        cur.close()
        conn.close()
        return count[0]


def vocab_matching(words):
    words = _START_VOCAB + [tf.compat.as_bytes(line.strip()) for line in words]
    vocab = { x : y for (y, x) in enumerate(words)}
    return vocab, words


def tokenizer(sentence):
    words = []
    for fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(fragment))
    return [w for w in words if w]


def all_words(data_path='../TrainDrinkNames/vocabulary.txt'):
    if not gfile.Exists(data_path):
        reader = DrinkDataReader(DB_CONN)
        vocab = vocab_matching(reader.name_vocabulary())[-1]
        with gfile.GFile(data_path, mode='wb') as vocab_file:
            for w in vocab:
                vocab_file.write(w + b'\n')
        return vocab
    return words_from_file(data_path)


def words_from_file(path):
    with gfile.GFile(path, mode='rb') as file:
        return [w.strip(b'\n') for w in file]


def name_to_ids(name, vocabulary):
    words = tokenizer(name)
    return [vocabulary.get(tf.compat.as_bytes(w)) for w in words]


def main():
    reader = DrinkDataReader(DB_CONN)
    data = reader.all_drink_data()
    print(data)
    vocabulary, words = vocab_matching(all_words())
    print(vocabulary)
    print([name_to_ids(name, vocabulary) for name in data.keys()])
    print([name for name in data.keys()])


if __name__ == '__main__':
    main()