import psycopg2
import re
import tensorflow as tf
from tensorflow.python.platform import gfile
import os

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

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


    def vocab_matching(self):
        words = self.name_vocabulary()
        words = [tf.compat.as_bytes(line.strip()) for line in words]
        vocab = { x : y for (y, x) in enumerate(words)}
        return vocab, words


def tokenizer(sentence):
    words = []
    for fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(fragment))
    return [w for w in words if w]

def all_words(data_path='~/TrainDrinkNames/vocabulary.txt'):
    if not os.path.exists(data_path):
        reader = DrinkDataReader(DB_CONN)
        vocab = reader.vocab_matching()[-1]
        with gfile.GFile(data_path, mode='wb') as vocab_file:
            for w in vocab:
                vocab_file.write(w + 'b\n')
        return vocab



def main():
    reader = DrinkDataReader(DB_CONN)
    print (reader.vocab_matching())
    print(reader.all_drink_data())


if __name__ == '__main__':
    main()