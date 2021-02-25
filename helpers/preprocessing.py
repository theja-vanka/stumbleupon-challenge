# Helper Libraries
import pandas as pd
import json
import numpy as np
import pickle


def loadglove():
    embeddings_dict = {}
    with open("data/glove.840B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def processcorpus(series):
    preprocessed_corpus = []
    for obj in series:
        res = json.loads(obj)
        _ = res['title'] + res['body'] + res['url']
        preprocessed_corpus.append(_.split())
    return preprocessed_corpus


def avgwordvec(model, corpus):
    avg_vectors = []
    for corpus in corpus:
        sent_vec = np.zeros(300)
        cnt_words = 0
        for word in corpus:
            if word in model.keys():
                vec = model[word]
                sent_vec += vec
                cnt_words += 1
        if cnt_words != 0:
            sent_vec /= cnt_words
        avg_vectors.append(sent_vec)


def main():
    df = pd.read_csv('./data/train.tsv', delimiter='\t')
    w2v_model = loadglove()

    preprocessed_corpus = processcorpus(df['boilerplate'].values)
    vectors = avgwordvec(w2v_model, preprocessed_corpus)

    pickle.dump(vectors, open("experiments/avgwv.pkl", "wb"))


if __name__ == '__main__':
    main()
