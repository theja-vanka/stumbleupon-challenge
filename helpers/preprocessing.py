# Helper Libraries
import pandas as pd
import json
import numpy as np
import pickle
from tqdm import tqdm


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
    for obj in tqdm(series):
        res = json.loads(obj)
        _ = ""
        if 'title' in res.keys() and res['title'] is not None:
            _ += str(res['title'])
        if 'body' in res.keys() and res['body'] is not None:
            _ += str(res['body'])
        if 'url' in res.keys() and res['url'] is not None:
            _ += res['url']
        preprocessed_corpus.append(_.split())
    return preprocessed_corpus


def avgwordvec(model, corpus):
    avg_vectors = []
    for corpus in tqdm(corpus):
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
    return avg_vectors


def createvector():
    df = pd.read_csv('./data/train.tsv', delimiter='\t')
    w2v_model = loadglove()

    preprocessed_corpus = processcorpus(df['boilerplate'].values)
    vectors = avgwordvec(w2v_model, preprocessed_corpus)

    pickle.dump(vectors, open("experiments/avgwv.pkl", "wb"))


def main():
    # createvector()
    df = pd.read_csv('./data/train.tsv', delimiter='\t')
    df['alchemy_category'] = df['alchemy_category'].astype('category')
    df['alchemy_category'] = df['alchemy_category'].cat.codes
    df = pd.get_dummies(df, columns=["alchemy_category"])
    y_vector = df[['label']].to_numpy()
    df = df.drop(['url', 'urlid', 'boilerplate', 'label'], axis=1)
    df = df.replace("?", 0)
    for col in df.columns:
        df[col] = df[col].astype(float)

    df = df.to_numpy()
    vectors = pickle.load(open("experiments/avgwv.pkl", "rb"))
    final_vector = np.concatenate((df, vectors), axis=1)
    final_vector = np.concatenate((final_vector, y_vector), axis=1)
    pickle.dump(vectors, open("experiments/dataset.pkl", "wb"))


if __name__ == '__main__':
    main()
