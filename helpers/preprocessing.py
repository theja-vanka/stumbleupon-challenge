# Helper Libraries
import pandas as pd
import json
import spacy


def main():
    nlp = spacy.load('en_core_web_trf')
    df = pd.read_csv('./data/train.tsv', delimiter='\t')
    boilerplate = df['boilerplate'].values
    for obj in boilerplate:
        res = json.loads(obj)
        _ = res['title'] + res['body'] + res['url']
        doc = nlp(_)
        tokens = [token.text for token in doc if not token.is_stop]
        print(tokens)


if __name__ == '__main__':
    main()
