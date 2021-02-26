import torch
import pickle
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np


def main():
    # Load data
    trainset = pickle.load(open("experiments/test.pkl", "rb"))

    # Scaler pickles
    scaler = pickle.load(open('./experiments/input_scaler.pkl', 'rb'))

    # Transform data
    X_train = trainset[:, 1:]
    id = trainset[:, [0]]
    X_train = torch.from_numpy(scaler.transform(X_train))
    # X_test = xscaler.transform(testset)

    # Load model
    model = torch.load(
            './experiments/weights/best_model.pkl',
            map_location=torch.device('cpu')
        )
    model.eval()

    # Descale for evaluation model
    inputs = X_train.unsqueeze(0).float()
    y_pred = model(inputs)
    y_pred = torch.round(y_pred).detach().numpy()
    frame = np.concatenate((id, y_pred), axis=1)

    frame = pd.DataFrame(frame, columns=['urlid', 'label'])
    frame['urlid'] = frame['urlid'].astype('int')
    frame['label'] = frame['label'].astype('int')
    frame.to_csv('experiments/submission.csv', index=False)

    # Evaluation
    # fpr, tpr, thresholds = roc_curve(y_pred, y_train)
    # loss = auc(fpr, tpr)
    # print(loss)


if __name__ == '__main__':
    main()
