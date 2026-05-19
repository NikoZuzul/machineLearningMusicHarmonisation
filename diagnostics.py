import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


def plotDepthAnalysis(
    Xtrain,
    ytrain,
    Xval,
    yval,
    modelBuilder,
    depths
):

    trainScores = []
    valScores = []

    for d in depths:

        model = modelBuilder(d)

        model.fit(Xtrain, ytrain)

        trainPred = model.predict(Xtrain)
        valPred = model.predict(Xval)

        trainScores.append(
            f1_score(ytrain, trainPred, average="macro")
        )

        valScores.append(
            f1_score(yval, valPred, average="macro")
        )

    plt.figure()

    plt.plot(depths, trainScores, label="train F1")
    plt.plot(depths, valScores, label="val F1")

    plt.xlabel("max_depth")
    plt.ylabel("F1 macro")

    plt.title("Overfitting check: depth vs performance")

    plt.legend()

    plt.show()


def plotLearningRateAnalysis(
    Xtrain,
    ytrain,
    Xval,
    yval,
    modelBuilder,
    lrs
):

    trainScores = []
    valScores = []

    for lr in lrs:

        model = modelBuilder(lr)

        model.fit(Xtrain, ytrain)

        trainPred = model.predict(Xtrain)
        valPred = model.predict(Xval)

        trainScores.append(
            f1_score(ytrain, trainPred, average="macro")
        )

        valScores.append(
            f1_score(yval, valPred, average="macro")
        )

    plt.figure()

    plt.plot(lrs, trainScores, label="train F1")
    plt.plot(lrs, valScores, label="val F1")

    plt.xlabel("learning_rate")
    plt.ylabel("F1 macro")

    plt.title("Learning rate sensitivity")

    plt.legend()

    plt.show()


def plotViterbiGain(
    rawF1,
    vitF1
):

    idx = np.arange(len(rawF1))

    plt.figure()

    plt.plot(idx, rawF1, label="RAW F1")
    plt.plot(idx, vitF1, label="Viterbi F1")

    plt.fill_between(
        idx,
        rawF1,
        vitF1,
        alpha=0.2
    )

    plt.title("Viterbi improvement per song")

    plt.legend()

    plt.show()


def plotLossComparison(
    rawLoss,
    vitLoss
):

    idx = np.arange(len(rawLoss))

    plt.figure()

    plt.plot(idx, rawLoss, label="RAW loss")
    plt.plot(idx, vitLoss, label="Viterbi loss")

    plt.title("Harmonic distance loss")

    plt.legend()

    plt.show()