# =========================================================
# CHORD PREDICTION WITH AUTOREGRESSIVE CONTEXT
# + VITERBI SMOOTHING
# + HARMONIC ANALYSIS
# =========================================================

import os
import random
import itertools
import numpy as np
import pandas as pd

from collections import defaultdict
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from xgboost import XGBClassifier

import AdvancedHarmonicAnalyzer as AHA


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# DATA LOADING
# =========================================================

def loadSongs(root="dataset"):

    songs = []

    for mode in ["major", "minor"]:

        modePath = os.path.join(root, mode)

        if not os.path.exists(modePath):
            continue

        for file in os.listdir(modePath):

            if not file.endswith(".csv"):
                continue

            path = os.path.join(modePath, file)

            try:

                df = pd.read_csv(path)

                if len(df) < 2:
                    continue

                df["song_id"] = file
                df["is_minor"] = 1 if mode == "minor" else 0

                songs.append(df)

            except Exception as e:

                print(f"Greška {file}: {e}")

    print(f"Loaded {len(songs)} songs")

    return songs


# =========================================================
# FEATURE BUILDING
# =========================================================

def buildXY(
    songs,
    encoder=None,
    le=None
):

    data = pd.concat(
        [s.copy() for s in songs],
        ignore_index=True
    )

    catCols = [
        c
        for c in data.columns
        if "prevChord" in c
    ]

    numCols = [

        c
        for c in data.columns

        if c not in catCols
        and c not in [
            "targetChord",
            "song_id"
        ]
    ]

    #
    # categorical
    #

    if encoder is None:

        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        Xcat = encoder.fit_transform(
            data[catCols]
        )

    else:

        Xcat = encoder.transform(
            data[catCols]
        )

    #
    # numerical
    #

    Xnum = data[numCols].reset_index(
        drop=True
    )

    X = pd.concat([

        Xnum,
        pd.DataFrame(Xcat)

    ], axis=1)

    #
    # target
    #

    if le is None:

        le = LabelEncoder()

        y = le.fit_transform(
            data["targetChord"]
        )

    else:

        y = le.transform(
            data["targetChord"]
        )

    return (

        X,
        y,
        encoder,
        le,

        data["song_id"].values,

        data["is_minor"].values
    )


# =========================================================
# TRANSITION MATRIX
# =========================================================

def buildTransition(
    y,
    songIds,
    alpha=0.1
):

    n = len(np.unique(y))

    trans = np.zeros((n, n))

    startCounts = np.zeros(n)

    previousSong = None

    for i in range(len(y)):

        currentSong = songIds[i]

        if currentSong != previousSong:

            startCounts[y[i]] += 1

            previousSong = currentSong

            continue

        prevChord = y[i - 1]
        nextChord = y[i]

        trans[prevChord, nextChord] += 1

    #
    # Laplace smoothing
    #

    trans += alpha

    trans /= trans.sum(
        axis=1,
        keepdims=True
    )

    start = (
        startCounts + alpha
    ) / np.sum(startCounts + alpha)

    return trans, start


# =========================================================
# VITERBI
# =========================================================

def viterbi(
    probs,
    trans,
    start
):

    T, S = probs.shape

    dp = np.zeros((T, S))

    ptr = np.zeros(
        (T, S),
        dtype=int
    )

    #
    # initialization
    #

    dp[0] = (
        np.log(start + 1e-9)
        +
        np.log(probs[0] + 1e-9)
    )

    #
    # forward
    #

    for t in range(1, T):

        for j in range(S):

            scores = (

                dp[t - 1]
                +
                np.log(trans[:, j] + 1e-9)

            )

            ptr[t, j] = np.argmax(scores)

            dp[t, j] = (

                np.max(scores)
                +
                np.log(probs[t, j] + 1e-9)

            )

    #
    # backtracking
    #

    out = np.zeros(T, dtype=int)

    out[-1] = np.argmax(dp[-1])

    for t in range(T - 2, -1, -1):

        out[t] = ptr[
            t + 1,
            out[t + 1]
        ]

    return out


# =========================================================
# AUTOREGRESSIVE PREDICTION
# =========================================================

def autoregressiveProbabilities(
    model,
    song,
    encoder,
    le
):

    song = song.copy().reset_index(
        drop=True
    )

    generated = []

    #
    # seed
    #

    generated.append(
        song.loc[0, "targetChord"]
    )

    generated.append(
        song.loc[1, "targetChord"]
    )

    allProbs = []

    #
    # first two:
    # teacher forcing
    #

    for t in [0, 1]:

        rowDf = pd.DataFrame([
            song.loc[t]
        ])

        Xrow, _, _, _, _, _ = buildXY(
            [rowDf],
            encoder,
            le
        )

        probs = model.predict_proba(Xrow)[0]

        allProbs.append(probs)

    #
    # rollout
    #

    for t in range(2, len(song)):

        #
        # overwrite harmonic context
        #

        if "prevChord1" in song.columns:

            song.loc[
                t,
                "prevChord1"
            ] = generated[-2]

        if "prevChord2" in song.columns:

            song.loc[
                t,
                "prevChord2"
            ] = generated[-1]

        rowDf = pd.DataFrame([
            song.loc[t]
        ])

        Xrow, _, _, _, _, _ = buildXY(
            [rowDf],
            encoder,
            le
        )

        probs = model.predict_proba(Xrow)[0]

        pred = np.argmax(probs)

        predChord = le.inverse_transform(
            [pred]
        )[0]

        generated.append(predChord)

        allProbs.append(probs)

    return np.array(allProbs)


# =========================================================
# MUSICAL ERROR ANALYSIS
# =========================================================

def chordLoss(
    yTrue,
    yPred,
    le,
    analyzers,
    isMinors
):

    total = 0

    count = 0

    inv = {

        i: c
        for i, c in enumerate(le.classes_)
    }

    for t, p, k in zip(
        yTrue,
        yPred,
        isMinors
    ):

        if t == p:
            continue

        trueChord = inv[t]
        predChord = inv[p]

        analyzer = (
            analyzers["minor"]
            if k
            else analyzers["major"]
        )

        try:

            d = analyzer.predictionDistance(
                predChord,
                trueChord
            )

            total += d

            count += 1

        except:

            count += 1

    return total / max(count, 1)


def jaccardDistance(
    yTrue,
    yPred
):

    setTrue = set(yTrue)

    setPred = set(yPred)

    intersection = len(
        setTrue.intersection(setPred)
    )

    union = len(
        setTrue.union(setPred)
    )

    if union == 0:
        return 0

    similarity = intersection / union

    return 1 - similarity


def countUnlikelyTransitions(
    seq,
    trans,
    threshold=0.01
):

    count = 0

    for i in range(1, len(seq)):

        if trans[
            seq[i - 1],
            seq[i]
        ] < threshold:

            count += 1

    return count


# =========================================================
# TRAINING
# =========================================================

def trainModel(
    X,
    y,
    params
):

    weights = compute_sample_weight(
        "balanced",
        y
    )

    model = XGBClassifier(

        objective="multi:softprob",

        eval_metric="mlogloss",

        n_estimators=500,

        random_state=SEED,

        **params
    )

    model.fit(
        X,
        y,
        sample_weight=weights
    )

    return model


# =========================================================
# VITERBI CORRECTION EXTRACTION
# =========================================================

def extractCorrectionExamples(
    yTrue,
    predRaw,
    predVit,
    le
):

    inv = {

        i: c
        for i, c in enumerate(le.classes_)
    }

    corrections = []

    for i in range(len(yTrue)):

        rawCorrect = (
            predRaw[i] == yTrue[i]
        )

        vitCorrect = (
            predVit[i] == yTrue[i]
        )

        #
        # Viterbi fixed mistake
        #

        if (
            not rawCorrect
            and vitCorrect
        ):

            corrections.append({

                "position": i,

                "true": inv[yTrue[i]],

                "raw": inv[predRaw[i]],

                "viterbi": inv[predVit[i]]
            })

    return corrections


# =========================================================
# RAW VS VITERBI
# =========================================================

def compareRawVsViterbi(
    model,
    songs,
    encoder,
    le,
    trans,
    start,
    analyzers
):

    rows = []

    bestExamples = []

    for song in songs:

        X, y, _, _, _, isMinor = buildXY(
            [song],
            encoder,
            le
        )

        #
        # autoregressive probabilities
        #

        probs = autoregressiveProbabilities(

            model,
            song,
            encoder,
            le
        )

        #
        # RAW
        #

        predRaw = np.argmax(
            probs,
            axis=1
        )

        #
        # VITERBI
        #

        predVit = viterbi(
            probs,
            trans,
            start
        )

        #
        # metrics
        #

        row = {

            "song_id":
            song["song_id"].iloc[0],

            "length":
            len(y),

            #
            # accuracy
            #

            "raw_acc":
            accuracy_score(y, predRaw),

            "vit_acc":
            accuracy_score(y, predVit),

            #
            # f1
            #

            "raw_f1":
            f1_score(
                y,
                predRaw,
                average="macro",
                zero_division=0
            ),

            "vit_f1":
            f1_score(
                y,
                predVit,
                average="macro",
                zero_division=0
            ),

            #
            # harmonic loss
            #

            "raw_harm_loss":
            chordLoss(
                y,
                predRaw,
                le,
                analyzers,
                isMinor
            ),

            "vit_harm_loss":
            chordLoss(
                y,
                predVit,
                le,
                analyzers,
                isMinor
            ),

            #
            # jaccard
            #

            "raw_jaccard":
            jaccardDistance(
                y,
                predRaw
            ),

            "vit_jaccard":
            jaccardDistance(
                y,
                predVit
            ),

            #
            # smoothing
            #

            "raw_unlikely":
            countUnlikelyTransitions(
                predRaw,
                trans
            ),

            "vit_unlikely":
            countUnlikelyTransitions(
                predVit,
                trans
            )
        }

        #
        # gains
        #

        row["acc_gain"] = (
            row["vit_acc"]
            -
            row["raw_acc"]
        )

        row["f1_gain"] = (
            row["vit_f1"]
            -
            row["raw_f1"]
        )

        row["harm_gain"] = (
            row["raw_harm_loss"]
            -
            row["vit_harm_loss"]
        )

        row["jaccard_gain"] = (
            row["raw_jaccard"]
            -
            row["vit_jaccard"]
        )

        row["smooth_gain"] = (
            row["raw_unlikely"]
            -
            row["vit_unlikely"]
        )

        rows.append(row)

        #
        # correction examples
        #

        corrections = extractCorrectionExamples(
            y,
            predRaw,
            predVit,
            le
        )

        bestExamples.append({

            "song_id":
            song["song_id"].iloc[0],

            "num_corrections":
            len(corrections),

            "corrections":
            corrections,

            "true_sequence": [

                le.inverse_transform([x])[0]
                for x in y
            ],

            "raw_sequence": [

                le.inverse_transform([x])[0]
                for x in predRaw
            ],

            "viterbi_sequence": [

                le.inverse_transform([x])[0]
                for x in predVit
            ]
        })

    return (
        pd.DataFrame(rows),
        bestExamples
    )


# =========================================================
# SUMMARY
# =========================================================

def printComparisonSummary(df):

    print("\n" + "=" * 70)

    print("RAW VS VITERBI")

    print("=" * 70)

    print("\n--- AVERAGES ---\n")

    print(f"RAW ACC:       {df['raw_acc'].mean():.4f}")
    print(f"VIT ACC:       {df['vit_acc'].mean():.4f}")

    print(f"RAW F1:        {df['raw_f1'].mean():.4f}")
    print(f"VIT F1:        {df['vit_f1'].mean():.4f}")

    print(f"RAW HARM LOSS: {df['raw_harm_loss'].mean():.4f}")
    print(f"VIT HARM LOSS: {df['vit_harm_loss'].mean():.4f}")

    print(f"RAW JACCARD:   {df['raw_jaccard'].mean():.4f}")
    print(f"VIT JACCARD:   {df['vit_jaccard'].mean():.4f}")

    print(f"RAW UNLIKELY:  {df['raw_unlikely'].mean():.2f}")
    print(f"VIT UNLIKELY:  {df['vit_unlikely'].mean():.2f}")


# =========================================================
# FULL EVALUATION
# =========================================================

def fullEvaluation(
    model,
    songs,
    encoder,
    le,
    trans,
    start,
    analyzers
):

    comparison, examples = compareRawVsViterbi(

        model,
        songs,
        encoder,
        le,
        trans,
        start,
        analyzers
    )

    allTrue = []

    allRaw = []

    allVit = []

    for song in songs:

        X, y, _, _, _, _ = buildXY(
            [song],
            encoder,
            le
        )

        probs = autoregressiveProbabilities(
            model,
            song,
            encoder,
            le
        )

        predRaw = np.argmax(
            probs,
            axis=1
        )

        predVit = viterbi(
            probs,
            trans,
            start
        )

        allTrue.extend(y)

        allRaw.extend(predRaw)

        allVit.extend(predVit)

    return (

        comparison,

        np.array(allTrue),

        np.array(allRaw),

        np.array(allVit),

        examples
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CHORD PREDICTION")
    print("=" * 70)

    #
    # analyzers
    #

    analyzers = {

        "major":
        AHA.analyzeMode(
            "major",
            "major"
        ),

        "minor":
        AHA.analyzeMode(
            "minor",
            "minor"
        )
    }

    #
    # songs
    #

    songs = loadSongs(".")

    random.shuffle(songs)

    #
    # split
    #

    n = len(songs)

    trainSongs = songs[:int(0.7 * n)]

    valSongs = songs[
        int(0.7 * n):
        int(0.85 * n)
    ]

    testSongs = songs[
        int(0.85 * n):
    ]

    print(f"\nTrain: {len(trainSongs)}")
    print(f"Val:   {len(valSongs)}")
    print(f"Test:  {len(testSongs)}")

    #
    # encoder
    #

    _, _, encoder, le, _, _ = buildXY(
        trainSongs
    )

    #
    # training
    #

    Xtrain, yTrain, _, _, trainIds, _ = buildXY(
        trainSongs,
        encoder,
        le
    )

    Xtrain, yTrain, trainIds = shuffle(

        Xtrain,
        yTrain,
        trainIds,

        random_state=SEED
    )

    trans, start = buildTransition(
        yTrain,
        trainIds
    )

    params = {

        "max_depth": 8,

        "learning_rate": 0.05,

        "subsample": 0.8,

        "colsample_bytree": 0.8
    }

    model = trainModel(
        Xtrain,
        yTrain,
        params
    )

    #
    # evaluation
    #

    comparison, allTrue, allRaw, allVit, examples = fullEvaluation(

        model,

        testSongs,

        encoder,

        le,

        trans,

        start,

        analyzers
    )

    #
    # summary
    #

    printComparisonSummary(
        comparison
    )

    #
    # top examples
    #

    print("\n" + "=" * 70)
    print("TOP VITERBI CORRECTIONS")
    print("=" * 70)

    examples = sorted(

        examples,

        key=lambda x: x["num_corrections"],

        reverse=True
    )

    for ex in examples[:5]:

        print(f"\nSONG: {ex['song_id']}")

        print(
            f"Corrections: "
            f"{ex['num_corrections']}"
        )

        print("\nTRUE:")
        print(ex["true_sequence"])

        print("\nRAW:")
        print(ex["raw_sequence"])

        print("\nVITERBI:")
        print(ex["viterbi_sequence"])

        print("\nFIXES:")

        for c in ex["corrections"][:10]:

            print(

                f"pos {c['position']}: "

                f"{c['raw']} -> "

                f"{c['viterbi']} "

                f"(true={c['true']})"
            )

    #
    # save
    #

    comparison.to_csv(
        "evaluation_results.csv",
        index=False
    )

    print(
        "\nSaved evaluation_results.csv"
    )

    return (
        model,
        comparison,
        examples
    )


if __name__ == "__main__":

    model, comparison, examples = main()