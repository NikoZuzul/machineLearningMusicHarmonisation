import os
import glob

import numpy as np
import pandas as pd

from collections import defaultdict

from scipy.stats import entropy
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn.metrics import mutual_info_score

from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt


class AdvancedHarmonicAnalyzer:

    def __init__(self, datasetPath, modeName):

        self.datasetPath = datasetPath
        self.modeName = modeName
        self.incomingCounts = defaultdict(
        lambda: defaultdict(int)
        )
        self.sequences = []

        self.chords = []
        self.chordToIndex = {}
        self.indexToChord = {}

        self.bigramCounts = defaultdict(
            lambda: defaultdict(int)
        )

        self.trigramCounts = defaultdict(
            lambda: defaultdict(int)
        )

        self.transitionMatrix = None

        self.contextVectors = {}

        self.distanceMatrix = None
    def buildIncomingCounts(self):

        for seq in self.sequences:

            for i in range(1, len(seq)):

                prevEvent = seq[i - 1]
                currentEvent = seq[i]

                prevChord = prevEvent["chord"]
                currentChord = currentEvent["chord"]

                weight = (
                    currentEvent["beatStrength"]
                    * currentEvent["duration"]
                )

                self.incomingCounts[
                    currentChord
                ][
                    prevChord
                ] += weight
    def loadSequences(self):

        csvFiles = glob.glob(
            os.path.join(self.datasetPath, "*.csv")
        )

        for file in csvFiles:

            df = pd.read_csv(file)

            songSequence = []

            for _, row in df.iterrows():

                event = {
                "chord": row["targetChord"],
                "beatStrength": row["note3_beatStrength"],
                "duration": row["note3_duration"]
                }

                songSequence.append(event)

            self.sequences.append(songSequence)

    def buildChordVocabulary(self):

        allChords = set()

        for seq in self.sequences:

            for event in seq:

                chord = event["chord"]

                allChords.add(chord)

        self.chords = sorted(list(allChords))

        self.chordToIndex = {
            chord: i
            for i, chord in enumerate(self.chords)
        }

        self.indexToChord = {
            i: chord
            for chord, i in self.chordToIndex.items()
        }

    def buildBigramCounts(self):

        for seq in self.sequences:

            for i in range(len(seq) - 1):

                currentEvent = seq[i]
                nextEvent = seq[i + 1]

                currentChord = currentEvent["chord"]
                nextChord = nextEvent["chord"]

                weight = (
                    currentEvent["beatStrength"]
                    * currentEvent["duration"]
                )

                self.bigramCounts[
                    currentChord
                ][
                    nextChord
                ] += weight

    

    def buildTransitionMatrix(self):

        n = len(self.chords)

        matrix = np.zeros((n, n))

        for chord in self.chords:

            rowSum = sum(
                self.bigramCounts[chord].values()
            )

            if rowSum == 0:
                continue

            i = self.chordToIndex[chord]

            for nextChord in self.bigramCounts[chord]:

                j = self.chordToIndex[nextChord]

                matrix[i, j] = (
                    self.bigramCounts[chord][nextChord]
                    / rowSum
                )

        self.transitionMatrix = matrix

    def buildContextVectors(self):

        for chord in self.chords:

            incomingVector = []
            outgoingVector = []

            #
            # INCOMING
            #

            incomingCounts = self.incomingCounts[chord]

            incomingTotal = sum(
                incomingCounts.values()
            )

            for prevChord in self.chords:

                if incomingTotal == 0:

                    prob = 0

                else:

                    prob = (
                        incomingCounts[prevChord]
                        / incomingTotal
                    )

                incomingVector.append(prob)

            #
            # OUTGOING
            #

            outgoingCounts = self.bigramCounts[chord]

            outgoingTotal = sum(
                outgoingCounts.values()
            )

            for nextChord in self.chords:

                if outgoingTotal == 0:

                    prob = 0

                else:

                    prob = (
                        outgoingCounts[nextChord]
                        / outgoingTotal
                    )

                outgoingVector.append(prob)

            #
            # FINAL BIDIRECTIONAL VECTOR
            #

            fullVector = np.concatenate([
                incomingVector,
                outgoingVector
            ])

            self.contextVectors[chord] = fullVector

    def jensenShannon(self, p, q):

        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        p = p + 1e-12
        q = q + 1e-12

        p = p / np.sum(p)
        q = q / np.sum(q)

        m = 0.5 * (p + q)

        return 0.5 * (
            entropy(p, m)
            + entropy(q, m)
        )

    def buildDistanceMatrix(self, metric=cosine):

        n = len(self.chords)

        distances = np.zeros((n, n))

        for i in range(n):

            for j in range(n):

                chordA = self.indexToChord[i]
                chordB = self.indexToChord[j]

                p = self.contextVectors[chordA]
                q = self.contextVectors[chordB]

                d = self.jensenShannon(p, q)

                distances[i, j] = d

        self.distanceMatrix = distances
    def computePredictionInformation(self,labels):

        chordToCluster = {}

        for chord, label in zip(
            self.chords,
            labels
        ):
            chordToCluster[chord] = label

        currentClusters = []
        nextChords = []

        for seq in self.sequences:

            for i in range(len(seq) - 1):

                currentChord = seq[i]["chord"]
                nextChord = seq[i + 1]["chord"]

                currentClusters.append(
                    chordToCluster[currentChord]
                )

                nextChords.append(nextChord)

        return mutual_info_score(
            currentClusters,
            nextChords
        )    

    def hierarchicalClustering(self):

        condensed = squareform(self.distanceMatrix)

        linkageMatrix = linkage(
            condensed,
            method="average"
        )

        plt.figure(figsize=(15, 8))

        dendrogram(
            linkageMatrix,
            labels=self.chords,
            leaf_rotation=90
        )

        plt.title(
            f"{self.modeName} harmonic hierarchy"
        )

        plt.tight_layout()
        plt.show()

        return linkageMatrix

    def evaluateClusterCounts(
        self,
        linkageMatrix,
        maxClusters=10
    ):

        results = []

        for k in range(2, maxClusters + 1):

            labels = fcluster(
                linkageMatrix,
                k,
                criterion="maxclust"
            )
            accuracy = self.computePredictionInformation(
            labels
            )

            

            clusterCounts = np.bincount(labels)

            probs = clusterCounts / np.sum(clusterCounts)

            complexity = entropy(probs, base=2)

            results.append({
                "clusters": k,
                "accuracy": accuracy,
                "complexity": complexity
            })

        return pd.DataFrame(results)

    def getClusters(self, linkageMatrix, k):

        labels = fcluster(
            linkageMatrix,
            k,
            criterion="maxclust"
        )

        clusters = defaultdict(list)

        for chord, label in zip(self.chords, labels):

            clusters[label].append(chord)

        return clusters

    def printTransitionMatrix(self):

        df = pd.DataFrame(
            self.transitionMatrix,
            index=self.chords,
            columns=self.chords
        )

        print(df.round(3))

    def printDistanceMatrix(self):

        df = pd.DataFrame(
            self.distanceMatrix,
            index=self.chords,
            columns=self.chords
        )

        print(df.round(3))

    def predictionDistance(
        self,
        predictedChord,
        actualChord
    ):

        i = self.chordToIndex[predictedChord]
        j = self.chordToIndex[actualChord]

        return self.distanceMatrix[i, j]

    def nearestChords(self, chord, topK=5):

        i = self.chordToIndex[chord]

        distances = []

        for j in range(len(self.chords)):

            if i == j:
                continue

            distances.append((
                self.indexToChord[j],
                self.distanceMatrix[i, j]
            ))

        distances.sort(key=lambda x: x[1])

        return distances[:topK]

    def plotDistanceMatrix(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.distanceMatrix, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(self.chords)), self.chords, rotation=90)
        plt.yticks(range(len(self.chords)), self.chords)
        plt.title(f"{self.modeName} distance matrix")
        plt.tight_layout()
        plt.show()

def analyzeMode(modePath, modeName):

    analyzer = AdvancedHarmonicAnalyzer(
        modePath,
        modeName
    )

    analyzer.loadSequences()
    

    analyzer.buildChordVocabulary()

    analyzer.buildBigramCounts()

    #analyzer.buildTrigramCounts()

    analyzer.buildTransitionMatrix()

    analyzer.buildContextVectors()

    analyzer.buildDistanceMatrix()
    analyzer.plotDistanceMatrix()

    print(f"\n===== {modeName.upper()} =====\n")

    print("\nTRANSITION MATRIX\n")
    analyzer.printTransitionMatrix()

    print("\nDISTANCE MATRIX\n")
    analyzer.printDistanceMatrix()

    linkageMatrix = analyzer.hierarchicalClustering()

    evalDf = analyzer.evaluateClusterCounts(
        linkageMatrix,
        maxClusters=8
    )

    print("\nCLUSTER EVALUATION\n")
    print(evalDf)

    evalDf["score"] = evalDf["accuracy"] - 0.1 * evalDf["complexity"]

    bestK = int(
    evalDf.sort_values("score", ascending=False)
    .iloc[0]["clusters"]
)

    print(f"\nBEST CLUSTER COUNT: {bestK}\n")

    clusters = analyzer.getClusters(
        linkageMatrix,
        bestK
    )

    print("\nHARMONIC FUNCTIONS\n")

    for clusterId in clusters:

        print(
            f"Function {clusterId}: "
            f"{clusters[clusterId]}"
        )
        print(f"\n===== {modeName.upper()} CLUSTERS =====\n")
        for clusterId, chords in clusters.items():
            print(f"Cluster {clusterId}: {', '.join(chords)}")

    return analyzer


if __name__ == "__main__":

    majorAnalyzer = analyzeMode(
        "major",
        "major"
    )

    minorAnalyzer = analyzeMode(
        "minor",
        "minor"
    )

    print("\n===== DISTANCE EXAMPLES =====\n")
    
    print(
        "ii_minor vs IV_major:",
        majorAnalyzer.predictionDistance(
            "2_minor",
            "4_major"
        )
    )

    print(
        "V_major vs I_major:",
        majorAnalyzer.predictionDistance(
            "5_major",
            "1_major"
        )
    )

    print("\n===== NEAREST CHORDS =====\n")

    nearest = majorAnalyzer.nearestChords(
        "1_major"
    )

    for chord, distance in nearest:

        print(chord, distance)