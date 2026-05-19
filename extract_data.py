import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

inputFiles = ["train_data.csv", "val_data.csv", "test_data.csv"]

os.makedirs("major", exist_ok=True)
os.makedirs("minor", exist_ok=True)

songId = 0

stats = {
    "major": {
        "chords": Counter(),
        "transitions": defaultdict(Counter)
    },
    "minor": {
        "chords": Counter(),
        "transitions": defaultdict(Counter)
    }
}

defaultColumns = [
"note0_scaleDegree","note0_beatStrength","note0_duration","note0_beat","note0_measure",
"note1_scaleDegree","note1_beatStrength","note1_duration","note1_beat","note1_measure",
"note2_scaleDegree","note2_beatStrength","note2_duration","note2_beat","note2_measure",
"note3_scaleDegree","note3_beatStrength","note3_duration","note3_beat","note3_measure",
"prevChord0","prevChord1","prevChord2","targetChord"
]

def cleanLine(line):
    return line.strip().strip('"').strip("'")

def parseKey(line):
    return "major" if "major" in line else "minor"

def processSong(data, columns, keyType):
    global songId

    if len(data) < 3:
        return

    df = pd.DataFrame(data, columns=columns)

    if "targetChord" not in df.columns:
        return

    df = df.dropna(subset=["targetChord"])

    chords = [c for c in df["targetChord"].tolist() if isinstance(c, str) and c.strip() != ""]

    if len(chords) < 3:
        return

    for c in chords:
        stats[keyType]["chords"][c] += 1

    for a, b in zip(chords[:-1], chords[1:]):
        stats[keyType]["transitions"][a][b] += 1

    outPath = f"{keyType}/song_{songId}.csv"
    with open(outPath, "w") as out:
        out.write(",".join(columns) + "\n")
        for row in data:
            out.write(",".join(row) + "\n")

    print(f"Saved song_{songId} ({keyType}) with {len(chords)} chords")

    songId += 1


# =========================
# PARSING
# =========================

for file in inputFiles:
    with open(file, "r") as f:
        lines = f.readlines()

    i = 0
    currentColumns = None
    currentKey = "major"
    data = []

    while i < len(lines):
        rawLine = lines[i]
        line = cleanLine(rawLine)

        # KEY HEADER
        if "# key=" in line:
            if data:
                processSong(data, currentColumns or defaultColumns, currentKey)
                data = []

            currentKey = parseKey(line)

            nextLine = cleanLine(lines[i+1])
            currentColumns = nextLine.split(",")

            i += 2
            continue

        # COLUMN HEADER
        if "note0_scaleDegree" in line:
            if data:
                processSong(data, currentColumns or defaultColumns, currentKey)
                data = []

            currentColumns = line.split(",")
            i += 1
            continue

        # DATA
        if line != "":
            parts = line.split(",")
            cols = currentColumns or defaultColumns

            if len(parts) == len(cols):
                data.append(parts)

        i += 1

    if data:
        processSong(data, currentColumns or defaultColumns, currentKey)


# =========================
# HEATMAP FUNKCIJA
# =========================

def plotHeatmap(mode):
    chordCounter = stats[mode]["chords"]
    transitionCounter = stats[mode]["transitions"]

    chords = sorted(chordCounter.keys())

    matrix = pd.DataFrame(0.0, index=chords, columns=chords)

    for a in transitionCounter:
        total = sum(transitionCounter[a].values())
        if total == 0:
            continue

        for b in transitionCounter[a]:
            matrix.loc[a, b] = transitionCounter[a][b] / total

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix.values, aspect="auto")
    plt.title(f"Chord Transition Heatmap ({mode})")
    plt.xticks(range(len(chords)), chords, rotation=90)
    plt.yticks(range(len(chords)), chords)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# =========================
# STATISTIKA
# =========================

for mode in ["major", "minor"]:
    print("\n========================")
    print(f"STATISTIKA ZA {mode.upper()}")
    print("========================")

    chordCounter = stats[mode]["chords"]
    transitionCounter = stats[mode]["transitions"]

    print("Ukupno akorda:", sum(chordCounter.values()))

    print("\nFrekvencija akorda:")
    for chord, cnt in chordCounter.most_common():
        print(f"{chord}: {cnt}")

    print("\nTablica prijelaza:")

    allChords = sorted(chordCounter.keys())
    matrix = pd.DataFrame(0, index=allChords, columns=allChords)

    for a in transitionCounter:
        for b in transitionCounter[a]:
            matrix.loc[a, b] = transitionCounter[a][b]

    print(matrix)

    # 🔥 HEATMAP
    plotHeatmap(mode)