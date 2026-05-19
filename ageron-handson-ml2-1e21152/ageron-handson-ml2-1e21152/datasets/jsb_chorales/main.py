import csv
from music21 import stream, note

csvFile = "C:\\Users\\andro\\Desktop\\8.semestar\\prezentacijski seminar\\ageron-handson-ml2-1e21152\\ageron-handson-ml2-1e21152\\datasets\\jsb_chorales\\train\\chorale_002.csv"

sopranoPart = stream.Part()
altoPart = stream.Part()
tenorPart = stream.Part()
bassPart = stream.Part()

with open(csvFile, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # preskoči prvi red (zaglavlje)
    for row in reader:
        notes = [int(p) for p in row if p != '']
        if len(notes) != 4:
            continue

        sopranoPart.append(note.Note(notes[0], quarterLength=1))
        altoPart.append(note.Note(notes[1], quarterLength=1))
        tenorPart.append(note.Note(notes[2], quarterLength=1))
        bassPart.append(note.Note(notes[3], quarterLength=1))

score = stream.Score([sopranoPart, altoPart, tenorPart, bassPart])
score.show('midi')