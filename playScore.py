import csv
from music21 import chord, key as keyModule

def label_to_chord(label, key):
    try:
        degree, quality = label.split("_")
        degree = int(degree)
        
        scale = key.getScale()
        root = scale.pitchFromDegree(degree)
        root.octave = 3  # postavi oktavu na 3 za konzistentnost
        
        if quality == "major":
            return chord.Chord([root, root.transpose("M3"), root.transpose("P5")])
        elif quality == "minor":
            return chord.Chord([root, root.transpose("m3"), root.transpose("P5")])
        elif quality == "diminished": 
            return chord.Chord([root, root.transpose("m3"), root.transpose("d5")])
        elif quality == "augmented":
            return chord.Chord([root, root.transpose("M3"), root.transpose("A5")])
        
    except:
        return None
from music21 import stream, note, chord, key as keyModule

def create_midi(notes, predictedChords, key):
    
    melodyPart = stream.Part()
    chordPart = stream.Part()
    
    for i in range(len(predictedChords)):
        
        n = notes[i]
        
        # 🎵 melodija
        newNote = note.Note(n.pitch)
        newNote.duration = n.duration
        # newNote.beatStrength = n.beatStrength  # read-only
        melodyPart.append(newNote)
        
        # 🎼 akordi
        cLabel = predictedChords[i]
        c = label_to_chord(cLabel, key)
        
        if c:
            c.duration = n.duration
            chordPart.append(c)
    
    score = stream.Score()
    score.insert(0, melodyPart)
    score.insert(0, chordPart)
    
    return score
print("Starting playScore")
with open("dataset/bwv314.csv") as f:
    first_line = f.readline().strip()
    first_line = first_line.strip('"')  # remove quotes if present
    if first_line.startswith("#"):
        info = first_line[1:]  # ukloni #
        key_str, timeSig, beatDuration_str = [x.split("=")[1] for x in info.split(",")]
        beatDuration = float(beatDuration_str)
        tonic, mode = key_str.split()
        key = keyModule.Key(tonic, mode)
        reader = csv.reader(f)
        header = next(reader)  # preskoči header
        notes = []
        predictedChords = []
        row = next(reader)  # uzmi prvi red s podacima
        for i in range(3):
            
            scaleDegree = int(row[i *5])
            beatStrength = float(row[i *5 + 1])
            duration = float(row[i *5 + 2]) * beatDuration
            beat = float(row[i *5 + 3])
            measure = int(row[i *5 + 4])
            
            pitch = key.getScale().pitchFromDegree(scaleDegree)
            n = note.Note(pitch)
            n.quarterLength = duration
            # n.beatStrength = beatStrength  # read-only
            # n.measureNumber = measure  # read-only
            notes.append(n)
            predictedChords.append(row[20 + i])  # targetChord
        for row in reader:
            if len(row) < 20 or not row[15] or not row[17] or not row[19]:
                continue
            try:
                scaleDegree = int(row[15])
                duration_val = float(row[17]) * beatDuration
                measure_val = int(row[19])
                pitch = key.getScale().pitchFromDegree(scaleDegree)
                n3 = note.Note(pitch)
                n3.quarterLength = duration_val
                notes.append(n3)
                predictedChords.append(row[-1])
            except (ValueError, IndexError):
                continue
            
        score = create_midi(notes, predictedChords, key)

        score.write("midi", "output.mid")
        score.show("midi")  # odmah reproducira
        print("Done")    