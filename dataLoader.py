import pandas as pd
import os
from music21 import stream, chord, pitch, key, interval, note

def process_jsb_dataset_to_features(root_path):
    # JSB dataset obično ima mape 'train', 'test' i 'valid'
    folders = ['train', 'valid', 'test'] 
    rows_per_measure = 16  # 1 takt = 16 redaka (šesnaestinke u 4/4 mjeri)
    
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        output_folder = os.path.join(root_path, f"{folder}_features")
        os.makedirs(output_folder, exist_ok=True)
        
        if not os.path.exists(folder_path):
            print(f"Mapa {folder_path} ne postoji, preskačem...")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                
                # 1. Čitanje cijele pjesme i rukovanje nulama (0,0,0,0)
                midi_sequence = []
                for _, row in df.iterrows():
                    # Uzmi samo vrijednosti nota koje su veće od 0
                    notes = [int(n) for n in row.values if n > 0]
                    midi_sequence.append(notes) # Ako su sve 0, notes će biti prazna lista []
                
                # 2. Pretvaranje u music21 stream (samo za analizu i transpoziciju)
                s = stream.Stream()
                for notes in midi_sequence:
                    if notes:
                        # Ako imamo note, dodajemo akord
                        c = chord.Chord([pitch.Pitch(midi=p) for p in notes])
                        # Postavljamo fiksno trajanje da izbjegnemo greške pri transpoziciji
                        c.quarterLength = 0.25 
                        s.append(c)
                    else:
                        # Ako je redak [0,0,0,0], dodajemo pauzu (Rest)
                        # To čuva broj redaka i sinkronizaciju s dobama
                        r = note.Rest()
                        r.quarterLength = 0.25
                        s.append(r)
                
                # 3. Analiza tonaliteta i transpozicija
                try:
                    # Analiziramo tonalitet (ignorira pauze automatski)
                    song_key = s.analyze('key')
                    orig_key = song_key.tonic.name
                    mode = song_key.mode
                    
                    # Određivanje cilja: C dur ili a mol
                    target_key = key.Key('C') if mode == 'major' else key.Key('a')
                    trans_interval = interval.Interval(song_key.tonic, target_key.tonic)
                    s_trans = s.transpose(trans_interval)
                except Exception as e:
                    print(f"Greška pri analizi {filename}: {e}. Preskačem transpoziciju.")
                    s_trans = s
                    orig_key = "Unknown"
                    mode = "Unknown"

                # 4. Vraćanje transponiranih nota u listu (pretvaranje natrag u MIDI brojeve)
                transposed_midi = []
                for element in s_trans:
                    if isinstance(element, chord.Chord):
                        transposed_midi.append(sorted([p.midi for p in element.pitches]))
                    else:
                        # Pauze postaju prazne liste (ekvivalent onim nulama s početka)
                        transposed_midi.append([])

                # 5. Kreiranje "Sliding Window" dataseta sa značajkama
                dataset_rows = []
                
                # Krećemo od 3 jer trebamo povijest od 3 prethodna koraka
                for i in range(3, len(transposed_midi)):
                    dataset_rows.append({
                        'prev_3': str(transposed_midi[i-3]),
                        'prev_2': str(transposed_midi[i-2]),
                        'prev_1': str(transposed_midi[i-1]),
                        'original_key': orig_key,
                        'mode': mode,
                        'beat_position': i % rows_per_measure,
                        'target_chord': str(transposed_midi[i])
                    })

                # 6. Spremanje u novi CSV
                new_df = pd.DataFrame(dataset_rows)
                new_df.to_csv(os.path.join(output_folder, filename), index=False)
                print(f"Uspješno obrađeno: {folder}/{filename} ({orig_key} {mode})")

# Putanja do tvog dataseta
path = r'C:\Users\andro\Desktop\8.semestar\prezentacijski seminar\ageron-handson-ml2-1e21152\ageron-handson-ml2-1e21152\datasets\jsb_chorales'
process_jsb_dataset_to_features(path)