import os
import pandas as pd
import ast
from music21 import chord, roman, key

base_path = r'C:\Users\andro\Desktop\8.semestar\prezentacijski seminar\ageron-handson-ml2-1e21152\ageron-handson-ml2-1e21152\datasets\jsb_chorales'

def load_and_analyze_split(folder_name):
    path = os.path.join(base_path, folder_name)
    all_data = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            all_data.append(pd.read_csv(os.path.join(path, file)))
    
    df = pd.concat(all_data, ignore_index=True)

    def analyze_chord(row):
        try:
            notes = ast.literal_eval(row['target_chord'])
            c = chord.Chord(notes)
            k = key.Key(row['original_key'], row['mode'])
            rn = roman.romanNumeralFromChord(c, k)
            # Vraćamo commonName i stupanj
            return c.commonName, rn.figure
        except:
            return "Unknown", "Unknown"

    df[['common_name', 'degree']] = df.apply(lambda row: pd.Series(analyze_chord(row)), axis=1)
    df['combined'] = df['degree'] + " (" + df['common_name'] + ")"
    return df

# 1. Učitavanje
train_df = load_and_analyze_split('train_features')

# 2. Razdvajanje i statistika
# ... (ostatak koda iznad je isti)

for m in ['major', 'minor']:
    subset = train_df[train_df['mode'] == m]
    
    print(f"\n" + "="*60)
    print(f" STATISTIKA ZA: {m.upper()} ".center(60, "="))
    print("="*60)
    
    # Izračun postotaka
    stats = subset['combined'].value_counts(normalize=True).head(10) * 100
    
    # POPRAVAK: Formatiranje pomoću map funkcije prije ispisa
    print(f"\nNajčešće harmonijske funkcije u {m}u:")
    print(stats.map("{:,.2f}%".format).to_string())
    
    # Udio septakorda
    sept_share = subset['common_name'].str.contains('Seventh').mean() * 100
    print(f"\nUdio septakorda u {m}u: {sept_share:.2f}%")