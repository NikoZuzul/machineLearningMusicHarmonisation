from music21 import corpus, roman
from music21 import chord as chordModule

# 1. Učitavanje
chorale = corpus.parse('bach/bwv66.6')
key = chorale.analyze('key')

# 2. Chordify i obavezni .flatten() za brzinu i izbjegavanje warninga
chords = chorale.chordify().flatten()

print(f"ANALIZIRANI KLJUČ: {key}")
print("-" * 30)

for n in chorale.parts[0].recurse().notes:
    offset = n.offset
    
    # Tražimo akord na tom offsetu (specifično tražimo Chord objekt)
    c = chords.getElementAtOrBefore(offset, [chordModule.Chord])

    if c and isinstance(c, chordModule.Chord):
        # Analiza
        rn = roman.romanNumeralFromChord(c, key)
        
        # POPRAVAK: Koristimo romanNumeralAlone za stupanj (npr. 'I')
        # i figure za obrat (npr. '6', '64')
        stupanj = rn.romanNumeralAlone 
        kvaliteta = c.quality
        obrat = rn.figure if rn.figure else "osnovni"

# Opcija A: Samo naziv tona (n.pr. C4)
print(f"Ton: {str(n.pitch):3} | Stupanj: {stupanj:4} | Kvaliteta: {kvaliteta:10} | Obrat: {obrat}")