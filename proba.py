from music21 import corpus

chorales = corpus.search('bach')

choraleIds = []

for c in chorales:
    try:
        score = c.parse()
        
        # uzmi path iz metadata
        path = score.metadata.corpusFilePath
        
        if path and 'bwv' in path:
            name = path.split('/')[-1]
            
            if '.' in name:  # filtrira korale
                choraleIds.append(path)
                
    except:
        continue

print(len(choraleIds))
print(choraleIds[:10])