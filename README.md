# Machine Learning Music Harmonisation

Projekt se bavi automatskom harmonizacijom melodija, tj. predviđanjem akorda za zadanu melodijsku liniju. Kao podloga koristi se JSB/Bach chorales dataset, a glazbeni zapis se pretvara u tablični skup značajki koje opisuju note, trajanja, poziciju u taktu, prethodne akorde i ciljni akord.

## Dataset

Dataset je podijeljen na `train_data.csv`, `val_data.csv` i `test_data.csv`. Svaki redak predstavlja jedan harmonijski trenutak s melodijskim i ritmičkim značajkama te oznakom `targetChord`. Skripta `extract_data.py` dodatno razdvaja pjesme prema tonalitetu u foldere `major/` i `minor/`, gdje se svaka pjesma sprema kao zaseban CSV. Matrice prijelaza između akorada spremaju se u `major_transition_matrix.csv` i `minor_transition_matrix.csv`.

## Metode

### Random Forest

Random Forest koristi više stabala odlučivanja za predikciju akorda. Dobar je za interpretaciju jer omogućuje analizu važnosti značajki. U ovom projektu pokazuje da su prethodni akordi, trajanje nota i ritmička pozicija među najvažnijim informacijama za harmonizaciju.

### Gradient Boosting / XGBoost

Gradient Boosting sekvencijalno gradi stabla tako da svako novo stablo ispravlja pogreške prethodnih. U projektu se koristi XGBoost s višeklasnom predikcijom (`multi:softprob`) jer osim najvjerojatnijeg akorda daje i vjerojatnosti svih klasa, što je korisno za Viterbijev algoritam.

### Viterbijev algoritam

Viterbijev algoritam ne bira samo lokalno najbolji akord, nego traži najvjerojatniji cijeli niz akorada. Kombinira vjerojatnosti iz XGBoost modela s matricom prijelaza između akorada. Time se dobiva glazbeno glađa progresija, iako lokalna accuracy metrika ne mora uvijek porasti.

### Hijerarhijsko klasteriranje

Hijerarhijsko klasteriranje koristi se za analizu funkcionalne sličnosti akorada. Svaki akord opisuje se kontekstom, odnosno akordima koji se pojavljuju prije i poslije njega. Udaljenost se računa Jensen-Shannonovom divergencijom, pa se akordi grupiraju prema harmonijskoj ulozi, a ne samo prema nazivu.

## Struktura datoteka

- `dataLoader.py` – obrađuje originalni JSB chorales dataset, transponira pjesme u C-dur ili a-mol i stvara značajke pomoću kliznog prozora.
- `extract_data.py` – parsira `train_data.csv`, `val_data.csv` i `test_data.csv`, razdvaja pjesme u `major/` i `minor/` te računa frekvencije i prijelaze akorada.
- `randomForest.py` – trenira Random Forest model, radi grid search i ispisuje metrike kao što su accuracy, macro F1, recall, function mismatch i Jaccardova udaljenost.
- `randomForest2.py` – proširena analiza s dodatnim značajkama, budućim notama, intervalima, Random Forestom i logističkom regresijom.
- `gradientBoostingViterbi.py` – glavna skripta za XGBoost model, autoregresivnu predikciju, Viterbijevo zaglađivanje i usporedbu RAW/Viterbi rezultata.
- `AdvancedHarmonicAnalyzer.py` – gradi matrice prijelaza, kontekstne vektore, harmonijske udaljenosti i hijerarhijske klastere akorada.
- `diagnostics.py` – pomoćne funkcije za crtanje grafova, provjeru prenaučenosti, learning rate analizu i usporedbu Viterbi/RAW rezultata.
- `proba.py`, `proba2.py`, `proba3.py` – pomoćne eksperimentalne skripte za rad s vjerojatnostima modela.
- `playScore.py` – skripta za reprodukciju ili rad s glazbenim zapisom.
- `catBoost.py` – rezervirana/eksperimentalna datoteka za CatBoost pristup.
- `major/` i `minor/` – obrađene pjesme odvojene prema tonalitetu.
- `train_data.csv`, `val_data.csv`, `test_data.csv` – početni skupovi podataka za treniranje, validaciju i testiranje.
- `evaluation_results.csv`, `evaluation_results2.csv` – spremljeni rezultati evaluacije modela.
- `major_transition_matrix.csv`, `minor_transition_matrix.csv` – prijelazne matrice akorada.
- `output.mid` – primjer generiranog ili obrađenog MIDI izlaza.

## Zaključak

Projekt kombinira lokalne klasifikacijske modele i sekvencijalno modeliranje. Random Forest i XGBoost predviđaju akorde iz značajki melodije i konteksta, dok Viterbijev algoritam poboljšava globalni harmonijski tok. Dodatna analiza pomoću informacijskih udaljenosti i klasteriranja omogućuje interpretaciju akorada kao funkcionalnih elemenata harmonijskog prostora.
