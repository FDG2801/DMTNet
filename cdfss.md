# Introductions

Parla dei problemi principali della segmentazione, ovvero la mancanza di dati e per questo motivo introduce il FSS il cui obbiettivo è appunto aumentare la precisione con poche immagini di base (e le sue etichette).
Dice inoltre che solitamente l'FSS si usa con il meta learning (che prevede meta training e meta testing), dove il meta-training è un training con più meta tasks ed usa le classi base, ma il problema resta la differenza di dominio fra test e training set. Il loro approccio propone infatti una soluzione a questo trasformando le feature specifiche del dominio in feature domain-agnostic (indipendenti dal dominio) - *ma quindi tipo GRL?*

In pratica:
- proposto DMTNet (Doubly Matching Transformation-based Network) che usa un modulo chiamato Self-Matching Transformation (SMT) per trasformare le features dell'immagine di query in maniera adattiva, per ridurre l'overfitting
- introdotto DHC (Dual Hypercorrelation Construction) che apprende le correlazione tra l'immagine di query sia sul primo piano che sullo sfondo, così da segmentare per somiglianza
- ideato TSF (Test-time Self Finetuning) che tuna alcuni paramentri della rete durante il testing così da migliorare significativamente le previsioni sulle immagini di query.
- condotto esperimenti

# Figures

![alt text](image.png)
Architettura di DMTNet

![alt text](image-1.png)
Risultati sui dataset

![alt text](image-2.png)
Comportamento di DMTNet

![alt text](image-3.png)
Risultati in tabella

![alt text](image-4.png)
Medie

![alt text](image-5.png)
Feature distribution

# Conclusion

Gli autori suggeriscono DMTNet per affrontare il problema della CD-FSS. DMTNet utilizza SMT per la matrice di trasformazione di ogni immagine (query e support) e la matrice viene poi utilizzata per trasformare le caratteristiche specifiche del dominio. Poi, viene usato quindi DHC per la ipercorrelazione fra query e primo piano e sfondo, e con queste si ottiene una prediction mask per primo piano e sfondo (da usare per il training?). Durante il meta-testing invece viene usato il TSF (fine tuning al test time). Raggiunge buone prestazioni.

# Read but skip/skim math and parts that do not make sense

## Related work

Per quanto riguarda il FSS parla di come esistano sostanzialmente due metodologie, ovvero
- basati su metriche, ovvero le immagini di supporto come prototipi di classe e misurano la somiglianza tra query e questi
- basasti su relazioni, relazione fra coppie query-supporto

Parla poi di segmentation cross-domain dividendolo in 
- adattamento del dominio (DASS) ovvero utilizzo sia dei dati del dominio originale che di quello destinazione durante il training per generalizzare meglio
- generalizzazione del dominio (DGSS) che usa invece tecniche come normalizzazione, sbiancamento o random domain per colmare il divario fra i due domini

Infine parla di CD FSS e delle differenze con FSS (ovvero che in CD FSS ci sono molti divari fra i domini)

## Method

### Overview of DMTNet

DMTNet è composto da due domini:
- SMT che impara una matriche di trasformazione per ogni immagini di supporto e di query per trasformare le caratteristiche del dominio indipendenti
- DHC che costruisce invece le correlazioni tra query, sfondo e primo piano

Nel meta training, invece, le immaigni di supporto e di query vengono elaborate da una CNN per estratte le features multi-level, SMT impara quindi una matrice di trasformazione, DHC fa le correlazioni e infine un encoder e un decoder vengono usati per ottenere la mask
Nel meta testing, invece, si usa il TSF che affina i parametri durante il testing e poi la maschera finale viene ottenuta perfezionando la maschera grezza del meta-training

### Self-Matching Transformation

Parla dell'SMT che sostanzialmente:
- riduce la dipendenza dalle caratteristiche di supporto in quando SMT sfrutta anche le informazioni della query per evitare l'overfitting o se il support set è troppo piccolo o troppo diverso rispetto alla query
- genera una mask approssimativa basata sulla similarità di query-primo piano-sfondo in questo modo: 1. calcolo prototipi globali e locali di primo piano e sfondo; 2. divisione delle feature map di supporto in più feature map locali per una prediction più accurata; 3. calcolo mappe di correlazioni fra query e prototipi locali di supporto; 3. genera una query mask iniziale combinando informazioni di primo piano e dello sfondo; 4. utilizza una cross loss entropy binaria 
- trasforma le caratteristiche in modo adattivo, ovvero per rendere le caratteristiche indipendenti dal dominio, sia per le query che per le support: 1. costruisce le matrici per query e support image; 2. definisce le matrici di peso; 3. calcola le matrici di trasformazione; 4. perfeziona le matrici di trasformaizone dell'immagine query integrando le informazioni dell'immagine di supporto

### Dual Hypercorrelation Construction

Spiega cosa fa il DHC, ovvero:
Per prima cosa considera le correlazioni di sfondo, ovvero parte dalla supposizione che gli oggetti della stessa categoria tendono a trovarsi in ambienti simili e quindi lo sfondo di query e support image possono essere utili; poi costruisce le due ipercorrelazioni duali tra le dense feature dell'immagine di query e le feature di primo piano e sfondo dell'immagine di supporto;
Calcola la correlazione usando **corrfl**, correlazioni tra le caratteristiche di primo piano dell'immagine di supporto e le caratteristiche dell'immagine di query, calcolate utilizzando la similarità del coseno e **corrbl**  e correlazioni tra le caratteristiche di sfondo dell'immagine di supporto e le caratteristiche dell'immagine di query, calcolate in modo simile
Successivamente genera la prediction mask, in particolare due: **Mf** che è la prediction mask del primo piano della query ed *Mb* che invece è dello sfondo

Poi parte il training e anche in questo caso un BCE e infine ottimizza end to end con due loss, l1 per il self-matching ed l2 per il dhc

### Test-time Self-Finetuning

Sostanzialmente serve per migliorare le privisioni durante il test per i domini mai visti. Si svolge in due passaggi:
1. il modello genera le maschere di supporto previste e aggiorna la rete utilizzando una funzione di perdita basata sulla BCE tra le maschere previste e le ground truth
2. l'intera rete viene congelata e viene eseguita la previsione finale per l'immagine di query
infine cerca di aggiustare solo pochi parametri dell'encoder

## Experiments

### Experimental setup

Metrica di valutazione: IoU

1-way 1-shot: Una classe di supporto con un'immagine di supporto.
1-way 5-shot: Una classe di supporto con cinque immagini di supporto.
Numero di task per esecuzione: 1.200 per ISIC2018, Chest X-ray e Deepglobe, 2.400 per FSS-1000.

### Implementation details

Resnet50 come backbone, vgg 16 come feature extractor
immagini 400x400, 1e-6 per isic 2018 come lr durante il meta-testing

### Comparison with State of the Art methods

Generalmente superiore al sota

### Ablation study

Generalmente comportamento ottimo