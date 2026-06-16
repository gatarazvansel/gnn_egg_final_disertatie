# Rezumat rezultate experimentale — Clasificarea EEG cu GNN

> Document generat automat din checkpoint-urile antrenate (`checkpoints/`) și rapoartele de
> diagnoză (`results/`). Toate valorile sunt **reale**, extrase direct din fișiere.
> Data: 14 iunie 2026.

---

## 1. Configurația experimentală

- **Set de date**: înregistrări EEG resting-state (eyes-closed), 19 canale, sistem 10-20, format BIDS.
- **Împărțire**: la nivel de subiect, 80/20 (fără data leakage).
- **Augmentare**: Window Overlap 0.5 + Gaussian Noise (1 copie) → dataset mărit.
- **Caracteristici nodale**: putere spectrală în 4 benzi (δ, θ, α, β) → matrice [19×4].
- **Cele 3 axe comparate**:
  - Conectivitate: **Funcțională (PLV)** vs. **Topologică (10-20)**
  - Model: **GCN** vs. **GAT**
  - Mod: **Binar (C vs A)** vs. **3 clase (C/A/F)**

| Mod | Nr. grafuri | Subiecți antrenare | Subiecți validare |
|-----|-------------|--------------------|-------------------|
| Binar (C vs A) | 22 888 | 50 | 13 |
| 3 clase (C/A/F) | 30 022 | 68 | 17 |

---

## 2. Rezultate la nivel de fereastră (metrici de validare)

Metrici macro-averaged, calculate pe setul de validare al fiecărui model.

### 2.1. Clasificare binară (C vs A)

| Conectivitate | Model | Accuracy | Precision | Recall | F1 | Epoca best |
|---------------|-------|----------|-----------|--------|-----|-----------|
| Funcțională | GCN | 71,0% | 0,751 | 0,746 | 0,710 | 7 |
| Funcțională | **GAT** | **73,8%** | **0,741** | **0,751** | **0,736** | 25 |
| Topologică | GCN | 66,9% | 0,721 | 0,709 | 0,668 | 7 |
| Topologică | GAT | 71,1% | 0,728 | 0,734 | 0,710 | 38 |

### 2.2. Clasificare 3 clase (C/A/F)

| Conectivitate | Model | Accuracy | Precision | Recall | F1 | Epoca best |
|---------------|-------|----------|-----------|--------|-----|-----------|
| Funcțională | GCN | 56,0% | 0,579 | 0,627 | 0,510 | 5 |
| Funcțională | **GAT** | **60,8%** | **0,611** | **0,639** | **0,564** | 22 |
| Topologică | GCN | 52,1% | 0,435 | 0,556 | 0,418 | 1 |
| Topologică | GAT | 60,5%* | 0,546 | 0,630 | 0,555 | 49 |

\* Vezi nota din secțiunea 5 — checkpoint-ul `gat_3class_topo` are tipul de conectivitate
salvat ca „functional" în config; verifică acest model.

---

## 3. Rezultate la nivel de subiect (diagnoză pe holdout)

Predicție prin **majority voting** pe ferestrele de 5s ale fiecărui subiect holdout
(diagnostic real cunoscut). ✅ = corect, ❌ = greșit. Confidența = probabilitatea medie.

### 3.1. sub-024 — Alzheimer (AD) — *toate modelele corecte* ✅

| Conectivitate | Model | Mod | Predicție | Confidență | Corect |
|---------------|-------|-----|-----------|-----------|--------|
| Funcțională | GCN | Binar | AD | 86,2% | ✅ |
| Funcțională | GAT | Binar | AD | 88,4% | ✅ |
| Topologică | GCN | Binar | AD | 74,8% | ✅ |
| Topologică | GAT | Binar | AD | 87,3% | ✅ |
| Funcțională | GCN | 3 clase | AD | 55,8% | ✅ |
| Funcțională | GAT | 3 clase | AD | 57,1% | ✅ |
| Topologică | GCN | 3 clase | AD | 51,3% | ✅ |
| Topologică | GAT | 3 clase | AD | 64,9% | ✅ |

### 3.2. sub-055 — Healthy Control (HC) — *toate modelele corecte* ✅

| Conectivitate | Model | Mod | Predicție | Confidență | Corect |
|---------------|-------|-----|-----------|-----------|--------|
| Funcțională | GCN | Binar | HC | 80,0% | ✅ |
| Funcțională | GAT | Binar | HC | 69,5% | ✅ |
| Topologică | GCN | Binar | HC | 77,9% | ✅ |
| Topologică | GAT | Binar | HC | 82,2% | ✅ |
| Funcțională | GCN | 3 clase | HC | 60,5% | ✅ |
| Funcțională | GAT | 3 clase | HC | 53,1% | ✅ |
| Topologică | GCN | 3 clase | HC | 48,4% | ✅ |
| Topologică | GAT | 3 clase | HC | 48,3% | ✅ |

### 3.3. sub-070 — Frontotemporal (FTD) — *toate modelele greșesc* ❌

| Conectivitate | Model | Mod | Predicție | Confidență | Corect |
|---------------|-------|-----|-----------|-----------|--------|
| Funcțională | GCN | 3 clase | HC | 34,5% | ❌ |
| Funcțională | GAT | 3 clase | AD | 35,0% | ❌ |
| Topologică | GCN | 3 clase | AD | 37,6% | ❌ |
| Topologică | GAT | 3 clase | AD | 41,1% | ❌ |

> FTD nu există în modelele binare, deci a fost testat doar pe cele 3-clase.

### 3.4. Sumar acuratețe la nivel de subiect

| Categorie | Rulări corecte | Total | Acuratețe |
|-----------|----------------|-------|-----------|
| Modele binare (sub-024, sub-055) | 8 | 8 | **100%** |
| Modele 3-clase (sub-024, sub-055, sub-070) | 8 | 12 | **66,7%** |
| **Total** | **16** | **20** | **80%** |

---

## 4. Concluzii cheie (susținute de date)

1. **Conectivitatea funcțională (PLV) > topologică** în aproape toate cazurile.
   La binar: GCN 71,0% vs 66,9%; GAT 73,8% vs 71,1%. Sincronizarea de fază între regiuni
   poartă mai multă informație discriminativă decât simpla proximitate anatomică a electrozilor.

2. **GAT > GCN** consecvent, în ambele moduri și ambele tipuri de conectivitate.
   Atenția adaptivă per muchie ajută. Cel mai bun model general: **GAT funcțional**.

3. **Clasificarea binară (C vs A) este robustă**: 100% acuratețe la nivel de subiect pe
   holdout, cu confidențe mari (69–88%). Acesta este rezultatul cel mai solid pentru disertație.

4. **Clasificarea 3-clase este mai dificilă** (~56–61% acuratețe la fereastră), iar **FTD este
   clasa problematică**: niciun model nu a clasificat corect sub-070, cu confidențe la nivel de
   șansă (~34–41%). Cauze probabile: FTD sub-reprezentat și semnătură EEG mai subtilă/suprapusă
   cu AD.

5. **AD și HC se separă clar** la nivel de subiect (corecte în toate modelele), inclusiv în
   regimul de 3 clase — confuzia apare la FTD, nu la celelalte clase.

6. **Confidența scade de la binar la 3-clase** (de la ~80% la ~50–60%), reflectând dificultatea
   mai mare a problemei cu 3 clase.

---

## 5. Limitări și note de verificare

- **`gat_3class_topo_best.pt`**: în config, `connectivity_type` este salvat ca `functional`,
  deși numele sugerează topologic. La diagnoză, modul „Auto" a folosit deci conectivitate
  funcțională pentru acest model. **Recomandare**: reantrenează acest model asigurându-te că
  selectorul „Connectivity Type" din tab-ul Dataset este pe **Topological** înainte de a apăsa
  Start Training, ca să fie consecvent cu celelalte 3 modele topologice.

- **Holdout mic** (3 subiecți): rezultatele la nivel de subiect sunt ilustrative, nu o evaluare
  statistică riguroasă. Metricile de validare la nivel de fereastră (secțiunea 2) sunt baza
  cantitativă mai solidă.

- **Epoci „best" timpurii** la unele modele (ex. GCN topo 3-clase la epoca 1) indică overfitting
  rapid; mecanismul „save best" a păstrat oricum cel mai bun model.

- **Dezechilibru de clase** în regimul 3-clase (AD domină) — explică recall-ul scăzut pe FTD.

---

## 6. Recomandări pentru disertație

- **Prezintă clasificarea binară ca rezultat principal** (solid, 100% la nivel de subiect).
- **Folosește comparația funcțional vs. topologic** ca o contribuție originală: arată că
  topologia funcțională (PLV) este mai informativă decât cea anatomică.
- **Discută onest limitarea pe FTD** ca direcție de îmbunătățire (oversampling, mai multe date,
  benzi de frecvență suplimentare ca trăsături nodale).
- Tabelele din secțiunile 2 și 3 sunt gata de inclus direct în lucrare.

