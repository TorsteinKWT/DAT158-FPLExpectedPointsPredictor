# FPL Expected Points Predictor

DAT158-prosjekt: predikerer forventede Fantasy Premier League-poeng for en spillers neste kamp, basert på formen de siste fem kampene, motstanderens form og FPLs vanskelighetsgrad for kampen (FDR).

## Oppsett

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Bruk

Tren modellen (laster ned data til `data/raw/` og skriver `model.joblib`):

```bash
python -m src.train
```

Start appen:

```bash
streamlit run app.py
```

## Live-test

Før hver runde lagres prediksjonene fra alle modellene, slik at de kan sammenlignes med fasiten etterpå:

```bash
python -m src.snapshot
```

Filen havner i `predictions/`. En runde lagres bare én gang, og aldri etter at første kamp har startet. Skriptet kjøres også daglig av GitHub Actions (`.github/workflows/lagre-prediksjoner.yml`), som committer filen. Resultatene vises på siden «Live-test» i appen når runden er ferdigspilt.

## Struktur

- `src/data.py` – datainnlasting og feature engineering, delt mellom trening og app
- `src/train.py` – trener og evaluerer modellen
- `app.py` – Streamlit-app som viser prediksjoner
- `notebooks/` – utforsking

Tidligere sesonger hentes fra [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League). Inneværende sesong hentes direkte fra det offisielle FPL-API-et, slik at alle ferdigspilte runder er med.

## Bidragsytere

- Torstein er tullete 🤡
