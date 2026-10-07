# FPL Expected Points Predictor

DAT158-prosjekt: predikerer forventede Fantasy Premier League-poeng for en spillers neste kamp, basert på formen de siste fem kampene.

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

## Struktur

- `src/data.py` – datainnlasting og feature engineering, delt mellom trening og app
- `src/train.py` – trener og evaluerer modellen
- `app.py` – Streamlit-app som viser prediksjoner
- `notebooks/` – utforsking

Tidligere sesonger hentes fra [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League). Inneværende sesong hentes direkte fra det offisielle FPL-API-et, slik at alle ferdigspilte runder er med.
