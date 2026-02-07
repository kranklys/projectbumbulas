# Contributing

Ačiū, kad norite prisidėti! Žemiau – paprastas, aiškus procesas, kad galėtume lengvai testuoti ir redaguoti pakeitimus GitHub'e.

## Greita eiga

1. **Fork** šį repo į savo GitHub paskyrą.
2. **Sukurkite naują šaką** (branch) savo pakeitimams.
3. **Paleiskite testus lokaliai**.
4. **Atidarykite Pull Request** į pagrindinę `main` šaką.

## Darbas lokaliai

```bash
# 1) Klonuokite savo fork

git clone https://github.com/<your-user>/projectbumbulas.git
cd projectbumbulas

# 2) Susikurkite šaką pakeitimams

git checkout -b feature/my-change

# 3) Įdiekite priklausomybes

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4) Paleiskite testus
pytest
```

## Pull Request gairės

- Aiškiai aprašykite, **ką** pakeitėte ir **kodėl**.
- Jei keičiami analizės kriterijai ar slenksčiai – pridėkite pavyzdį ar paaiškinimą.
- Jei keičiate API logiką ar konfiguraciją – nurodykite, kaip tai įtakoja vartotoją.

## CI patikra

GitHub Actions automatiškai paleis testus kiekvienam PR ir `main` šakai. Jei CI nesėkmingas, pataisykite prieš prašydami peržiūros.
