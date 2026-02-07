# Polymarket Smart Money Tracker (Foundation)

> **Svarbu:** Šis projektas **nevykdo** sandorių ir **nepriima** prekybinių sprendimų. Jis skirtas **stebėjimui ir analizei**, kad žmogus pats priimtų galutinį sprendimą.

## Projekto tikslai

Šio projekto tikslas – sukurti pagrindą botui, kuris:
- seka **aukšto laimėjimo rodiklio** pinigines (high-win-rate wallets),
- stebi **banginių (whale) judėjimus** Polymarket platformoje,
- pateikia aiškų, žmogui suprantamą vaizdą apie **„smart money“** aktyvumą.

## Ką ši versija jau turi

- Tvarkingą projekto struktūrą.
- Konfigūraciją Polymarket CLOB API.
- API klientą su metodu **paskutiniams sandoriams** gauti.
- Analizatorių, kuris aptinka **whale** sandorius ir **smart money** adresus.
- Paprastą ciklą, kuris kas 60s atnaujina duomenis ir logina svarbius sandorius.
- „Liquidity & Spread Guard“ filtrą: jei rinka **per mažai likvidi** arba **su plačiu spread**, signalas rodomas tik esant labai aukštam balui.
- **Multi‑Signal Fusion**: jei per 60 min į tą pačią rinką įeina ≥3 „smart“ piniginės, signalas tampa **MOMENTUM_ALERT**.
- „Smart Money History“: bot’as įsimena, kokias pinigines jau matė, ir suteikia „Repeat Offender“ bonusą watchlist adresams.
- Telegram pranešimus apie „smart trade“ įvykius.
- `logging` naudojimą vietoje `print`, kad būtų lengviau derinti.

## Ko reikės

- Kompiuterio su **Windows / macOS / Linux**.
- Interneto ryšio.
- Įdiegto **Python 3.10+**.

## Kaip paleisti (žingsnis po žingsnio)

### 1) Atsisiųskite projektą
1. Atidarykite šio projekto GitHub puslapį.
2. Paspauskite **Code → Download ZIP**.
3. Išarchyvuokite ZIP failą į norimą vietą (pvz. **Darbalaukį**).

### 2) Įdiekite Python (jei neturite)
1. Atsisiųskite Python iš: https://www.python.org/downloads/
2. Diegimo metu pažymėkite **"Add Python to PATH"**.

### 3) Įdiekite reikalingas bibliotekas

#### Windows
1. Atidarykite **Command Prompt**.
2. Nueikite į projekto aplanką:
   ```bash
   cd Desktop\projectbumbulas
   ```
3. Sukurkite virtualią aplinką:
   ```bash
   python -m venv .venv
   ```
4. Aktyvuokite ją:
   ```bash
   .venv\Scripts\activate
   ```
5. Įdiekite bibliotekas:
   ```bash
   pip install -r requirements.txt
   ```

#### macOS / Linux
1. Atidarykite **Terminal**.
2. Nueikite į projekto aplanką:
   ```bash
   cd ~/Desktop/projectbumbulas
   ```
3. Sukurkite virtualią aplinką:
   ```bash
   python3 -m venv .venv
   ```
4. Aktyvuokite ją:
   ```bash
   source .venv/bin/activate
   ```
5. Įdiekite bibliotekas:
   ```bash
   pip install -r requirements.txt
   ```

## Greitas API kliento pavyzdys

Šiame etape projektas pateikia API klientą, kurį galima naudoti taip:

```python
from src.polymarket_api import PolyClient

client = PolyClient()
trades = client.fetch_latest_trades(limit=5)
print(trades)
```

> Šis pavyzdys **neatlieka** jokios prekybos – tik grąžina viešus duomenis.

## Kaip paleisti Smart Money „Engine“

Šis „engine“ kas 60 sekundžių:
- pasiima naujausius sandorius,
- atrenka **whale** sandorius (> $10k),
- patikrina **watchlist** adresus,
- pažymi sandorius, kurie pakeitė kainą > 2% (High Impact),
- taiko **risk guard** (mažai likvidžios rinkos slopinamos, nebent balas > 90),
- aptinka **momentum** (daug aukšto balo signalų per 60 min).

Paleidimas:
```bash
python main.py
```

> Rezultatai bus matomi terminale per `logging` pranešimus.

### Momentum ir rizikos filtrų paaiškinimas

- **MOMENTUM_ALERT**: jei ≥3 aukšto balo sandoriai toje pačioje rinkoje per 60 min.
- **Risk Guard**: jei rinka „High“ rizikos, signalas siunčiamas tik kai balas > 90.

### Trader History (persistencija)

Bot’as saugo stebėtas pinigines faile:
```
data/trader_history.json
```
Šis failas naudojamas „Repeat Offender“ bonusui (kai watchlist piniginė pasikartoja).

### Telegram pranešimų įjungimas (nebūtina)

Norėdami gauti pranešimus į Telegram:

1. Susikurkite Telegram botą per **@BotFather** ir gaukite token.
2. Sužinokite savo **chat_id** (pvz., per @userinfobot).
3. Nustatykite aplinkos kintamuosius:

```bash
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

> Jei kintamieji nenustatyti, programa toliau veiks, tačiau Telegram žinučių nesiųs.

## Kas toliau?

Tolimesni žingsniai galėtų būti:
- „Smart money“ piniginių aptikimas (pagal istorinius laimėjimus).
- Whale judėjimų filtravimas pagal apimtis.
- Paprastas „dashboard“ neprogramuojantiems vartotojams.

Jei norite, galime pridėti:
- Rinkų filtrus,
- „alert“ sistemą,
- Aiškią „smart money“ suvestinę.
