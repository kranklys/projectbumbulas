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
- Paprastą ciklą, kuris periodiškai atnaujina duomenis ir logina svarbius sandorius.
- Telegram pranešimus apie „smart trade“ įvykius.
- `logging` naudojimą vietoje `print`, kad būtų lengviau derinti.

## Ko reikės

- Kompiuterio su **Windows / macOS / Linux**.
- Interneto ryšio.
- Įdiegto **Python 3.10+**.

## Kaip paleisti (žingsnis po žingsnio, žmogui be programavimo žinių)

Žemiau esantys žingsniai yra maksimaliai paprasti. Jei kuriame nors žingsnyje stringi – sustok ir patikrink, ar viską padarei tiksliai taip, kaip parašyta.

### 1) Atsisiųskite projektą
1. Atidarykite šio projekto GitHub puslapį.
2. Paspauskite **Code → Download ZIP**.
3. Išarchyvuokite ZIP failą į norimą vietą (pvz. **Darbalaukį**).

### 2) Įdiekite Python (jei neturite)
1. Atsisiųskite Python iš: https://www.python.org/downloads/
2. Diegimo metu pažymėkite **"Add Python to PATH"**.

### 3) Atidarykite terminalą / komandų eilutę

- **Windows:** paspauskite **Start** → įrašykite **Command Prompt** → atidarykite.
- **macOS:** atidarykite **Terminal** (Applications → Utilities).
- **Linux:** atidarykite **Terminal**.

### 4) Nueikite į projekto aplanką

Pavyzdžiai (pasirinkite savo kelią):

**Windows**
```bash
cd Desktop\projectbumbulas
```

**macOS / Linux**
```bash
cd ~/Desktop/projectbumbulas
```

> Jei projektą išarchyvavote kitoje vietoje, atitinkamai pakeiskite kelią.

### 5) Sukurkite virtualią aplinką (rekomenduojama)

#### Windows
```bash
python -m venv .venv
```

Aktyvuokite:
```bash
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
```

Aktyvuokite:
```bash
source .venv/bin/activate
```

### 6) Įdiekite reikalingas bibliotekas
```bash
pip install -r requirements.txt
```

## 7) Paleiskite botą

```bash
python main.py
```

Jei viskas gerai, terminale matysite `logging` pranešimus apie naujus sandorius.

> Norėdami sustabdyti – paspauskite **Ctrl + C**.

## 8) (Nebūtina) Telegram pranešimų įjungimas

Norėdami gauti pranešimus į Telegram:

1. Susikurkite Telegram botą per **@BotFather** ir gaukite token.
2. Sužinokite savo **chat_id** (pvz., per @userinfobot).
3. Nustatykite aplinkos kintamuosius (prieš paleidžiant botą):

**Windows (Command Prompt)**
```bash
set TELEGRAM_TOKEN=your_bot_token
set TELEGRAM_CHAT_ID=your_chat_id
```

**macOS / Linux**
```bash
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

Tada paleiskite botą:
```bash
python main.py
```

> Jei kintamieji nenustatyti, programa veiks, tačiau Telegram žinučių nesiųs.

## Papildomas pavyzdys (tik smalsiems)

Šiame etape projektas pateikia API klientą, kurį galima naudoti taip:

```python
from src.polymarket_api import PolyClient

client = PolyClient()
trades = client.fetch_latest_trades(limit=5)
print(trades)
```

> Šis pavyzdys **neatlieka** jokios prekybos – tik grąžina viešus duomenis.

## Kas toliau?

Tolimesni žingsniai galėtų būti:
- „Smart money“ piniginių aptikimas (pagal istorinius laimėjimus).
- Whale judėjimų filtravimas pagal apimtis.
- Paprastas „dashboard“ neprogramuojantiems vartotojams.

Jei norite, galime pridėti:
- Rinkų filtrus,
- „alert“ sistemą,
- Aiškią „smart money“ suvestinę.
