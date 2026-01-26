# Flask Antenna Calculator (tugas mikrostrip)

UI + parameter tetap mengikuti PDF tugas:
- frekuensi: 1.8, 2.1, 2.3, 2.4, 3.3 GHz
- h = 1.6 mm, Zo = 50 Ω, εr = 4.4, c = 3e8 m/s

## 1) Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Run

```bash
python app.py
```

Buka:
http://127.0.0.1:5000

## Catatan
- "Mode rumus Wp": ada 2 pilihan:
  - Modul (sesuai PDF)
  - Standar (umum)
