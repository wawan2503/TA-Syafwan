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
- Rumus mengikuti pendekatan umum rectangular microstrip patch (transmission-line model) + microstrip line (closed-form).
- Ukuran ground/substrate di app ini mencakup segmen feedline (layout pada gambar), jadi: `Lg = Lp + Lf + 6h`.

## Rumus (ringkas)
- Patch width: `Wp = c/(2f) * sqrt(2/(εr+1))`
- Patch effective permittivity: `εeff = (εr+1)/2 + (εr-1)/2 * (1/sqrt(1+12h/Wp))` (dengan koreksi Hammerstad untuk `Wp/h < 1`)
- Fringing extension: `ΔL = 0.412 h * ((εeff+0.3)(Wp/h+0.264))/((εeff-0.258)(Wp/h+0.8))`
- Patch length: `Lp = c/(2f*sqrt(εeff)) - 2ΔL`
- Feedline width `Wf` dihitung dari `Zo` memakai aproksimasi closed-form `Wf/h` (umum dipakai di banyak paper/desain microstrip).
- Feedline length (quarter guided wavelength): `Lf = (c/(f*sqrt(εeff_line))) / 4`

## Referensi (jurnal/paper)
- J. W. Howell, “Microstrip Antennas,” *IEEE Transactions on Antennas and Propagation*, 1975. DOI: 10.1109/TAP.1975.1141009
- K. R. Carver and J. W. Mink, “Microstrip Antenna Technology,” *IEEE Transactions on Antennas and Propagation*, 1981. DOI: 10.1109/TAP.1981.1142523
- E. O. Hammerstad and O. Jensen, “Accurate Models for Microstrip Computer-Aided Design,” *IEEE MTT-S International Microwave Symposium Digest*, 1980. DOI: 10.1109/MWSYM.1980.1124303
- M. Kirschning and R. H. Jansen, “Accurate Model for Effective Dielectric Constant of Microstrip with Validity up to Millimetre-Wave Frequencies,” *Electronics Letters*, 1982. DOI: 10.1049/el:19820186
