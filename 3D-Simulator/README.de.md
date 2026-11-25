# 3D-Simulator

[English](README.md) · **Deutsch**

Rechnet den 3D-Frequenzgang einer Box (Basshorn, Hybrid, Bandpass, geschlossene Box …) aus dem SketchUp-Modell, mit Finite Elementen auf dem
Voxelraster der Luft. Gehört zur **Raum Simulation / Stack Lab²³**: die App erzeugt den Auftrag, der Simulator rechnet, das Ergebnis geht zurück in die App.

> Das Programm selbst (Menü und Meldungen) ist auf Englisch.

## Installation (Windows) – ein Doppelklick

1. Dieses Repository als Zip herunterladen (**Code → Download ZIP**) und entpacken.
2. **`Start.bat`** doppelklicken.

Beim ersten Start richtet das Skript **alles in einem Rutsch** ein (ca. 2–5 Minuten, braucht Internet):

* Python wird automatisch installiert, falls es fehlt (über winget, nur für deinen Benutzer),
* eine eigene Python-Umgebung `.venv` mit allen Paketen (numpy, scipy, scikit-fem, pyamg),
* die Ordner **`Input`** und **`Output`**,
* bei einer NVIDIA-Grafikkarte die Frage, ob die GPU-Beschleunigung mit installiert werden soll (1–2 GB, geht auch später im Menü).

Danach startet jedes Mal sofort das Menü. Zum reinen Einrichten ohne Start: `Installieren.bat`.

> Falls winget fehlt, öffnet sich die Python-Downloadseite: installieren (unten **„Add python.exe to PATH“** ankreuzen) und `Start.bat` nochmal starten.

## So rechnest du eine Box

1. In der App: **Boxen → Vermessung → 3D-Auftrag exportieren** (Marker für Druck- und Rückkammer müssen gesetzt sein). Die Datei
   `<Boxname>.auftrag.json` landet im Download-Ordner; das Modell (.dae) steckt schon darin.
2. `Start.bat` → Menü **`3`** (Aufträge aus dem Download-Ordner holen) → Menü **`1`** (alle Aufträge rechnen).
3. Das Ergebnis liegt in **`Output`** als `<Boxname>.box.json`.
4. In der App bei der Hornsimulation **3D-Ergebnis laden** und diese Datei wählen.

Statt Menü `3` kannst du den Auftrag auch von Hand in den Ordner `Input` legen.

## Menü

| Taste | Funktion |
|---|---|
| `1` | Alle Aufträge in `Input` rechnen |
| `2` | Einen Auftrag auswählen und rechnen |
| `3` | Aufträge aus dem Download-Ordner nach `Input` holen |
| `4` | Dateien in `Input` und `Output` anzeigen |
| `5` | Ordner `Input` und `Output` im Explorer öffnen |
| `6` | Nur Prozessor verwenden (GPU ausschalten) |
| `7` | Fertige Aufträge nach `Input\done` verschieben: ON/OFF (Standard ON) |
| `8` | GPU-Beschleunigung einrichten (NVIDIA) |
| `0` | Beenden |

## Dauer und Speicher

Raster 2 cm: Probelauf, etwa 30 s. Raster 1 cm: das Ergebnis, etwa 4 Minuten und bis 5 GB Arbeitsspeicher
(Zeiten je nach Rechner; mit NVIDIA-GPU schneller). Das Raster wählst du in der App beim Export.

## Ohne Menü (Fortgeschrittene)

```bat
.venv\Scripts\python rechner\rs3d.py Input\meine-box.auftrag.json
```
`--cpu` erzwingt den Prozessor. Der Auftragsaufbau ist im Kopf von `rechner/rs3d.py` beschrieben (die JSON-Schlüssel sind absichtlich deutsch: so schreibt sie die App).

## Für Entwickler

Der Rechenkern (`rechner/*.py` außer `menue_3d.py`) wird aus dem Hauptprojekt (`werkzeug/`) übernommen:
dort `py werkzeug/sync_3d.py` ausführen, dann diesen Ordner ins Repository übernehmen.
In dieser Kopie sind Menü und Meldungen englisch; Kommentare und Docstrings im Rechenkern sind noch deutsch.
