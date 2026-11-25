"""3D Simulator of the Raum Simulation / Stack Lab²³ – a menu you navigate with numbers.

Workflow: in the app go to Boxes → Measurement → "Export 3D job" (it lands in the Downloads folder) → fetch it here with [3] and calculate with [1] →
the result (<name>.box.json) is in the Output folder → in the app load it with "Load 3D result" in the horn simulation.
Start it with Start.bat (sets everything up on the first run).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

HIER = os.path.dirname(os.path.abspath(__file__))
WURZEL = os.path.dirname(HIER)
INPUT = os.path.join(WURZEL, 'Input')
OUTPUT = os.path.join(WURZEL, 'Output')
ERLEDIGT = os.path.join(INPUT, 'done')
DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
if HIER not in sys.path:
    sys.path.insert(0, HIER)


def sekunden(s):
    return '%d min %02d s' % (s // 60, s % 60) if s >= 60 else '%.0f s' % s


def groesse(pfad):
    mb = os.path.getsize(pfad) / 1048576
    return '%.1f MB' % mb if mb >= 1 else '%.0f KB' % (os.path.getsize(pfad) / 1024)


def auftraege():
    """Jobs in the Input folder (.json), without the subfolder done."""
    return sorted(f for f in os.listdir(INPUT) if f.lower().endswith('.json') and os.path.isfile(os.path.join(INPUT, f)))


def ergebnisse():
    return sorted(f for f in os.listdir(OUTPUT) if f.lower().endswith('.json'))


def rechnerinfo():
    """Where the calculation runs: GPU (if set up) or CPU."""
    try:
        import gpu
        ok, name = gpu.verfuegbar()
        return 'GPU ' + name if ok else 'CPU'
    except Exception:
        return 'CPU'


# Logo (cat in a ring) and lettering from the start animation of the app, drawn with block characters in the top right corner of the window.
# Cat: converted from the app's css/katze.png into half blocks (once, built in: no image library needed).
KATZE = [
    '                             ▄▄███',
    '         ▄▄ ▀▀▀▀▀▀         ▄█████▀',
    '      ▄▀▀                ▄██████▀',
    '    ▄▀        ▄▄████▄   ███████▀',
    '  ▄▀        ▄█████████ ███████▀',
    ' ▄      ▄▄ █████████████████▀ ▄',
    '  ▄   ▄▄██▄███████████████▀    █',
    '█ ██████████████████████▀      █',
    '█  █████████████████████',
    '█ ▀███▄ ████████████████',
    '▀   ▀ ▄▄█ ▀█████▀██████        █',
    ' ▀         █████ █████',
    '  ▀        ████   ████       ▄',
    '   ▀▄      ████   ██▀██    ▄▀',
    '       ▄  ▄████   ██ █▀  ▄▀',
    '         ██▀██▀ ▄██ ▄█▀▀',
    '                   ███',
]
# Lettering "STACK LAB²³" from a pixel font (3 × 5), in half blocks (3 rows high)
SCHRIFT = {
    'S': ['███', '█  ', '███', '  █', '███'], 'T': ['███', ' █ ', ' █ ', ' █ ', ' █ '], 'A': [' █ ', '█ █', '███', '█ █', '█ █'],
    'C': ['███', '█  ', '█  ', '█  ', '███'], 'K': ['█ █', '█ █', '██ ', '█ █', '█ █'], 'L': ['█  ', '█  ', '█  ', '█  ', '███'],
    'B': ['██ ', '█ █', '██ ', '█ █', '██ '], ' ': ['  ', '  ', '  ', '  ', '  '],
}
LOGO_BREITE = 38   # room for logo and lettering (36 characters) plus right margin


def schriftzug(text='STACK LAB'):
    zeilen = ['', '', '']
    for ch in text:
        g = SCHRIFT[ch] + ['   ' if len(SCHRIFT[ch][0]) == 3 else '  ']   # 6. Pixelzeile leer
        for r in range(3):
            for c in range(len(g[0])):
                o, u = g[2 * r][c] != ' ', g[2 * r + 1][c] != ' '
                zeilen[r] += '█' if o and u else '▀' if o else '▄' if u else ' '
            zeilen[r] += ' '
    zeilen[0] = zeilen[0].rstrip() + '²³'
    return [z.rstrip() for z in zeilen]


def terminal_breite():
    try:
        return shutil.get_terminal_size((100, 30)).columns
    except OSError:
        return 100


def ansi_an():
    """Enable escape sequences in the Windows console (newer Windows versions know them but need them switched on)."""
    if os.name == 'nt':
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            m = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(m)):
                k.SetConsoleMode(h, m.value | 0x0004)
                return True
            return False
        except Exception:
            return False
    return sys.stdout.isatty()


def logo_zeichnen():
    """Write logo and lettering at fixed positions in the top right and restore the cursor. Window too narrow: nothing."""
    breite = terminal_breite()
    if breite < LOGO_BREITE + 78 or not sys.stdout.isatty() or not ansi_an():
        return
    spalte = breite - LOGO_BREITE
    wort = schriftzug()
    zeilen = [z.ljust(LOGO_BREITE - 2) for z in KATZE] + [''] + [(' ' * ((LOGO_BREITE - 2 - len(z)) // 2) + z) for z in wort]
    out = ['\x1b7']   # remember cursor
    for i, z in enumerate(zeilen):
        if z.strip():
            out.append('\x1b[%d;%dH\x1b[97m%s\x1b[0m' % (2 + i, spalte, z))
    out.append('\x1b8')
    sys.stdout.write(''.join(out))
    sys.stdout.flush()


def breite_links():
    """Maximum width of the text on the left so it does not run under the logo."""
    b = terminal_breite()
    return b - LOGO_BREITE - 3 if b >= LOGO_BREITE + 78 else 200


def kuerzen(text, n):
    return text if len(text) <= n else text[:max(1, n - 1)] + '…'


def kopf(einstellungen):
    os.system('cls' if os.name == 'nt' else 'clear')
    logo_zeichnen()
    print()
    print('  ' + '=' * 62)
    print('    3D Simulator  (Raum Simulation / Stack Lab²³)')
    print('  ' + '=' * 62)
    print()
    n = breite_links()
    print(kuerzen('  Input:  %s   (%d job(s))' % (INPUT, len(auftraege())), n))
    print(kuerzen('  Output: %s   (%d result(s))' % (OUTPUT, len(ergebnisse())), n))
    print('  Running on: %s' % einstellungen['rechner'])
    print()


def warte():
    print()
    input('  Press Enter to return to the menu ...')


def rechne_datei(name, einstellungen):
    """Calculate one job from Input with rs3d.py. Returns True if it worked."""
    pfad = os.path.join(INPUT, name)
    cmd = [sys.executable, os.path.join(HIER, 'rs3d.py'), pfad] + (['--cpu'] if einstellungen['cpu'] else [])
    env = dict(os.environ, RS3D_AUSGABE=OUTPUT, PYTHONIOENCODING='utf-8')
    print()
    print('  ' + '-' * 62)
    print('  Calculating: %s' % name)
    print('  ' + '-' * 62)
    t0 = time.time()
    try:
        ok = subprocess.run(cmd, cwd=WURZEL, env=env).returncode == 0
    except KeyboardInterrupt:
        print('\n  (aborted)')
        return False
    print('  ' + '-' * 62)
    if ok:
        print('  Finished in %s.' % sekunden(time.time() - t0))
        if einstellungen['erledigt']:
            try:
                os.makedirs(ERLEDIGT, exist_ok=True)
                ziel = os.path.join(ERLEDIGT, name)
                if os.path.exists(ziel):
                    os.remove(ziel)
                shutil.move(pfad, ziel)
                print('  Job moved to Input\\done')
            except OSError as e:
                print('  Note: the job could not be moved (%s)' % e)
    else:
        print('  FAILED (see message above). The job stays in Input.')
    return ok


def alle_rechnen(einstellungen):
    liste = auftraege()
    if not liste:
        print('  No jobs in the Input folder. Use [3] first or put the file there by hand.')
        return
    ok = 0
    for n in liste:
        ok += bool(rechne_datei(n, einstellungen))
    print()
    print('  Total: %d of %d succeeded.  Results: %s' % (ok, len(liste), OUTPUT))


def einen_rechnen(einstellungen):
    liste = auftraege()
    if not liste:
        print('  No jobs in the Input folder.')
        return
    print()
    for i, n in enumerate(liste, 1):
        print('  %2d  %-50s %10s' % (i, n, groesse(os.path.join(INPUT, n))))
    print()
    w = input('  Number (Enter = back): ').strip()
    if w.isdigit() and 1 <= int(w) <= len(liste):
        rechne_datei(liste[int(w) - 1], einstellungen)
    elif w:
        print('  Invalid number.')


def aus_downloads_holen():
    """Aufträge der App (*.auftrag*.json) aus dem Download-Ordner nach Input verschieben."""
    treffer = sorted(glob.glob(os.path.join(DOWNLOADS, '*.auftrag*.json')), key=os.path.getmtime, reverse=True)[:15]
    if not treffer:
        print('  There is no job in the Downloads folder (*.auftrag.json):')
        print('  ' + DOWNLOADS)
        print('  In the app: Boxes → Measurement → "Export 3D job".')
        return
    print()
    print('  Jobs in the Downloads folder (newest first):')
    print()
    for i, p in enumerate(treffer, 1):
        print('  %2d  %-50s %10s   %s' % (i, os.path.basename(p), groesse(p), time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))))
    print()
    w = input('  Number(s) separated by spaces, a = all, Enter = back: ').strip().lower()
    if not w:
        return
    wahl = treffer if w == 'a' else [treffer[int(x) - 1] for x in w.split() if x.isdigit() and 1 <= int(x) <= len(treffer)]
    for p in wahl:
        ziel = os.path.join(INPUT, os.path.basename(p))
        shutil.move(p, ziel)
        print('  Moved to Input: %s' % os.path.basename(p))
    if not wahl:
        print('  Nothing selected.')


def dateien_zeigen():
    print()
    print('  Input:')
    for n in auftraege() or ['(leer)']:
        print('    ' + n)
    print()
    print('  Output:')
    for n in ergebnisse() or ['(leer)']:
        pfad = os.path.join(OUTPUT, n)
        print('    %-50s %10s' % (n, groesse(pfad) if os.path.isfile(pfad) else ''))


def ordner_oeffnen():
    for p in (INPUT, OUTPUT):
        try:
            if os.name == 'nt':
                os.startfile(p)  # noqa: S606
            else:
                subprocess.run(['xdg-open' if sys.platform != 'darwin' else 'open', p], check=False)
        except OSError as e:
            print('  Could not open %s: %s' % (p, e))


def gpu_einrichten(einstellungen):
    print()
    print('  GPU acceleration (NVIDIA graphics cards only, download of about 1-2 GB):')
    if input('  Install now? (y/N): ').strip().lower() != 'y':
        return
    req = os.path.join(WURZEL, 'requirements-gpu.txt')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req])
    print('  ' + ('Done.' if r.returncode == 0 else 'The installation failed (see message above).'))
    einstellungen['rechner'] = rechnerinfo()


def menue():
    einstellungen = {'cpu': False, 'erledigt': True, 'rechner': rechnerinfo()}
    os.makedirs(INPUT, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)
    while True:
        kopf(einstellungen)
        print('  [1] Calculate all jobs in Input')
        print('  [2] Pick one job and calculate it')
        print('  [3] Fetch jobs from the Downloads folder')
        print('  [4] Show files in Input and Output')
        print('  [5] Open the Input and Output folders')
        print('  [6] Use CPU only: %s  (toggle)' % ('ON' if einstellungen['cpu'] else 'OFF'))
        print('  [7] Move finished jobs to Input\\done: %s  (toggle)' % ('ON' if einstellungen['erledigt'] else 'OFF'))
        print('  [8] Set up GPU acceleration (NVIDIA)')
        print('  [0] Quit')
        print()
        wahl = input('  Choice: ').strip()
        try:
            if wahl == '1':
                alle_rechnen(einstellungen); warte()
            elif wahl == '2':
                einen_rechnen(einstellungen); warte()
            elif wahl == '3':
                aus_downloads_holen(); warte()
            elif wahl == '4':
                dateien_zeigen(); warte()
            elif wahl == '5':
                ordner_oeffnen()
            elif wahl == '6':
                einstellungen['cpu'] = not einstellungen['cpu']
                einstellungen['rechner'] = 'Prozessor (gewählt)' if einstellungen['cpu'] else rechnerinfo()
            elif wahl == '7':
                einstellungen['erledigt'] = not einstellungen['erledigt']
            elif wahl == '8':
                gpu_einrichten(einstellungen); warte()
            elif wahl == '0':
                return
            else:
                print('\n  Invalid choice. Please enter 0 to 8.')
                time.sleep(1.5)
        except KeyboardInterrupt:
            print('\n  (aborted)')
            time.sleep(1)


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    try:
        menue()
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        import traceback
        print('\n  Unexpected error:')
        traceback.print_exc()
        try:
            input('\n  Press Enter to quit ...')
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == '__main__':
    main()
