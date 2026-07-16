import os
import csv
import json
import time
import random
import requests
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

# Raiz del proyecto: anclada al fichero, NO al CWD -> los scripts funcionan
# desde cualquier directorio.
ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Falta API_KEY. Copia .env.example a .env y pon tu key de "
        "https://developer.riotgames.com/ (las dev key caducan cada 24h)."
    )

HEADERS = {"X-Riot-Token": API_KEY}

PLATFORM = "euw1.api.riotgames.com"
REGION = "europe.api.riotgames.com"

# OJO: son DOS identificadores distintos de la misma cola, y cada API quiere el suyo.
#   QUEUE     (string) -> league-v4, para SEMBRAR jugadores por liga.
#   QUEUE_ID  (int)    -> match-v5,  para FILTRAR que partidas bajamos.
# Usar solo el primero fue el bug: by-puuid/ids sin `queue` devuelve TODAS las colas
# del jugador (Arena, co-op vs IA, ARAM...) y el 19.1% del dataset acabo sin ser
# soloQ. Ver experiments/queue_ablation.py.
QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420
TARGET_MATCHES = 10000          # objetivo para la baseline; sube esto cuando tengas la personal key
MATCHES_PER_PLAYER = 20
SAVE_EVERY = 25

# --- banda de elo objetivo ---------------------------------------------------
# El elo afecta sobre todo a la CALIBRACION: en elo bajo las ventajas se
# convierten peor, asi que el mismo estado merece una probabilidad mas cerca del
# 50%. Un modelo de alto elo servido en elo medio sale SOBRECONFIADO. Como el
# Live Client Data no expone el rango de los 10 jugadores, el elo no puede ser
# una feature: hay que entrenar en la banda en la que se va a servir.
#
# tier -> divisiones a sembrar, o None si el tier no tiene divisiones.
# Master/GM/Challenger no las tienen (todos son "I") y NO se pueden pedir por
# entries/{queue}/MASTER/{div}: eso responde 400. Van por su endpoint propio
# ({tier}leagues/by-queue), que devuelve la liga entera de una tacada.
# Grandmaster y Challenger quedan fuera a proposito: el objetivo es la banda
# media-alta, no el top del ladder.
SEED_TIERS = {
    "EMERALD": ("I", "II", "III", "IV"),
    "DIAMOND": ("I", "II", "III", "IV"),
    "MASTER": None,
}
SEED_PAGES_PER_DIVISION = 8     # league-v4 da ~205 jugadores por pagina
# Techo de semillas por tier. Existe porque los dos endpoints tienen FORMA
# distinta: Emerald/Diamond dan ~205 por pagina (~6.5k por tier con 8 paginas),
# pero masterleagues suelta 10.000 de golpe. Sin techo, Master seria la mayoria
# de la cola y la banda se iria arriba por un accidente de la API, no por una
# decision. Subelo o bajalo para pesar mas un tier u otro.
SEED_MAX_PER_TIER = 3000

# Limites de la development key: 20 req/s y 100 req cada 120 s.
# Nos quedamos por debajo para no comernos 429 constantes.
MAX_PER_SECOND = 18
MAX_PER_WINDOW = 95
WINDOW_SECONDS = 120

OUT_DIR = ROOT / "riot_dataset"
RAW_DIR = OUT_DIR / "matches"
TIMELINE_DIR = OUT_DIR / "timelines"   # lo baja ESTE script, no build_features
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TIMELINE_DIR, exist_ok=True)

# Procedencia de cada partida: match_id -> (tier, division) del jugador semilla.
# NO es cosmetico. riot_dataset/matches/ ya contiene un crawl viejo sembrado en
# Master+; sin esto, un crawl nuevo en Esmeralda se MEZCLA con el en el mismo
# directorio y features.csv sale con dos bandas de elo revueltas, sin que nada lo
# delate. Misma leccion que queue_id: registrar la procedencia y filtrar al
# cargar (train.load_dataset). Las partidas sin fila aqui quedan como UNKNOWN.
RANKS_CSV = OUT_DIR / "match_rank.csv"

seen_match_ids = set()
seen_puuids = set()
player_queue = deque()

_req_times = deque()

def _throttle():
    """Bloquea hasta que sea seguro hacer otra peticion segun los limites."""
    while True:
        now = time.time()
        while _req_times and now - _req_times[0] > WINDOW_SECONDS:
            _req_times.popleft()
        # ventana de 2 min
        if len(_req_times) >= MAX_PER_WINDOW:
            time.sleep(WINDOW_SECONDS - (now - _req_times[0]) + 0.1)
            continue
        # ventana de 1 s
        in_last_second = sum(1 for t in _req_times if now - t < 1.0)
        if in_last_second >= MAX_PER_SECOND:
            time.sleep(0.2)
            continue
        break
    _req_times.append(time.time())

def load_existing():
    """Cuenta las partidas ya descargadas para reanudar sin volver a bajarlas."""
    for fn in os.listdir(RAW_DIR):
        if fn.endswith(".json"):
            seen_match_ids.add(fn[:-5])

def riot_get(url, params=None, retries=5):
    for attempt in range(retries):
        _throttle()
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry_after + 0.2)
            continue
        if r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed request: {url}")

def get_league_entries(tier, division, page=1):
    """Jugadores de un tier CON divisiones (EMERALD I, DIAMOND IV, ...).

    Endpoint paginado, ~205 por pagina. Solo vale de IRON a DIAMOND: con MASTER
    y arriba responde 400 (verificado).
    """
    url = f"https://{PLATFORM}/lol/league/v4/entries/{QUEUE}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_league(tier):
    """Jugadores de un tier SIN divisiones (master/grandmaster/challenger).

    Otro endpoint y otra forma: devuelve un LeagueList con la liga entera en una
    sola respuesta (Master ~10.000 entries), sin paginar.
    """
    url = f"https://{PLATFORM}/lol/league/v4/{tier.lower()}leagues/by-queue/{QUEUE}"
    return riot_get(url).get("entries", [])

def get_match_ids(puuid, start=0, count=20):
    # `queue` NO es opcional: sin el, la API devuelve todas las colas del jugador.
    return riot_get(f"https://{REGION}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                    params={"start": start, "count": count, "queue": QUEUE_ID})

def get_match(match_id):
    url = f"https://{REGION}/lol/match/v5/matches/{match_id}"
    return riot_get(url)

def get_timeline(match_id):
    url = f"https://{REGION}/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(url)

def save_match(match):
    match_id = match["metadata"]["matchId"]
    path = os.path.join(RAW_DIR, f"{match_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(match, f)

def save_timeline(match_id, timeline):
    path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline, f)

def _candidatos_de_tier(tier, divisions):
    """[(puuid, tier, division)] de un tier, por el endpoint que le toque."""
    fuera = []
    if divisions is None:                     # master/gm/challenger: una sola llamada
        try:
            entries = get_apex_league(tier)
        except Exception as ex:
            print(f"  ! {tier}: {ex}", flush=True)
            return fuera
        fuera += [(e["puuid"], tier, e.get("rank", "I")) for e in entries if e.get("puuid")]
        print(f"  {tier}: {len(entries)} jugadores en la liga", flush=True)
        return fuera

    for division in divisions:                # emerald/diamond: paginado
        antes = len(fuera)
        for page in range(1, SEED_PAGES_PER_DIVISION + 1):
            try:
                entries = get_league_entries(tier, division, page=page)
            except Exception as ex:
                print(f"  ! {tier} {division} p{page}: {ex}", flush=True)
                break
            if not entries:
                break                          # se acabaron las paginas
            fuera += [(e["puuid"], tier, division) for e in entries if e.get("puuid")]
        print(f"  {tier} {division}: {len(fuera) - antes} jugadores", flush=True)
    return fuera


def seed_from_tiers():
    """Siembra jugadores SOLO de la banda de elo objetivo (SEED_TIERS).

    Sin random walk a proposito. El crawler antiguo metia 5 participantes de cada
    partida a la cola, lo que hace que el elo derive ladder abajo sin medida ni
    control. Sembrando directo del tier la banda queda fijada por construccion, y
    en estos tiers hay jugadores de sobra para TARGET_MATCHES sin expandir.
    """
    seeds = []
    for tier, divisions in SEED_TIERS.items():
        cand = _candidatos_de_tier(tier, divisions)
        if len(cand) > SEED_MAX_PER_TIER:     # que la forma del endpoint no decida la mezcla
            cand = random.sample(cand, SEED_MAX_PER_TIER)
            print(f"  {tier}: recortado a {SEED_MAX_PER_TIER} semillas", flush=True)
        seeds += cand

    # Barajamos para no vaciar EMERALD antes de tocar MASTER: si el crawl se corta
    # a medio camino, la banda sigue repartida y no sesgada hacia un tier.
    random.shuffle(seeds)

    added = 0
    for puuid, tier, division in seeds:
        if puuid not in seen_puuids:
            seen_puuids.add(puuid)
            player_queue.append((puuid, tier, division))
            added += 1
    return added

def load_ranks():
    """match_id -> (tier, division) del jugador semilla, de RANKS_CSV.

    Es la BANDA DE ELO de la partida. Aproximacion honesta: la etiquetamos con el
    rango del jugador por el que llegamos a ella, y el matchmaking de soloQ hace
    que los otros 9 anden cerca. Lo usan crawl() (para saber cuantas lleva) y
    build_features (para emitir la columna `tier`).
    """
    if not os.path.exists(RANKS_CSV):
        return {}
    with open(RANKS_CSV, encoding="utf-8") as f:
        return {r["match_id"]: (r["tier"], r["division"]) for r in csv.DictReader(f)}

def crawl():
    # El objetivo cuenta SOLO partidas de la banda objetivo, no todo lo que hay en
    # disco: riot_dataset/matches/ ya trae un crawl viejo (Challenger+GM) y,
    # contandolo, el crawl terminaria al instante sin bajar nada. seen_match_ids se
    # sigue usando para no redescargar, pero no para decidir cuando parar.
    en_banda = len(load_ranks())
    nuevo = not os.path.exists(RANKS_CSV)

    with open(RANKS_CSV, "a", newline="", encoding="utf-8") as fr:
        w = csv.writer(fr)
        if nuevo:
            w.writerow(["match_id", "tier", "division"])

        while player_queue and en_banda < TARGET_MATCHES:
            puuid, tier, division = player_queue.popleft()

            try:
                match_ids = get_match_ids(puuid, start=0, count=MATCHES_PER_PLAYER)
            except Exception:
                continue

            random.shuffle(match_ids)

            for match_id in match_ids:
                if match_id in seen_match_ids:
                    continue

                try:
                    match = get_match(match_id)
                    timeline = get_timeline(match_id)
                except Exception:
                    continue

                # El timeline PRIMERO: load_existing() da una partida por hecha si
                # existe su match JSON, asi que guardarlo el ultimo hace que
                # "match en disco" implique "timeline en disco". Si petamos entre
                # medias, el match no esta y se reintenta entero.
                save_timeline(match_id, timeline)
                save_match(match)
                w.writerow([match_id, tier, division])
                fr.flush()

                seen_match_ids.add(match_id)
                en_banda += 1
                print(f"[{en_banda}/{TARGET_MATCHES}] {match_id} ({tier} {division})"
                      f"  (cola: {len(player_queue)} jugadores)", flush=True)

                if en_banda % SAVE_EVERY == 0:
                    with open(os.path.join(OUT_DIR, "progress.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "matches_en_banda": en_banda,
                            "target": TARGET_MATCHES,
                            "queued_players": len(player_queue),
                            "seen_players": len(seen_puuids),
                            "matches_en_disco": len(seen_match_ids),
                        }, f)

                if en_banda >= TARGET_MATCHES:
                    break

if __name__ == "__main__":
    load_existing()
    con_rango = load_ranks()
    print(f"Partidas ya en disco: {len(seen_match_ids)} "
          f"({len(con_rango)} con procedencia de elo registrada)", flush=True)
    banda = ", ".join(f"{t} {'/'.join(d) if d else '(sin divisiones)'}"
                      for t, d in SEED_TIERS.items())
    print(f"Sembrando en: {banda}  (cola {QUEUE}, id {QUEUE_ID})", flush=True)
    n = seed_from_tiers()
    print(f"Sembrados {n} jugadores de la banda objetivo", flush=True)
    crawl()
    print(f"Total partidas unicas: {len(seen_match_ids)}", flush=True)