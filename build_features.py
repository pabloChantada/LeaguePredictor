"""
Construye la tabla de features TEMPORAL (una fila por minuto) para la baseline.

Lee los match JSON de riot_dataset/matches/, descarga (y cachea) el timeline de
cada uno y, para cada minuto de la partida, escribe una fila con el estado del
juego a ese minuto (features en diferencia, perspectiva azul), el minuto como
feature, deltas de momentum (cambio en los ultimos DELTA_WINDOW min) y el target
blue_win. El match_id se guarda para poder partir train/test POR PARTIDA.

Reutiliza la key + el limitador de crawler.py. Reanudable: los timelines se cachean
en disco, asi que relanzarlo no vuelve a descargar lo ya bajado.
"""
import os
import sys
import csv
import json

import crawler  # solo por las rutas del dataset (RAW_DIR / TIMELINE_DIR / RANKS_CSV)

MINUTE_START = 5    # primer minuto que emitimos (antes es demasiado ruido)
MINUTE_STEP = 1     # cada cuantos minutos emitimos una fila
DELTA_WINDOW = 5    # ventana para las features de momentum (cambio en N min)

RAW_DIR = crawler.RAW_DIR
OUT_DIR = crawler.OUT_DIR
TIMELINE_DIR = crawler.TIMELINE_DIR   # lo llena crawler.py; aqui solo se lee
FEATURES_CSV = os.path.join(OUT_DIR, "features.csv")

# tipos de monstruo elite (monsterType en ELITE_MONSTER_KILL)
DRAGON = "DRAGON"
HERALD = "RIFTHERALD"
BARON = "BARON_NASHOR"
GRUB = "HORDE"  # void grubs
# tipos de edificio (buildingType en BUILDING_KILL)
TOWER = "TOWER_BUILDING"
INHIB = "INHIBITOR_BUILDING"

# roles: teamPosition de Riot -> clave corta (orden fijo para columnas estables)
ROLE_MAP = {"TOP": "TOP", "JUNGLE": "JGL", "MIDDLE": "MID",
            "BOTTOM": "BOT", "UTILITY": "SUP"}
ROLES = ["TOP", "JGL", "MID", "BOT", "SUP"]

# TODO(escalado): meter identidad de campeon via escalado de comp (scaling_diff).
# Scaffold ya listo en make_scaling_table.py + champion_scaling.csv (a curar a mano).
# Descartado de momento: con valores placeholder no movia el AUC. Reconsiderar solo
# si el modelo se estanca; probar entonces la interaccion scaling_diff * minute.

# de estas calculamos delta de momentum.
# cs/level incluidos porque SI son calculables en vivo (Live Client Data da
# creepScore y level; oro y xp exactos NO) -> permiten un modelo live-compatible.
DELTA_BASE = ["gold_diff", "xp_diff", "kills_diff", "cs_diff", "level_diff"]


def get_timeline(match_id):
    """Timeline del match desde disco. None si no esta.

    NO descarga: bajarlos es tarea de crawler.py, que guarda match y timeline en
    la misma pasada. Asi este script es puro disco -> CSV: determinista,
    reejecutable y sin depender de la API key ni del rate limiter.
    """
    path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# La banda de elo la registra el crawler (es quien sabe de que tier salio cada
# semilla); aqui solo se lee. Las partidas de crawls antiguos no tienen fila y
# salen como UNKNOWN -> train.load_dataset las descarta.
load_ranks = crawler.load_ranks


def snapshot(pid_team, role_pairs, frames, minute):
    """Estado del juego (features en diferencia) al minuto dado.

    role_pairs: {clave_rol: {"blue": pid, "red": pid}} para los diffs por posicion.
    """
    pf_by_pid = {int(k): v for k, v in frames[minute]["participantFrames"].items()}

    # --- economia por equipo en el frame del minuto (oro / xp / cs) ---
    blue_gold = red_gold = blue_xp = red_xp = blue_cs = red_cs = 0
    blue_level = red_level = 0  # nivel: proxy de xp que SI existe en vivo
    # championStats/damageStats. OJO: medidos en el modelo dieron -0.001 de AUC
    # (parecian ortogonales al oro, pero no lo son al set completo: correlacionan
    # con level/xp/tiempo). NO entran a FEATURES en train.py; se siguen
    # calculando solo para tenerlos disponibles en el EDA. Ver TODO alli.
    team_extra = {100: {}, 200: {}}
    for pid, pf in pf_by_pid.items():
        team = pid_team.get(pid)
        cs = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
        if team == 100:
            blue_gold += pf.get("totalGold", 0); blue_xp += pf.get("xp", 0); blue_cs += cs
            blue_level += pf.get("level", 0)
        elif team == 200:
            red_gold += pf.get("totalGold", 0); red_xp += pf.get("xp", 0); red_cs += cs
            red_level += pf.get("level", 0)
        if team in team_extra:
            cst, dst = pf.get("championStats", {}), pf.get("damageStats", {})
            for key, val in (("health_max", cst.get("healthMax", 0)),
                             ("armor", cst.get("armor", 0)),
                             ("attack_dmg", cst.get("attackDamage", 0)),
                             ("dmg_taken", dst.get("totalDamageTaken", 0)),
                             ("dmg_champs", dst.get("totalDamageDoneToChampions", 0))):
                team_extra[team][key] = team_extra[team].get(key, 0) + val

    # --- diffs por rol (oro/xp): cuanto mas farmeado va NUESTRO rol que el suyo ---
    role_feats = {}
    for rk in ROLES:
        pair = role_pairs.get(rk, {})
        b, r = pair.get("blue"), pair.get("red")
        if b in pf_by_pid and r in pf_by_pid:
            role_feats[f"gold_diff_{rk}"] = pf_by_pid[b].get("totalGold", 0) - pf_by_pid[r].get("totalGold", 0)
            role_feats[f"xp_diff_{rk}"] = pf_by_pid[b].get("xp", 0) - pf_by_pid[r].get("xp", 0)
        else:  # rol sin asignar (autofill/remake) -> 0
            role_feats[f"gold_diff_{rk}"] = 0
            role_feats[f"xp_diff_{rk}"] = 0

    # --- eventos acumulados hasta el minuto (kills / estructuras / objetivos) ---
    blue_kills = red_kills = blue_towers = red_towers = blue_inhibs = red_inhibs = 0
    blue_drakes = red_drakes = blue_heralds = red_heralds = 0
    blue_barons = red_barons = blue_grubs = red_grubs = 0
    for fr in frames[: minute + 1]:
        for e in fr["events"]:
            etype = e["type"]
            if etype == "CHAMPION_KILL":
                team = pid_team.get(e.get("killerId"))
                if team == 100: blue_kills += 1
                elif team == 200: red_kills += 1
            elif etype == "BUILDING_KILL":
                owner = e.get("teamId")  # equipo dueño -> la derriba el contrario
                btype = e.get("buildingType")
                if btype == TOWER:
                    if owner == 200: blue_towers += 1
                    elif owner == 100: red_towers += 1
                elif btype == INHIB:
                    if owner == 200: blue_inhibs += 1
                    elif owner == 100: red_inhibs += 1
            elif etype == "ELITE_MONSTER_KILL":
                team = e.get("killerTeamId")
                monster = e.get("monsterType")
                if monster == DRAGON:
                    if team == 100: blue_drakes += 1
                    elif team == 200: red_drakes += 1
                elif monster == HERALD:
                    if team == 100: blue_heralds += 1
                    elif team == 200: red_heralds += 1
                elif monster == BARON:
                    if team == 100: blue_barons += 1
                    elif team == 200: red_barons += 1
                elif monster == GRUB:
                    if team == 100: blue_grubs += 1
                    elif team == 200: red_grubs += 1

    feats = {
        "gold_diff": blue_gold - red_gold,
        "xp_diff": blue_xp - red_xp,
        "cs_diff": blue_cs - red_cs,
        "level_diff": blue_level - red_level,
        "kills_diff": blue_kills - red_kills,
        "tower_diff": blue_towers - red_towers,
        "inhib_diff": blue_inhibs - red_inhibs,
        "dragon_diff": blue_drakes - red_drakes,
        "herald_diff": blue_heralds - red_heralds,
        "baron_diff": blue_barons - red_barons,
        "grub_diff": blue_grubs - red_grubs,
    }
    for key in ("health_max", "armor", "attack_dmg", "dmg_taken", "dmg_champs"):
        feats[f"{key}_diff"] = team_extra[100].get(key, 0) - team_extra[200].get(key, 0)
    feats.update(role_feats)
    return feats


def rows_for_match(match, timeline, rango=("UNKNOWN", "")):
    """Una lista de filas (una por minuto) para una partida, con deltas de momentum."""
    info = match["info"]
    pid_team = {p["participantId"]: p["teamId"] for p in info["participants"]}
    blue_win = 1 if any(t["teamId"] == 100 and t["win"] for t in info["teams"]) else 0
    match_id = match["metadata"]["matchId"]
    # parche mayor (p.ej. "16.13") para poder cortar por version en el EDA
    patch = ".".join(info.get("gameVersion", "").split(".")[:2])
    # Cola de la partida. IMPRESCINDIBLE: el crawler baja TODAS las colas de cada
    # jugador (match-v5 by-puuid/ids no filtra), asi que aqui entran Arena, co-op
    # vs IA, ARAM... que NO son el problema que modelamos. train.load_dataset()
    # filtra por esto. Ver QUEUE_SOLOQ en train.py.
    queue_id = info.get("queueId")

    # mapa rol -> {"blue": pid, "red": pid} desde teamPosition
    role_pairs = {}
    for p in info["participants"]:
        rk = ROLE_MAP.get(p.get("teamPosition"))
        if not rk:
            continue
        side = "blue" if p["teamId"] == 100 else "red"
        role_pairs.setdefault(rk, {})[side] = p["participantId"]

    frames = timeline["info"]["frames"]
    last = len(frames) - 1
    if last < MINUTE_START:
        return []  # partida mas corta que el primer minuto que emitimos

    # snapshot por minuto, indexado por minuto para calcular deltas
    snaps = {m: snapshot(pid_team, role_pairs, frames, m)
             for m in range(MINUTE_START, last + 1, MINUTE_STEP)}

    rows = []
    for m, snap in snaps.items():
        row = {"match_id": match_id, "queue_id": queue_id,
               "tier": rango[0], "division": rango[1], "patch": patch,
               "minute": m, "blue_win": blue_win}
        row.update(snap)
        prev = snaps.get(m - DELTA_WINDOW)
        for col in DELTA_BASE:
            row[f"{col}_d{DELTA_WINDOW}"] = (snap[col] - prev[col]) if prev else 0
        rows.append(row)
    return rows


def build():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    ranks = load_ranks()
    print(f"{len(files)} matches en disco, {len(ranks)} con banda de elo registrada")
    all_rows = []
    games_ok = games_short = games_sin_timeline = 0
    for i, fn in enumerate(files, 1):
        match_id = fn[:-5]
        try:
            with open(os.path.join(RAW_DIR, fn), encoding="utf-8") as f:
                match = json.load(f)
            timeline = get_timeline(match_id)
            if timeline is None:
                games_sin_timeline += 1
                continue  # sin timeline en disco -> relanza crawler.py
            rows = rows_for_match(match, timeline, ranks.get(match_id, ("UNKNOWN", "")))
        except Exception as ex:
            print(f"  ! {match_id}: {ex}", flush=True)
            continue
        if not rows:
            games_short += 1
            continue
        all_rows.extend(rows)
        games_ok += 1
        if i % 25 == 0:
            print(f"[{i}/{len(files)}] partidas -> {len(all_rows)} filas", flush=True)

    if not all_rows:
        print("No hay filas. ¿Ya descargaste partidas con crawler.py?")
        return

    fieldnames = list(all_rows[0].keys())
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_blue = sum(r["blue_win"] for r in all_rows)
    print(f"\nEscrito {FEATURES_CSV}")
    print(f"  partidas validas: {games_ok}  (cortas: {games_short}, "
          f"sin timeline: {games_sin_timeline})")
    print(f"  filas (minuto x partida): {len(all_rows)}")
    print(f"  balance blue_win: {n_blue}/{len(all_rows)} = {n_blue/len(all_rows):.3f}")
    por_tier = {}
    for r in all_rows:
        por_tier[r["tier"]] = por_tier.get(r["tier"], 0) + 1
    print("  filas por banda de elo:")
    for t, n in sorted(por_tier.items(), key=lambda x: -x[1]):
        print(f"    {t:>10}: {n:>7}")


if __name__ == "__main__":
    build()
