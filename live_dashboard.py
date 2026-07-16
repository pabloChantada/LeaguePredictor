"""
Dashboard en vivo del win%.

    streamlit run live_dashboard.py

Lee la partida en curso (Live Client Data API), predice cada POLL_SECONDS y
dibuja la curva de probabilidad segun avanza la partida.

Funciona en cualquier modo (ranked, normal, CUSTOM, practice tool) mientras la
partida corra EN ESTE PC. Ojo: el modelo se entreno con soloQ 5v5 de alto elo,
asi que en una custom con bots el numero no significa gran cosa — sirve para
validar que la tuberia funciona.
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import live_predict as lp

st.set_page_config(page_title="LoL Win Probability", layout="wide")

# Etiqueta legible de cada contador de lp.COUNTERS, en orden de marcador.
ETIQUETAS = {
    "kills": "Asesinatos", "cs": "CS", "level": "Nivel (suma)",
    "towers": "Torres", "inhibs": "Inhibidores", "dragons": "Dragones",
    "heralds": "Heraldos", "barons": "Barones", "grubs": "Larvas del Vacio",
}

# Paleta validada con el validador de la guia de dataviz (ver README).
# El par azul/rojo separa ΔE 21.6 en protanopia (umbral 8) -> legible con
# daltonismo. El verde/rojo anterior daba ΔE 4.2: un daltonico rojo-verde no
# distinguia "ganando" de "perdiendo". Los pasos oscuros son los de la guia para
# superficie oscura, NO un volteo automatico del claro.
CLARO = {"azul": "#2a78d6", "rojo": "#e34948", "tinta": "#0b0b0b",
         "muted": "#898781", "grid": "#e1e0d9", "base": "#c3c2b7"}
OSCURO = {"azul": "#3987e5", "rojo": "#e66767", "tinta": "#ffffff",
          "muted": "#898781", "grid": "#2c2c2a", "base": "#383835"}


def paleta():
    # st.context.theme es el tema que el navegador ACABA renderizando.
    # theme.base NO vale: lee la config, que vale None si no la has puesto, asi
    # que la grafica salia con la paleta clara dentro de un tema oscuro (linea y
    # etiqueta negras sobre fondo negro, rejilla blanca).
    tema = getattr(st.context, "theme", None)
    return OSCURO if tema is not None and tema.type == "dark" else CLARO


@st.cache_resource
def cargar_modelo():
    return lp.load_model()


def reinicia_partida():
    """Deja el estado de sesion como al arrancar: cada partida empieza limpia."""
    st.session_state.hist = []      # estados por minuto (para el momentum)
    st.session_state.curva = []     # (minuto_float, p_blue) para la grafica
    st.session_state.last_t = None  # gameTime del ultimo poll (detecta partida nueva)


def reloj(segundos):
    return f"{int(segundos // 60)}:{int(segundos % 60):02d}"


def pinta_curva(curva, mi_equipo, col):
    """Win% de TU equipo a lo largo de la partida.

    El relleno usa los colores de EQUIPO, no "verde=bien/rojo=mal": el area se
    pinta del color del bando que va por delante. Asi el color siempre significa
    lo mismo aqui y en el marcador (azul=equipo azul), y la altura sigue siendo
    TU probabilidad, sea cual sea tu bando.
    """
    df = pd.DataFrame(curva, columns=["t", "p_blue"])
    p = (df.p_blue if mi_equipo == lp.BLUE else 1 - df.p_blue) * 100
    mio = col["azul"] if mi_equipo == lp.BLUE else col["rojo"]
    suyo = col["rojo"] if mi_equipo == lp.BLUE else col["azul"]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_alpha(0)              # se adapta al fondo de Streamlit
    ax.set_facecolor("none")

    # relleno divergente respecto al 50%: el color dice de quien es la ventaja
    ax.fill_between(df.t, 50, p, where=(p >= 50), interpolate=True, alpha=.18, color=mio, lw=0)
    ax.fill_between(df.t, 50, p, where=(p < 50), interpolate=True, alpha=.18, color=suyo, lw=0)
    ax.plot(df.t, p, lw=2, color=col["tinta"], solid_capstyle="round")
    ax.axhline(50, color=col["base"], lw=1)   # hairline solida, nunca discontinua

    # etiqueta directa solo en el punto final (nunca un numero en cada punto)
    t_fin, p_fin = df.t.iloc[-1], p.iloc[-1]
    ax.scatter([t_fin], [p_fin], s=90, color=mio, zorder=3,
               edgecolor=col.get("anillo", "none"), linewidth=2)
    ax.annotate(f"{p_fin:.0f}%", (t_fin, p_fin), textcoords="offset points",
                xytext=(10, 0), va="center", fontsize=12, fontweight="bold",
                color=col["tinta"])

    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.margins(x=.08)
    ax.set_xlabel("minuto", color=col["muted"], fontsize=9)
    ax.set_ylabel("tu prob. de ganar (%)", color=col["muted"], fontsize=9)
    ax.grid(axis="y", color=col["grid"], lw=1, alpha=.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=col["muted"], labelsize=9)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(col["base"])
    fig.tight_layout()
    return fig


def pinta_marcador(cont, col):
    """Marcador azul vs rojo con los absolutos de cada bando."""
    filas = []
    for k in lp.COUNTERS:
        a, r = cont[k][lp.BLUE], cont[k][lp.RED]
        filas.append(f"""<tr>
            <td style="text-align:right;font-weight:600;width:38%">{a}</td>
            <td style="text-align:center;color:{col['muted']};font-size:.85em">{ETIQUETAS[k]}</td>
            <td style="text-align:left;font-weight:600;width:38%">{r}</td></tr>""")
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse">
      <tr>
        <th style="text-align:right;color:{col['azul']}">AZUL</th>
        <th></th>
        <th style="text-align:left;color:{col['rojo']}">ROJO</th>
      </tr>
      {''.join(filas)}
    </table>""", unsafe_allow_html=True)


def main():
    st.title("Probabilidad de victoria en vivo")
    model, names = cargar_modelo()
    st.caption(f"Modelo: {len(names)} features, AUC 0.836, ECE 1.1% (soloQ) · "
               f"refresco cada {lp.POLL_SECONDS}s")

    if "hist" not in st.session_state:
        reinicia_partida()

    hueco = st.empty()
    col = paleta()

    while True:
        try:
            data = lp.fetch_live_data()
        except lp.NoGameRunning as ex:
            with hueco.container():
                st.info("**Sin partida en curso.** Entra a una partida —vale una "
                        "**custom** o el **practice tool**— y esto arrancara solo.")
                st.caption(f"Detalle: {ex}")
            time.sleep(lp.POLL_SECONDS)
            continue

        # Partida nueva -> fuera la curva y el historial de la anterior. No basta
        # con que se vea mal: los _d5 compararian contra la partida previa.
        t = lp.game_time(data)
        if lp.is_new_game(st.session_state.last_t, t):
            reinicia_partida()
        st.session_state.last_t = t

        estado = lp.read_state(data)
        # un estado por minuto: es lo que espera el calculo de momentum
        if not st.session_state.hist or st.session_state.hist[-1]["minute"] != estado["minute"]:
            st.session_state.hist.append(estado)

        feats = lp.build_features(estado, st.session_state.hist)
        p_blue = lp.predict_blue_winrate(model, feats, names)
        mi_equipo = lp.active_team(data)
        p_mio = p_blue if mi_equipo == lp.BLUE else 1 - p_blue

        # x = gameTime real, no el minuto entero: con un poll cada 10s el minuto
        # entero apilaba ~6 puntos en la misma x y la curva salia a escalones.
        st.session_state.curva.append((t / 60, p_blue))

        with hueco.container():
            # Un solo numero grande: el win% de TU equipo. El del azul no va como
            # tile aparte — la curva ya lo dice, y con contexto.
            lado = "azul" if mi_equipo == lp.BLUE else "rojo"
            c1, c2 = st.columns([2, 1])
            # tendencia: cuanto se ha movido TU win% en los ultimos 5 min
            hace5 = [q for tt, q in st.session_state.curva if tt <= t / 60 - 5]
            if hace5:
                antes = hace5[-1] if mi_equipo == lp.BLUE else 1 - hace5[-1]
                delta = f"{(p_mio - antes) * 100:+.1f} pts en 5 min"
            else:
                delta = None
            c1.metric(f"Tu equipo ({lado})", f"{p_mio:.1%}", delta=delta)
            c2.metric("Tiempo", reloj(t))

            g, m = st.columns([3, 1])
            with g:
                if len(st.session_state.curva) > 1:
                    st.pyplot(pinta_curva(st.session_state.curva, mi_equipo, col))
                    # Nota discreta, no una caja de alarma: es una advertencia sobre
                    # como leer el numero, no un problema que haya que atender.
                    if estado["minute"] > 25:
                        st.caption("Minuto >25: las partidas largas están igualadas "
                                   "*por eso* duran — fíate menos del número.")
            with m:
                pinta_marcador(lp.read_counters(data), col)

            # gemelo en tabla: todo valor de la grafica es legible sin color
            with st.expander("Vector de features (lo que ve el modelo)"):
                st.dataframe(pd.DataFrame([{k: feats[k] for k in names}]),
                             hide_index=True, use_container_width=True)
                st.dataframe(pd.DataFrame(st.session_state.curva,
                                          columns=["minuto", "p_azul"]),
                             hide_index=True, use_container_width=True, height=200)

        time.sleep(lp.POLL_SECONDS)


if __name__ == "__main__":
    main()
