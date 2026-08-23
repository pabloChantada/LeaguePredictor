import pytest
# Asumiendo que el archivo se llama live_predict_2.py según tu estructura
import src.serve.live_predict as lp

def test_is_new_game():
    """Prueba la lógica para detectar si ha empezado una nueva partida en base al tiempo."""
    # El tiempo avanza normalmente (misma partida)
    assert lp.is_new_game(100.0, 105.0) is False
    
    # Pequeño desajuste o lag de red (el reloj retrocede menos de 5 segundos)
    assert lp.is_new_game(100.0, 96.0) is False 
    
    # El tiempo retrocede drásticamente -> Nueva partida detectada
    assert lp.is_new_game(100.0, 10.0) is True
    
    # Primer ciclo de la aplicación (estado inicial)
    assert lp.is_new_game(None, 10.0) is False

def test_structure_owner():
    """Prueba que el nombre interno de la torreta asigne el equipo correcto."""
    # Formato nuevo de la API
    assert lp._structure_owner("Turret_TOrder_L0_P3") == lp.BLUE
    assert lp._structure_owner("Inhib_TChaos_L1_P1") == lp.RED
    # Formato antiguo tolerado por el código
    assert lp._structure_owner("Mid_T1_Tower") == lp.BLUE
    assert lp._structure_owner("Top_T2_Tower") == lp.RED
    # Elemento desconocido
    assert lp._structure_owner("Unknown_Structure") is None

def test_build_features():
    """Prueba que el momentum (diferencia en 5 minutos) se calcule bien."""
    # Estado actual en el minuto 10
    state = {
        "minute": 10,
        "kills_diff": 5,
        "cs_diff": 20,
        "level_diff": 2,
        "tower_diff": 1,
    }
    # Historial de minutos anteriores
    history = [
        {"minute": 4, "kills_diff": 0, "cs_diff": 0, "level_diff": 0},
        # El código buscará (10 - DELTA_WINDOW) = 5. Este es el 'prev'
        {"minute": 5, "kills_diff": 2, "cs_diff": 10, "level_diff": 1}, 
        {"minute": 6, "kills_diff": 3, "cs_diff": 15, "level_diff": 1}
    ]
    
    # Sobreescribimos momentáneamente DELTA_WINDOW por si acaso
    original_delta = lp.DELTA_WINDOW
    lp.DELTA_WINDOW = 5
    
    feats = lp.build_features(state, history)
    
    # Recuperamos valor original
    lp.DELTA_WINDOW = original_delta
    
    # d5 = estado actual (min 10) - estado previo (min 5)
    assert feats["kills_diff_d5"] == 5 - 2   # 3
    assert feats["cs_diff_d5"] == 20 - 10    # 10
    assert feats["level_diff_d5"] == 2 - 1   # 1
    # Nos aseguramos de que las absolutas se mantienen
    assert feats["tower_diff"] == 1