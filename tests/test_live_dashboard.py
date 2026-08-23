import src.serve.live_dashboard as ld


class TestClock:
    def test_formats_minutes_and_seconds(self):
        assert ld.clock(725) == "12:05"

    def test_pads_single_digit_seconds(self):
        assert ld.clock(63) == "1:03"

    def test_zero(self):
        assert ld.clock(0) == "0:00"


class TestPalette:
    def test_defaults_to_light_without_a_theme(self):
        # st.context.theme is unavailable outside a real Streamlit session.
        assert ld.palette() == ld.LIGHT
