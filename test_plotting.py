import numpy as np
from utils.colors import BGRCuteColors
from plotting import Row, Col, Padding, TextRenderer


def test_basic_row_and_col_behaviour():

    inputs = {
        "a": np.zeros(shape=(640, 480), dtype=np.uint8),
        "b": np.zeros(shape=(1024, 768, 3), dtype=np.uint8),
        "c": np.zeros(shape=(2880, 1440), dtype=np.uint8)
    }

    assert Row("a", "b", "c").pack(inputs).size == (2880, 2688)
    assert Col("a", "b", "c").pack(inputs).size == (4544, 1440)
    assert Col(Row("a"), Row("b"), Row("c")).pack(inputs).size == (4544, 1440)
    assert Row(Col("a"), Col("b"), Col("c")).pack(inputs).size == (2880, 2688)
    assert Row(Col("a", Row("b")), "c").pack(inputs).size == (2880, 2208)


def test_integration_row_and_col():

    inputs = {
        "a": np.array(BGRCuteColors.CYAN, dtype=np.uint8) * np.ones(shape=(640, 480, 3), dtype=np.uint8),
        "b": np.array(BGRCuteColors.SALMON, dtype=np.uint8) * np.ones(shape=(100, 600, 3), dtype=np.uint8),
        "c": np.array(BGRCuteColors.CRIMSON, dtype=np.uint8) * np.ones(shape=(288, 244, 3), dtype=np.uint8),
        "d": np.array(BGRCuteColors.SUN_YELLOW, dtype=np.uint8) * np.ones(shape=(288, 244, 3), dtype=np.uint8)
    }
    Col(Row("a", "b"), Row("c", "d")).render(inputs)
    Col("a", "b", "c", "d").render(inputs)
    Row(Padding("a"), "b", Padding(Col("c", "d"))).render(inputs)


def test_integration_text_rendering():
    TextRenderer().render('heheh')
    TextRenderer().render('heheh \npoetry of the heart \n okekoekeoke')
