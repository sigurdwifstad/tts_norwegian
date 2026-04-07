"""Unit tests for the Norwegian number-to-words conversion in text_normalizer."""
from dataset import text_normalizer, _number_to_norwegian, _year_to_norwegian


def test_single_digits():
    assert _number_to_norwegian(0) == "null"
    assert _number_to_norwegian(1) == "en"
    assert _number_to_norwegian(5) == "fem"
    assert _number_to_norwegian(9) == "ni"


def test_teens():
    assert _number_to_norwegian(10) == "ti"
    assert _number_to_norwegian(11) == "elleve"
    assert _number_to_norwegian(12) == "tolv"
    assert _number_to_norwegian(13) == "tretten"
    assert _number_to_norwegian(19) == "nitten"


def test_tens():
    assert _number_to_norwegian(20) == "tjue"
    assert _number_to_norwegian(30) == "tretti"
    assert _number_to_norwegian(40) == "foerti"
    assert _number_to_norwegian(44) == "foertifire"
    assert _number_to_norwegian(50) == "femti"
    assert _number_to_norwegian(69) == "sekstini"
    assert _number_to_norwegian(99) == "nittini"


def test_hundreds():
    assert _number_to_norwegian(100) == "hundre"
    assert _number_to_norwegian(101) == "hundre og en"
    assert _number_to_norwegian(110) == "hundre og ti"
    assert _number_to_norwegian(200) == "to hundre"
    assert _number_to_norwegian(569) == "fem hundre og sekstini"
    assert _number_to_norwegian(999) == "ni hundre og nittini"
    assert _number_to_norwegian(315) == "tre hundre og femten"


def test_years_special():
    assert _year_to_norwegian(2000) == "to tusen"
    assert _year_to_norwegian(2001) == "to tusen og en"
    assert _year_to_norwegian(2005) == "to tusen og fem"
    assert _year_to_norwegian(2009) == "to tusen og ni"


def test_years_split():
    assert _year_to_norwegian(1996) == "nitten nittiseks"
    assert _year_to_norwegian(1900) == "nitten hundre"
    assert _year_to_norwegian(1814) == "atten fjorten"
    assert _year_to_norwegian(2010) == "tjue ti"
    assert _year_to_norwegian(2024) == "tjue tjuefire"
    assert _year_to_norwegian(2099) == "tjue nittini"
    assert _year_to_norwegian(1066) == "ti sekstiseks"


def test_normalizer_inline():
    assert "nitten nittiseks" in text_normalizer("i 1996 var")
    assert "to tusen og en" in text_normalizer("aar 2001")
    assert "foertifire" in text_normalizer("det er 44 stykker")
    assert "fem hundre og sekstini" in text_normalizer("det koster 569 kroner")


def test_normalizer_preserves_pause_tokens():
    result = text_normalizer("hei <sil> 42 verden")
    assert "<sil>" in result
    assert "foerti" in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

