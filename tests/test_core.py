"""Tests for Neuroevo."""
from src.core import Neuroevo
def test_init(): assert Neuroevo().get_stats()["ops"] == 0
def test_op(): c = Neuroevo(); c.search(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Neuroevo(); [c.search() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Neuroevo(); c.search(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Neuroevo(); r = c.search(); assert r["service"] == "neuroevo"
