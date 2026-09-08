"""Opt-in HTTP integration check: NER_TEST_BASE_URL must point to a running service."""

import os

import pytest
from scripts.smoke import run


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("NER_TEST_BASE_URL"), reason="NER_TEST_BASE_URL is not set")
def test_live_api():
    report = run(os.environ["NER_TEST_BASE_URL"], os.environ.get("NER_API_KEY"))
    assert report["ready"]["status"] == "ready"
    assert len(report["results"]) == 6
