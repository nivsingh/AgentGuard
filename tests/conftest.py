import json
import sqlite3

import pytest

from data import generate
from data.docs_util import load_docs


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    generate.main(out)
    return out


@pytest.fixture(scope="session")
def conn(built):
    c = sqlite3.connect(built / "agentguard.db")
    yield c
    c.close()


@pytest.fixture(scope="session")
def truths(built):
    return json.loads((built / "truths.json").read_text())


@pytest.fixture(scope="session")
def docs(built):
    return load_docs(built / "docs")
