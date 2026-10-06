import os

if os.environ.get("VERCEL") == "1":
    os.environ.setdefault("PYTHAINLP_DATA", "/tmp/pythainlp-data")

from pythainlp.tokenize import word_tokenize


def thai_tokenize(text: str) -> list[str]:
    return word_tokenize(text, engine="newmm", keep_whitespace=False)