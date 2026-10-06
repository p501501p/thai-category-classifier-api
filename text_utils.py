from pythainlp.tokenize import word_tokenize


def thai_tokenize(text: str) -> list[str]:
    return word_tokenize(text, engine="newmm", keep_whitespace=False)