"""Username extraction and persistable username→caption map.

extract_username strips the extension and an optional leading NN- prefix,
then isolates a handle via date, unix-timestamp, space, or media-token delimiters.
An optional numeric post/status id between handle and date is skipped.
Handles may start with a digit or underscore but must contain a letter. Returns
None if isolation is unclear (IMG_1234, image0, no delimiter).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from photo_scanner.paths import USERNAME_CAPTIONS_PATH

_GENERIC = frozenset({"img", "image", "dsc", "dscn", "photo", "pxl", "screenshot"})


_HANDLE = r"[A-Za-z0-9_][A-Za-z0-9_.]*?"
# optional post/status id between handle and date: -123, _-123, -123_1
_ID_BEFORE_DATE = r"(?:[_-]-?\d+(?:_\d+)?)?"


def _is_media_token(token: str) -> bool:
    """Twitter-ish media id: long, has a letter, and mixed case or a digit."""
    if len(token) < 10:
        return False
    if not re.search(r"[A-Za-z]", token):
        return False
    mixed = re.search(r"[A-Z]", token) and re.search(r"[a-z]", token)
    return bool(mixed or re.search(r"\d", token))


def extract_username(filename: str) -> str | None:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+-", "", stem)

    m = re.match(rf"^({_HANDLE}){_ID_BEFORE_DATE}-((?:19|20)\d{{6}})(?:[_-]|$)", stem)
    if m:
        user = _valid_username(m.group(1))
        if user:
            return user

    m = re.match(rf"^({_HANDLE})_((?:19|20)\d{{6}})_", stem)
    if m:
        user = _valid_username(m.group(1))
        if user:
            return user

    # "dear.zia ClipDown.App_…" / "dear.zia 455141170_…"
    m = re.match(r"^(\S+)\s+", stem)
    if m:
        user = _valid_username(m.group(1))
        if user:
            return user

    # unix-ish timestamp: handle_1733585670_…
    m = re.match(rf"^({_HANDLE})_(\d{{10,}})(?:_|$)", stem)
    if m:
        user = _valid_username(m.group(1))
        if user:
            return user

    # twitter media token (alnum / _ / -), optional _N frame suffix; `_+` separator
    m = re.match(r"^(.+)_+([A-Za-z0-9_-]{10,}?)(?:_\d+)?$", stem)
    if m and _is_media_token(m.group(2)):
        user = _valid_username(m.group(1).rstrip("_"))
        if user:
            return user

    return None


def _valid_username(token: str) -> str | None:
    if not token or token.lower() in _GENERIC:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.]*", token):
        return None
    if not re.search(r"[A-Za-z]", token):
        return None
    return token


def is_learnable_caption(caption: str) -> bool:
    if not caption or not caption.strip():
        return False
    if "Author:" in caption or "Source:" in caption:
        return False
    return True


def normalize_caption(caption: str) -> str:
    """Keep only the primary label; drop `;`-separated extras."""
    return caption.split(";", 1)[0].strip()


def pick_caption(captions: set[str]) -> str | None:
    """Return one caption if unique, or the sole most-specific label.

    More specific means a longer label that starts with a shorter one plus a
    space (`aespa` + `aespa giselle` → `aespa giselle`). Sibling labels
    (`aespa giselle` vs `aespa karina`) stay unresolved.
    """
    if not captions:
        return None
    if len(captions) == 1:
        return next(iter(captions))
    maxima = [
        c
        for c in captions
        if not any(
            other != c and (other == c or other.startswith(c + " "))
            for other in captions
        )
    ]
    if len(maxima) == 1:
        return maxima[0]
    return None


def majority_caption(counts: Counter[str]) -> str | None:
    """Most common caption when it strictly outnumbers all others combined."""
    if not counts:
        return None
    top, top_n = counts.most_common(1)[0]
    if top_n > sum(counts.values()) - top_n:
        return top
    return None


def choose_caption(counts: Counter[str]) -> str | None:
    """Prefer an overwhelming majority; else a unique most-specific label."""
    return majority_caption(counts) or pick_caption(set(counts))


def load_map(path: Path | None = None) -> dict[str, str]:
    path = path or USERNAME_CAPTIONS_PATH
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        username, caption = line.split("=", 1)
        username = username.strip()
        if username:
            mapping[username] = caption.strip()
    return mapping


def save_map(mapping: dict[str, str], path: Path | None = None) -> None:
    path = path or USERNAME_CAPTIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{username}={caption}" for username, caption in sorted(mapping.items())]
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


@dataclass
class LearnResult:
    mapping: dict[str, str]  # username -> caption (only unambiguous)
    conflicts: dict[str, set[str]]  # username -> set of disagreeing captions
    unmapped_usernames: set[str]  # usernames that appeared but had no learnable teacher


def learn_from_teachers(
    teachers: list[tuple[str, str]],
    existing: dict[str, str] | None = None,
) -> LearnResult:
    existing = dict(existing) if existing else {}
    seen: set[str] = set()
    captions: dict[str, Counter[str]] = defaultdict(Counter)

    for filename, caption in teachers:
        username = extract_username(filename)
        if username is None:
            continue
        seen.add(username)
        if is_learnable_caption(caption):
            primary = normalize_caption(caption)
            if primary:
                captions[username][primary] += 1

    mapping = dict(existing)
    conflicts: dict[str, set[str]] = {}
    unmapped: set[str] = set()
    for username in seen:
        if username in existing:
            continue
        found = captions.get(username, Counter())
        chosen = choose_caption(found)
        if chosen is not None:
            mapping[username] = chosen
        elif found:
            conflicts[username] = set(found)
        else:
            unmapped.add(username)
    return LearnResult(mapping, conflicts, unmapped)


def resolve_caption(username: str, mapping: dict[str, str]) -> str | None:
    return mapping.get(username)


if __name__ == "__main__":
    import tempfile

    assert extract_username("01-katarinabluu-20260828_154043-3408454469.jpg") == "katarinabluu"
    assert extract_username("katarinabluu_20260828_154043_abcdef12.jpg") == "katarinabluu"
    assert extract_username("GING_xjqluv_HQuLAjGaIAAXiTA.jpg") == "GING_xjqluv"
    assert extract_username("somibase_20260830_140144_7861bb47.jpg") == "somibase"
    assert extract_username("01-dear.zia-20260830_110002-134018694.jpg") == "dear.zia"
    assert extract_username("dear.zia ClipDown.App_468741966_n.jpeg") == "dear.zia"
    assert extract_username("dear.zia 455141170_1036762104485380_n.jpeg") == "dear.zia"
    assert extract_username("10ve.xx-20260829_211039-1011023314.jpg") == "10ve.xx"
    assert extract_username("10ve.xx_1733585670_3517894033408699610_8443250103.jpg") == "10ve.xx"
    assert extract_username("01-8t8ear-20260817_142403-902476507.jpg") == "8t8ear"
    assert extract_username("8t8ear_DceSMYeEpyx_2.jpg") == "8t8ear"
    assert extract_username("_IUofficial_HRnzB-rasAAEieq.jpg") == "_IUofficial"
    assert extract_username("_IUofficial_HRnzB_Ia4AAviMn.jpg") == "_IUofficial"
    assert extract_username("_IUofficial 2025-01-25T104229 1.jpeg") == "_IUofficial"
    assert extract_username("aespapic_HRq8p_oWgAAgwIf.jpg") == "aespapic"
    assert extract_username("01-__chappy___-20260907_151455-336796402.jpg") == "__chappy___"
    assert extract_username("__chappy___ 2025-01-06T132841-1.jpeg") == "__chappy___"
    assert extract_username("IMG_1234.jpg") is None
    assert extract_username("image0.jpg") is None
    assert extract_username("_uyis.c-3395296735-20260911_031425.jpg") == "_uyis.c"
    assert extract_username("minjeong.log_-822244289-20260911_104556.jpg") == "minjeong.log"
    assert extract_username("ourxche-3772705043-20260906_145639.jpg") == "ourxche"
    assert extract_username("01-hiiragi428-3983804215000436647_1-20260911_133459.jpg") == "hiiragi428"
    assert extract_username(
        "02-cocona_sakuragi_official-3983715551017112789_2-20260911_111504.jpg"
    ) == "cocona_sakuragi_official"
    assert extract_username("04-_chaechae_1-3983337933897388746_4-20260910_220844.jpg") == "_chaechae_1"
    assert extract_username("cosmopolitankorea-2691120700-20260911_090009.mp4") == "cosmopolitankorea"
    assert extract_username("ningfusion_HR3LK_EWEAQ17_7.jpg") == "ningfusion"
    assert extract_username("katarinea__bQDRbkXbnWc1vPkU.mp4") == "katarinea"

    assert is_learnable_caption("aespa karina")
    assert not is_learnable_caption("")
    assert not is_learnable_caption("   ")
    assert not is_learnable_caption("Author: someone")
    assert not is_learnable_caption("Source: twitter")
    assert not is_learnable_caption("aespa karina Author: x")
    assert is_learnable_caption("author: lowercase is learnable")

    learned = learn_from_teachers(
        [
            ("01-katarinabluu-20260828_154043-3408454469.jpg", "aespa karina"),
            ("katarinabluu_20260828_154043_abcdef12.jpg", "aespa karina"),
            ("somibase_20260830_140144_7861bb47.jpg", "Author: skip me"),
            ("GING_xjqluv_HQuLAjGaIAAXiTA.jpg", "Source: skip me"),
            ("IMG_1234.jpg", "aespa karina"),
        ]
    )
    assert learned.mapping == {"katarinabluu": "aespa karina"}
    assert learned.conflicts == {}
    assert learned.unmapped_usernames == {"somibase", "GING_xjqluv"}

    conflicted = learn_from_teachers(
        [
            ("katarinabluu_20260828_154043_abcdef12.jpg", "aespa karina"),
            ("katarinabluu_20260829_154043_abcdef13.jpg", "aespa winter"),
        ]
    )
    assert "katarinabluu" not in conflicted.mapping
    assert conflicted.conflicts["katarinabluu"] == {"aespa karina", "aespa winter"}

    assert normalize_caption("aespa giselle; aespa karina") == "aespa giselle"
    assert pick_caption({"aespa", "aespa giselle"}) == "aespa giselle"
    assert pick_caption({"aespa giselle", "aespa karina"}) is None

    multi = learn_from_teachers(
        [
            ("aerichandesu_20260828_154043_abcdef12.jpg", "aespa"),
            ("aerichandesu_20260828_154043_abcdef13.jpg", "aespa giselle"),
            ("aerichandesu_20260828_154043_abcdef14.jpg", "aespa giselle; aespa karina"),
            ("aerichandesu_20260828_154043_abcdef15.jpg", "aespa giselle; aespa ningning"),
            ("aerichandesu_20260828_154043_abcdef16.jpg", "aespa giselle; aespa ningnn"),
            ("aerichandesu_20260828_154043_abcdef17.jpg", "aespa giselle; aespa winter"),
            ("aerichandesu_20260828_154043_abcdef18.jpg", "aespa ningning"),
        ]
    )
    assert multi.mapping["aerichandesu"] == "aespa giselle"
    assert "aerichandesu" not in multi.conflicts

    assert majority_caption(Counter({"aespa giselle": 810, "aespa": 2, "aespa ningning": 1})) == "aespa giselle"
    assert majority_caption(Counter({"aespa giselle": 2, "aespa ningning": 2})) is None
    assert choose_caption(Counter({"aespa": 1, "aespa giselle": 1})) == "aespa giselle"

    kept = learn_from_teachers(
        [("katarinabluu_20260828_154043_abcdef12.jpg", "aespa winter")],
        existing={"katarinabluu": "aespa karina"},
    )
    assert kept.mapping["katarinabluu"] == "aespa karina"
    assert "katarinabluu" not in kept.conflicts

    mapping = {"katarinabluu": "aespa karina", "otheruser": "aespa karina"}
    assert resolve_caption("katarinabluu", mapping) == "aespa karina"
    assert resolve_caption("Katarinabluu", mapping) is None

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "username_captions.txt"
        save_map({"otheruser": "aespa karina", "katarinabluu": "a=b"}, path)
        assert load_map(path) == {"katarinabluu": "a=b", "otheruser": "aespa karina"}
        path.write_text("# comment\n\nkatarinabluu=aespa karina\n", encoding="utf-8")
        assert load_map(path) == {"katarinabluu": "aespa karina"}
        assert load_map(Path(tmp) / "missing.txt") == {}

    print("caption_map ok")
