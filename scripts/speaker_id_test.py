"""Unit tests for src/speaker_id.py — the pure logic (registry, identify,
enroll averaging, intent regex) with the Resemblyzer encoder MOCKED, so the
suite is fast + deterministic and needs no torch/audio (the discipline the
other *_test.py suites use for heavy deps)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.speaker_id as sid  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


# --- Mock the encoder so embed()/_ensure_imported() don't touch resemblyzer.
# A pre-set _encoder makes _ensure_imported() short-circuit True; we then
# replace the module-level embed() with a controllable fake.
sid._encoder = object()
sid.AVAILABLE = True

_DIM = sid._EMBED_DIM  # 256


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


# Canonical embeddings in 256-d. uA, uB orthogonal (cosine 0).
uA = _unit(np.eye(_DIM)[0])
uB = _unit(np.eye(_DIM)[1])
# A probe with cosine 0.8 to uA and 0.1 to uB (0.8^2+0.1^2+0.5916^2 ≈ 1).
probe_A = np.array([0.8, 0.1] + [0.0] * (_DIM - 3) + [np.sqrt(0.35)], dtype=np.float32)

# fake embed: key by the FIRST sample of the passed array so each test can
# request a specific embedding deterministically. Returns None for key 0
# (simulates an unusable / too-quiet clip).
_VEC_BY_KEY = {1: uA, 2: uB, 3: probe_A}


def _fake_embed(audio: np.ndarray):
    if audio is None or audio.size == 0:
        return None
    key = int(audio.flat[0])
    if key == 0:
        return None  # unusable clip
    return _VEC_BY_KEY.get(key, uA)


sid.embed = _fake_embed


def _clip(key: int) -> np.ndarray:
    return np.full(8, key, dtype=np.int16)


# --- Test 1: _slug ---------------------------------------------------------
check("_slug lowercases + hyphenates", sid._slug("My Aunt") == "my-aunt")
check("_slug strips junk", sid._slug("  Saul!! ") == "saul")
check("_slug empty -> 'speaker'", sid._slug("") == "speaker")


# --- Test 2: matches_enroll_intent ----------------------------------------
check("intent: 'enroll my voice'", sid.matches_enroll_intent("Jarvis, enroll my voice"))
check("intent: 'learn your voice'", sid.matches_enroll_intent("please learn your voice"))
check("intent: 'remember my voice'", sid.matches_enroll_intent("remember my voice"))
check("intent NEG: 'remember the milk'", not sid.matches_enroll_intent("remember the milk"))
check("intent NEG: bare 'enroll me'", not sid.matches_enroll_intent("enroll me in the class"))
check("intent NEG: empty", not sid.matches_enroll_intent(""))


# --- Test 3: identify over a registry --------------------------------------
regA = [sid.Speaker(name="Saul", lang="en", embedding=uA)]
regAB = [sid.Speaker(name="Saul", lang="en", embedding=uA),
         sid.Speaker(name="Bob", lang="es", embedding=uB)]

# empty registry -> not recognized, score 0
r = sid.identify(_clip(3), [], threshold=0.7)
check("identify empty registry -> not recognized", not r.recognized and r.score == 0.0)

# probe_A (cos 0.8 to uA) vs threshold 0.7 -> recognized Saul
r = sid.identify(_clip(3), regA, threshold=0.7)
check("identify probe_A @0.7 -> recognized", r.recognized)
check("identify probe_A -> name=Saul", r.name == "Saul")
check("identify probe_A -> lang=en", r.lang == "en")
check("identify probe_A -> score~=0.8", abs(r.score - 0.8) < 1e-3)

# same probe vs stricter threshold 0.85 -> NOT recognized (fail-open caller's call)
r = sid.identify(_clip(3), regA, threshold=0.85)
check("identify probe_A @0.85 -> not recognized", not r.recognized)
check("identify not-recognized still reports best score", abs(r.score - 0.8) < 1e-3)
check("identify not-recognized -> name None", r.name is None and r.lang is None)

# multi-user argmax: probe_A is closer to Saul(uA, .8) than Bob(uB, .1)
r = sid.identify(_clip(3), regAB, threshold=0.7)
check("identify multi-user picks closest (Saul)", r.recognized and r.name == "Saul")

# a probe that IS uB -> picks Bob (Spanish)
r = sid.identify(_clip(2), regAB, threshold=0.7)
check("identify uB -> Bob (es)", r.recognized and r.name == "Bob" and r.lang == "es")

# unusable clip (embed returns None) -> not recognized, score 0
r = sid.identify(_clip(0), regAB, threshold=0.7)
check("identify unusable clip -> not recognized", not r.recognized and r.score == 0.0)


# --- Test 4: enroll_from_audio + load_registry round-trip ------------------
with tempfile.TemporaryDirectory() as td:
    reg_dir = Path(td)

    # enroll Saul from 3 clips that all embed to uA -> mean == uA
    ok, msg = sid.enroll_from_audio("Saul", "en", [_clip(1)] * 3, reg_dir)
    check("enroll Saul -> ok", ok)
    check("enroll Saul -> files written",
          (reg_dir / "saul.npy").exists() and (reg_dir / "saul.json").exists())

    loaded = sid.load_registry(reg_dir)
    check("load_registry -> 1 speaker", len(loaded) == 1)
    check("loaded name/lang correct", loaded[0].name == "Saul" and loaded[0].lang == "en")
    check("loaded embedding shape (256,)", loaded[0].embedding.shape == (_DIM,))
    check("loaded embedding ~= uA", float(np.dot(loaded[0].embedding, uA)) > 0.999)

    # enroll a second speaker (Bob, es)
    ok, _ = sid.enroll_from_audio("Bob", "es", [_clip(2)] * 3, reg_dir)
    check("enroll Bob -> ok", ok)
    check("registry now has 2", len(sid.load_registry(reg_dir)) == 2)
    check("list_names has both", set(sid.list_names(reg_dir)) == {"Saul", "Bob"})

    # identify end-to-end against the LOADED registry
    r = sid.identify(_clip(3), sid.load_registry(reg_dir), threshold=0.7)
    check("end-to-end identify -> Saul", r.recognized and r.name == "Saul")

    # insufficient usable clips (only 1 of 3 usable) -> fail, no overwrite
    ok, msg = sid.enroll_from_audio("Ghost", "en", [_clip(1), _clip(0), _clip(0)], reg_dir)
    check("enroll with <2 usable clips -> fail", not ok)
    check("failed enroll wrote nothing", not (reg_dir / "ghost.npy").exists())

    # delete_speaker
    check("delete Bob -> True", sid.delete_speaker("Bob", reg_dir))
    check("registry back to 1 after delete", len(sid.load_registry(reg_dir)) == 1)
    check("delete missing -> False", not sid.delete_speaker("Nobody", reg_dir))


# --- Test 5: load_registry robustness --------------------------------------
with tempfile.TemporaryDirectory() as td:
    reg_dir = Path(td)
    reg_dir.mkdir(exist_ok=True)
    # wrong-shape embedding -> skipped, not crash
    np.save(reg_dir / "bad.npy", np.zeros(128, dtype=np.float32))
    check("load_registry skips wrong-shape file", sid.load_registry(reg_dir) == [])
    # a valid one alongside the bad one still loads
    np.save(reg_dir / "good.npy", uA)
    loaded = sid.load_registry(reg_dir)
    check("load_registry loads good, skips bad", len(loaded) == 1 and loaded[0].name == "good")

# missing dir -> empty, no crash
check("load_registry missing dir -> []",
      sid.load_registry(Path(tempfile.gettempdir()) / "no-such-jarvis-dir-xyz") == [])


# --- Test 6: parse_named_enroll_intent (M69 Phase 4) -----------------------
# Named household enrollment, checked BEFORE the primary 'enroll my voice'.
def _pn(t):
    return sid.parse_named_enroll_intent(t)


check("named: \"enroll Alice's voice\" -> (Alice, en)", _pn("enroll Alice's voice") == ("Alice", "en"))
check("named: 'enroll voice Alice' -> (Alice, en)", _pn("enroll voice Alice") == ("Alice", "en"))
check("named: 'enroll voice Bob in Spanish' -> (Bob, es)",
      _pn("enroll voice Bob in Spanish") == ("Bob", "es"))
check("named: \"enroll Bob's voice in spanish\" -> (Bob, es)",
      _pn("enroll Bob's voice in spanish") == ("Bob", "es"))
check("named: lowercase name capitalized", _pn("enroll voice alice") == ("Alice", "en"))
check("named: english keyword -> en", _pn("enroll voice Bob in english") == ("Bob", "en"))
# 'my'/'your' are the PRIMARY user, not a named target -> None
check("named: 'enroll my voice' -> None (primary)", _pn("enroll my voice") is None)
check("named: 'enroll your voice' -> None (primary)", _pn("enroll your voice") is None)
check("named: non-enroll text -> None", _pn("remember the milk") is None)
check("named: empty -> None", _pn("") is None)
# The disjointness that makes ordering work: a named utterance that the
# PRIMARY regex ALSO matches ('enroll voice Alice') must be caught by named.
check("named beats primary on 'enroll voice Alice'",
      _pn("enroll voice Alice") is not None
      and sid.matches_enroll_intent("enroll voice Alice"))


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
