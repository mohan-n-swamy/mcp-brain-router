"""The lean GROK_HOME shares ~/.grok/auth.json through a symlink (2026-11-08 both-homes sign-out)."""
import json
import os

import pytest

from mcp_brain_router import backends

LOGIN = {"https://auth.x.ai::c": {"key": "k", "refresh_token": "r", "expires_at": "x"}}


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    lean = tmp_path / "lean"
    monkeypatch.setattr(backends, "_GROK_LEAN_HOME", str(lean))
    (tmp_path / ".grok").mkdir()
    return tmp_path / ".grok" / "auth.json", lean / "auth.json"


def write(p, data, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    if mtime is not None:
        os.utime(p, (mtime, mtime))


def linked(src, dest):
    return dest.is_symlink() and os.path.realpath(dest) == os.path.realpath(src)


def test_fresh_lean_home_links_the_cli_login(homes):
    src, dest = homes
    write(src, LOGIN)
    backends._ensure_grok_lean_home()
    assert linked(src, dest)


def test_old_copy_becomes_a_link_to_the_newer_login(homes):
    src, dest = homes
    write(src, LOGIN, mtime=2000)
    write(dest, {"https://auth.x.ai::c": {"refresh_token": "old"}}, mtime=1000)
    backends._ensure_grok_lean_home()
    assert linked(src, dest) and json.loads(src.read_text()) == LOGIN


def test_newer_lean_login_is_copied_back_then_linked(homes):
    # grok replaced the link with a file when the router's call refreshed the token
    src, dest = homes
    newer = {"https://auth.x.ai::c": {"refresh_token": "new"}}
    write(src, {"https://auth.x.ai::c": {"refresh_token": "rotated-away"}}, mtime=1000)
    write(dest, newer, mtime=2000)
    backends._ensure_grok_lean_home()
    assert linked(src, dest) and json.loads(src.read_text()) == newer


def test_signed_out_cli_never_signs_out_the_router(homes):
    # the 2026-11-08 cascade: the CLI wrote {} (newer); the old step copied it over the router's login
    src, dest = homes
    write(dest, LOGIN, mtime=1000)
    write(src, {}, mtime=2000)
    backends._ensure_grok_lean_home()
    assert json.loads(src.read_text()) == LOGIN and linked(src, dest)
    assert (src.stat().st_mode & 0o777) == 0o600


def test_no_login_anywhere_changes_nothing(homes):
    src, dest = homes
    write(src, {}, mtime=2000)
    write(dest, {}, mtime=1000)
    backends._ensure_grok_lean_home()
    assert not dest.is_symlink() and json.loads(dest.read_text()) == {}


def test_link_elsewhere_is_replaced(homes, tmp_path):
    src, dest = homes
    write(src, LOGIN)
    other = tmp_path / "other.json"
    write(other, {})
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(other, dest)
    backends._ensure_grok_lean_home()
    assert linked(src, dest) and json.loads(other.read_text()) == {}


def test_existing_link_is_left_alone(homes):
    src, dest = homes
    write(src, LOGIN)
    backends._ensure_grok_lean_home()
    ino = os.lstat(dest).st_ino
    backends._ensure_grok_lean_home()
    assert os.lstat(dest).st_ino == ino


def test_validity_reads_no_value():
    assert backends._grok_auth_valid("/nonexistent/auth.json") is False
