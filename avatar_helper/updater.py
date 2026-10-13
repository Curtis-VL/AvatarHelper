"""Update checker: compares the installed version with the latest GitHub release, and can
download and install it.

Releases are published by .github/workflows/release.yml, one per change to the add-on, each
with an installable zip. The check runs in a background thread so Blender never waits on the
network; the panels show the result when it's in.
"""
import json
import os
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request

import bpy

REPO = "Curtis-VL/AvatarHelper"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
TIMEOUT = 15  # seconds

# Shared with the background thread; only plain values, never Blender data.
state = {"status": "idle", "latest": None, "url": "", "page": RELEASES_PAGE, "error": ""}
_lock = threading.Lock()


def current_version():
    return tuple(sys.modules[__package__].bl_info["version"])


def version_text(v):
    return ".".join(str(x) for x in v) if v else "?"


def parse_version(tag):
    """'v0.1.23' -> (0, 1, 23)."""
    nums = re.findall(r"\d+", tag or "")
    return tuple(int(n) for n in nums[:3]) if nums else None


def is_dev_install():
    """The add-on folder is a link to (or inside) a git checkout: updating would overwrite the
    source, so it's updated with git instead."""
    here = os.path.dirname(os.path.abspath(__file__))
    real = os.path.realpath(here)
    return os.path.normcase(real) != os.path.normcase(here) or os.path.isdir(os.path.join(real, "..", ".git"))


def _set(**kw):
    with _lock:
        state.update(kw)


def check(background=True):
    """Look up the latest release. In the background by default; the result lands in `state`."""
    if state["status"] in ("checking", "downloading"):
        return
    _set(status="checking", error="")
    if background:
        threading.Thread(target=_check, daemon=True).start()
        bpy.app.timers.register(_redraw_when_done, first_interval=0.5)
    else:
        _check()


def _check():
    try:
        req = urllib.request.Request(LATEST_URL, headers={
            "Accept": "application/vnd.github+json", "User-Agent": "AvatarHelper-update-check"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
        latest = parse_version(data.get("tag_name"))
        zips = [a for a in data.get("assets", []) if a.get("name", "").lower().endswith(".zip")]
        if latest is None or not zips:
            raise ValueError("the latest release has no add-on zip")
        status = "available" if latest > current_version() else "current"
        _set(status=status, latest=latest, url=zips[0]["browser_download_url"],
             page=data.get("html_url") or RELEASES_PAGE)
    except urllib.error.HTTPError as e:
        msg = "no releases found (the repository may be private)" if e.code == 404 else f"GitHub said {e.code}"
        _set(status="error", error=msg)
    except Exception as e:  # network down, bad JSON, ...: report, never raise in the thread
        _set(status="error", error=str(e) or type(e).__name__)


def install():
    """Download the latest release and install it over this one. Takes effect after a restart."""
    if is_dev_install():
        raise ValueError("This is a development copy (linked to a git checkout); update it with git instead")
    if state["status"] != "available" or not state["url"]:
        raise ValueError("Check for updates first")
    _set(status="downloading")
    try:
        req = urllib.request.Request(state["url"], headers={"User-Agent": "AvatarHelper-update-check"})
        with urllib.request.urlopen(req, timeout=TIMEOUT * 4) as r:
            payload = r.read()
        path = os.path.join(tempfile.gettempdir(), f"AvatarHelper-{version_text(state['latest'])}.zip")
        with open(path, "wb") as f:
            f.write(payload)
        result = bpy.ops.preferences.addon_install(filepath=path, overwrite=True)
        if result != {'FINISHED'}:
            raise ValueError("Blender couldn't install the downloaded zip")
    except Exception:
        _set(status="available")
        raise
    _set(status="installed")


def _redraw_when_done():
    if state["status"] == "checking":
        return 0.5
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ('VIEW_3D', 'PREFERENCES'):
                area.tag_redraw()
    return None


def _startup_check():
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is not None and getattr(addon.preferences, "check_updates", False):
        check()
    return None


def register():
    if not bpy.app.background:
        bpy.app.timers.register(_startup_check, first_interval=3.0)


def unregister():
    for fn in (_startup_check, _redraw_when_done):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)


def draw(layout, compact=False):
    """Update status and buttons. compact: just a notice when there's something to do."""
    status = state["status"]
    if compact:
        if status == "available":
            row = layout.row(align=True)
            row.alert = True
            row.label(text=f"Update available: {version_text(state['latest'])}", icon='IMPORT')
            row.operator("avatar_helper.update_install", text="Update")
        elif status == "installed":
            layout.label(text="Updated: restart Blender to finish", icon='INFO')
        return
    col = layout.column()
    col.label(text=f"Installed: {version_text(current_version())}" + ("  (development copy)" if is_dev_install() else ""))
    if status == "checking":
        col.label(text="Checking for updates...", icon='SORTTIME')
    elif status == "downloading":
        col.label(text="Downloading...", icon='SORTTIME')
    elif status == "current":
        col.label(text=f"Up to date (latest {version_text(state['latest'])})", icon='CHECKMARK')
    elif status == "available":
        col.label(text=f"Update available: {version_text(state['latest'])}", icon='IMPORT')
    elif status == "installed":
        col.label(text="Updated: restart Blender to finish", icon='INFO')
    elif status == "error":
        col.label(text="Couldn't check: " + state["error"], icon='ERROR')
    row = col.row(align=True)
    row.operator("avatar_helper.update_check", icon='FILE_REFRESH')
    sub = row.row(align=True)
    sub.enabled = status == "available" and not is_dev_install()
    sub.operator("avatar_helper.update_install", icon='IMPORT')
    row.operator("wm.url_open", text="", icon='URL').url = state["page"]
