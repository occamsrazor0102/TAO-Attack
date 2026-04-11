# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# PyInstaller spec files may not define __file__ in all execution contexts.
project_root = Path.cwd().resolve()

a = Analysis(
    ["attack.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=[(str(project_root / "data"), "data")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="attack",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
