# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 1. Define exactly what files to include
added_files = [
    ('src', 'src'),  # Pulls the entire src folder into the internal app path
]

a = Analysis(
    ['src/app.py'],  # The only entry point
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'dash', 'dash_bootstrap_components', 'pandas', 
        'numpy', 'scipy.signal', 'pyarrow'
    ],
    # 2. EXCLUDE the root folder clutter
    excludes=['venv', '.venv', 'tests', 'data', 'old scripts', 'setup.py'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CogMo_Toolkit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compresses the final .exe to save space
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Set to False once you're sure it works to hide the black terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)