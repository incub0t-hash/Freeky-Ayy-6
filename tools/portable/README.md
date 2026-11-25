# Portable best-run renderer

Extract the release ZIP to a folder on Windows 10 or 11 x64, then drag your own `game_scarlett_master.exe` onto `Play.cmd`. Its folder must also contain `shaders.rpf` and the rest of that game installation.

If Windows needs the C++ runtime, run `Install-Runtime.cmd` first. It opens Microsoft's signed installer included in the package.

The first launch prepares the 2,492 reference shaders from your game, then starts the game. Graphics-driver pipelines are compiled on that machine as the game runs. Later launches reuse the local shader preparation. State, player saves and logs are stored under `%LOCALAPPDATA%\Freeky-Ayy-6\best-run`.

The package includes the original best-run VIctory executable, a cache-preparation executable, a player checkpoint, exact renderer settings, launch and monitoring scripts, the runtime installer, source history and research branches, and file checksums. The cache is recreated from the recipient's game because it contains translated shaders and decoded textures.

The reference renderer is `9e05a6eb2a49ceb14370b598c5dfc215ffe03a25`. Its best documented diner shots show both characters and clothing at approximately 1.08 FPS. Other shots still have visual defects, and repeated runs can vary with asset streaming, pipeline readiness and hardware.

For a captured comparison with a four-minute cap, use

```powershell
.\scripts\Play.ps1 -GameExe 'E:\game\game_scarlett_master.exe'
```

Pass `-PlayerData 'E:\VIctory\data'` to use your own checkpoint. Interactive `Play.cmd` keeps a persistent player save. The default checkpoint is copied when that persistent save is first created.

To restore the complete source archive, use Git

```powershell
git clone -b diner-reference .\source.bundle .\source
git -C .\source branch -a
```

The release also carries patches for tracked local source changes that had not been committed in the original worktrees, when present. Their inventory identifies each base revision.

The renderer needs a Direct3D 12 GPU and its current Windows driver. The reference used an RTX 3080 Ti with 12 GB VRAM and 32 GB system RAM. The launch supervisor sets a 16 GiB process working-set ceiling and stops on low physical memory or commit headroom.
