#!/usr/bin/env bash
# Add the frozen API server to a Linux AppImage built by `npm run tauri:build`.
#
# On Linux the sidecar ships in /usr/libexec/music-cluster/ (deb.files in
# src-tauri/tauri.linux.conf.json), outside usr/lib, where the AppImage tool
# would rewrite its libraries. Tauri builds the AppImage without deb.files, so
# this unpacks it, adds the sidecar and packs it again.
#
#   scripts/add_sidecar_to_appimage.sh src-tauri/target/release/bundle/appimage/*.AppImage
set -euo pipefail

appimage=$(realpath "$1")
sidecar=$(realpath "$(dirname "$0")/../src-tauri/sidecar")
[ -x "$sidecar/music-cluster-api" ] || { echo "No sidecar in $sidecar: run scripts/build_sidecar.py" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"

"$appimage" --appimage-extract > /dev/null
mkdir -p squashfs-root/usr/libexec/music-cluster
cp -a "$sidecar" squashfs-root/usr/libexec/music-cluster/

curl -fsSL -o appimagetool \
  https://github.com/AppImage/appimagetool/releases/download/1.9.0/appimagetool-x86_64.AppImage
chmod +x appimagetool
ARCH=x86_64 ./appimagetool --appimage-extract-and-run squashfs-root "$appimage"
