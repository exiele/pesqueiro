#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/build/appimage"
CACHE="$ROOT/build/cache"
APPDIR="$WORK/AppDir"
PYTHON_APPIMAGE_URL="https://github.com/niess/python-appimage/releases/download/python3.12/python3.12.15-cp312-cp312-manylinux_2_28_x86_64.AppImage"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml")"
OUTPUT="$ROOT/dist/Pesqueiro-$VERSION-x86_64.AppImage"

GST_PLUGINS=(coreelements videoconvertscale pipewire)
SPA_PLUGINS=(support audioconvert videoconvert control)
PIPEWIRE_MODULES=(protocol-native client-node client-device adapter metadata rt spa-node-factory spa-device-factory link-factory profiler session-manager)
SYSTEM_LIBS='^(ld-linux.*|libc|libm|libdl|libpthread|librt|libutil|libresolv|libnsl|libanl|libBrokenLocale|libthread_db)\.so'

for tool in gst-launch-1.0 gst-inspect-1.0 gcc ldd objdump curl; do
    command -v "$tool" >/dev/null || { echo "missing build tool: $tool" >&2; exit 1; }
done

fetch() {
    local url="$1" target="$2"
    [ -x "$target" ] && return
    mkdir -p "$(dirname "$target")"
    curl -fL --progress-bar -o "$target" "$url"
    chmod +x "$target"
}

copy_with_libs() {
    local binary="$1" libdir="$2" library
    while read -r library; do
        [[ "$(basename "$library")" =~ $SYSTEM_LIBS ]] && continue
        [ -e "$libdir/$(basename "$library")" ] || cp -L "$library" "$libdir/"
    done < <(ldd "$binary" | awk '$3 ~ /^\// {print $3}')
}

fetch "$PYTHON_APPIMAGE_URL" "$CACHE/python.AppImage"
fetch "$APPIMAGETOOL_URL" "$CACHE/appimagetool.AppImage"

rm -rf "$WORK"
mkdir -p "$WORK"
(cd "$WORK" && "$CACHE/python.AppImage" --appimage-extract >/dev/null && mv squashfs-root AppDir)

PYTHON_ROOT="$APPDIR/opt/python3.12"
PYTHON="$PYTHON_ROOT/bin/python3.12"

echo "== Python dependencies"
"$PYTHON" -m pip install --no-warn-script-location --no-cache-dir \
    "Pillow>=10.0" "numpy>=1.26" "dbus-next>=0.2.3" "PySide6-Essentials>=6.7" "evdev>=1.7"
"$PYTHON" -m pip install --no-warn-script-location --no-cache-dir --no-deps "$ROOT"

echo "== Pruning"
SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$ROOT/build/lib" "$ROOT/build/bdist."*
rm -rf "$SITE"/PySide6/{examples,include,typesystems,glue,scripts,support,metatypes} \
       "$SITE"/PySide6/Qt/{qml,translations}
find "$SITE" -name '__pycache__' -type d -prune -exec rm -rf {} +
rm -f "$PYTHON_ROOT"/bin/pip* "$PYTHON_ROOT"/bin/f2py* "$PYTHON_ROOT"/bin/numpy-config
rm -rf "$SITE"/pip "$SITE"/pip-* "$SITE"/build "$SITE"/build-* "$SITE"/pyproject_hooks*

echo "== GStreamer"
GST_ROOT="$APPDIR/usr/lib/gst"
mkdir -p "$GST_ROOT"/{bin,lib,plugins,spa-0.2,pipewire-0.3} "$APPDIR/usr/bin"

GST_BIN="$(command -v gst-launch-1.0)"
GST_PLUGIN_DIR="$(dirname "$(find /usr/lib /usr/lib64 -name libgstpipewire.so -print -quit 2>/dev/null)")"
SPA_DIR="$(find /usr/lib /usr/lib64 -type d -name spa-0.2 -print -quit 2>/dev/null)"
PW_DIR="$(find /usr/lib /usr/lib64 -type d -name pipewire-0.3 -print -quit 2>/dev/null)"

cp -L "$GST_BIN" "$GST_ROOT/bin/"
copy_with_libs "$GST_BIN" "$GST_ROOT/lib"
for plugin in "${GST_PLUGINS[@]}"; do
    cp -L "$GST_PLUGIN_DIR/libgst$plugin.so" "$GST_ROOT/plugins/"
    copy_with_libs "$GST_PLUGIN_DIR/libgst$plugin.so" "$GST_ROOT/lib"
done
for plugin in "${SPA_PLUGINS[@]}"; do
    cp -rL "$SPA_DIR/$plugin" "$GST_ROOT/spa-0.2/"
    for object in "$GST_ROOT/spa-0.2/$plugin"/*.so; do copy_with_libs "$object" "$GST_ROOT/lib"; done
done
for module in "${PIPEWIRE_MODULES[@]}"; do
    cp -L "$PW_DIR/libpipewire-module-$module.so" "$GST_ROOT/pipewire-0.3/"
    copy_with_libs "$PW_DIR/libpipewire-module-$module.so" "$GST_ROOT/lib"
done
install -m 755 "$ROOT/packaging/gst-launch-1.0" "$APPDIR/usr/bin/gst-launch-1.0"
find "$GST_ROOT" -type f -print0 | xargs -0 -n1 objdump -T 2>/dev/null \
    | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n1 | sed 's/GLIBC_//' > "$GST_ROOT/glibc-min"
echo "bundled GStreamer needs glibc $(cat "$GST_ROOT/glibc-min")"

echo "== AppDir metadata"
install -m 755 "$ROOT/packaging/AppRun" "$APPDIR/AppRun"
cp "$ROOT/packaging/pesqueiro.desktop" "$APPDIR/pesqueiro.desktop"
rm -f "$APPDIR"/*.desktop.orig "$APPDIR"/python*.desktop "$APPDIR/.DirIcon" "$APPDIR/python.png"
"$PYTHON" "$ROOT/packaging/make_icon.py" "$APPDIR/pesqueiro.png"
ln -s pesqueiro.png "$APPDIR/.DirIcon"

echo "== Packaging"
mkdir -p "$ROOT/dist"
ARCH=x86_64 "$CACHE/appimagetool.AppImage" --no-appstream "$APPDIR" "$OUTPUT"
echo "Built $OUTPUT"
