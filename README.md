# Pesqueiro

Fishing automation for World of Warcraft on Linux (Wayland). Pesqueiro watches the game through the desktop screen-sharing portal, finds the bobber by colour, finds the game cursor by template matching, moves a virtual mouse onto the bobber with a human-like path, and right-clicks when it bites.

## How it works

1. Casts with a configurable key.
2. Waits for the bobber to land, then parks the cursor next to it.
3. Watches the bobber for a downward dip, a shrinking red area, or a bounce.
4. Moves onto the bobber, right-clicks to loot, and casts again.

Screen capture needs your permission through the desktop portal. Input is sent through a virtual device (`/dev/uinput`), so no X11 is required.

## Requirements

- Linux with a Wayland compositor and a working `xdg-desktop-portal` ScreenCast backend (GNOME, KDE, Hyprland, Sway with `xdg-desktop-portal-wlr`, ...).
- Python 3.11 or newer.
- GStreamer with the PipeWire plugin: `gst-launch-1.0`, `pipewiresrc`, `videoconvert`, `fdsink`.
  - Debian/Ubuntu: `sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-pipewire`
  - Fedora: `sudo dnf install gstreamer1 gstreamer1-plugins-base pipewire-gstreamer`
  - Arch: `sudo pacman -S gstreamer gst-plugins-base gst-plugin-pipewire`
- Write access to `/dev/uinput` (see [Input permissions](#input-permissions)).

## Install

### AppImage (recommended)

Download `Pesqueiro-<version>-x86_64.AppImage` from the releases page, then:

```sh
chmod +x Pesqueiro-*-x86_64.AppImage
./Pesqueiro-*-x86_64.AppImage
```

The AppImage bundles Python, PySide6, numpy, Pillow, dbus-next, evdev, GStreamer and the PipeWire plugin. The only things it needs from the system are a Wayland session with a ScreenCast portal, FUSE 2 (`libfuse2`; or run it with `--appimage-extract-and-run`), the usual desktop libraries Qt relies on (OpenGL, fontconfig, xkbcommon), and [`/dev/uinput` access](#input-permissions).

The bundled GStreamer is built against a recent glibc. On a distribution with an older glibc the AppImage automatically falls back to the system `gst-launch-1.0`, so install the GStreamer packages from [Requirements](#requirements) there.

Run the command line tool with `./Pesqueiro-*-x86_64.AppImage cli --wayland --mode red`.

Cursor templates and debug output are stored in `~/.local/share/pesqueiro/` (set `PESQUEIRO_HOME` to change it).

### From source

```sh
git clone [<repository-url> pesqueiro](https://github.com/exiele/pesqueiro)
cd pesqueiro
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[gui,move]'
```

Extras:

| Extra  | Installs | Needed for                          |
| ------ | -------- | ----------------------------------- |
| `gui`  | PySide6  | The desktop application             |
| `move` | evdev    | Moving the mouse, clicking, casting |

Use `python3 -m venv --system-site-packages .venv` to reuse a system-wide PySide6.

## Input permissions

Pesqueiro creates a virtual mouse and keyboard through `/dev/uinput`. Grant your user access once:

```sh
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo usermod -aG input "$USER"
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Log out and back in for the group change to apply. If `/dev/uinput` does not exist, run `sudo modprobe uinput` (add `uinput` to `/etc/modules-load.d/` to make it permanent).

Without this access the preview and detection still work, but the bot cannot cast, move, or click.

## Cursor templates

The game cursor is located by matching the images in `cursors/` (`~/.local/share/pesqueiro/cursors/` when using the AppImage). No images are shipped with the project: you must add crops of your own in-game cursors before auto-fishing works.

- Append the hotspot (the pixel that points at the target) as `@x,y`, for example `real_gauntlet@3,3.png`. Without a suffix the hotspot is the top-left corner.
- Supported formats: PNG, WebP, GIF, BMP.

Without any template, detection runs but auto-fishing is disabled.

Template images are ignored by git, so they stay local to your machine.

## Use

```sh
.venv/bin/pesqueiro-gui
```

1. Pick the bobber colour (**Vermelha** = red, **Azul** = blue, for lava) and the cast key (**Tecla de lançamento**, default `4`).
2. Click **Iniciar captura de tela** (start screen capture) and choose the game window or display in the desktop prompt.
3. The preview shows the detection region (green frame), the bobber (orange crosshair) and the cursor (blue square).
4. Click **Iniciar pesca** (start fishing). After a 3 second countdown, switch to the game window. Pesqueiro casts and fishes on its own.
5. Click **Parar pesca** to stop fishing and **Parar captura de tela** to end the capture.

Keep the in-game cursor visible. The bobber is searched in the central half of the captured frame.

Start with `--debug` to show the log pane and the snapshot and trace tools:

```sh
.venv/bin/pesqueiro-gui --debug
```

Snapshots and traces are written to `debug/`. Use **Copy all** to copy the log for a bug report.

## Command line

Inspect a saved screenshot:

```sh
.venv/bin/pesqueiro path/to/frame.png --mode red --origin 300,200
```

Grab one frame through the portal and locate the bobber:

```sh
.venv/bin/pesqueiro --wayland --mode red
```

Output is JSON. The exit status is `0` when a bobber was found, `2` when not, and `3` on a capture or file error.

| Option                   | Meaning                                        |
| ------------------------ | ---------------------------------------------- |
| `--mode red\|blue`       | Bobber colour (`blue` for lava)                |
| `--origin X,Y`           | Screen position of the image's top-left corner |
| `--previous X,Y`         | Restrict the search around a previous position |
| `--radius N`             | Search radius used with `--previous`           |
| `--colour-multiplier`    | Colour dominance tuning                        |
| `--closeness-multiplier` | Colour similarity tuning                       |

## Troubleshooting

- **"cannot open /dev/uinput"**: see [Input permissions](#input-permissions).
- **"GStreamer is required"**: install the GStreamer packages listed above.
- **Capture times out**: make sure a portal backend for your desktop is installed and running.
- **Cursor not found**: add a `real_` template cut from your own game cursor.
- **"input is not mapping to the captured screen"**: capture the whole display or the game window at its native scale.

## Development

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Build the AppImage into `dist/` (needs `curl`, `gcc`, Linux kernel headers, and GStreamer with the PipeWire plugin installed on the build machine):

```sh
packaging/build-appimage.sh
```

## Disclaimer

This project was made for educational purposes only. Automating gameplay may violate the game's terms of service. Use at your own risk.
