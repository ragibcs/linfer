#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${LINFER_REPO_URL:-https://github.com/ragibcs/linfer.git}"
SRC_DIR="${LINFER_SRC_DIR:-$HOME/.cache/linfer-src}"
INSTALL_DIR="${LINFER_INSTALL_DIR:-$HOME/.local/bin}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! need_cmd pacman; then
  echo "This installer is for Arch Linux (pacman required)."
  exit 1
fi

missing_pkgs=()
for pkg in git rustup base-devel; do
  if ! pacman -Q "$pkg" >/dev/null 2>&1; then
    missing_pkgs+=("$pkg")
  fi
done

if ((${#missing_pkgs[@]} > 0)); then
  echo "Installing required packages: ${missing_pkgs[*]}"
  sudo pacman -S --needed --noconfirm "${missing_pkgs[@]}"
fi

if ! rustup toolchain list | grep -q "stable"; then
  rustup default stable
fi

mkdir -p "$INSTALL_DIR"
mkdir -p "$(dirname "$SRC_DIR")"

if [ -d "$SRC_DIR/.git" ]; then
  echo "Updating existing source at $SRC_DIR"
  git -C "$SRC_DIR" pull --ff-only
else
  echo "Cloning source to $SRC_DIR"
  rm -rf "$SRC_DIR"
  git clone "$REPO_URL" "$SRC_DIR"
fi

echo "Building linfer (release)"
cd "$SRC_DIR"
cargo build --release

install -m 755 "$SRC_DIR/target/release/linfer" "$INSTALL_DIR/linfer"

if ! echo ":$PATH:" | grep -q ":$INSTALL_DIR:"; then
  shell_name="$(basename "${SHELL:-bash}")"
  case "$shell_name" in
    fish)
      if command -v fish >/dev/null 2>&1; then
        fish -lc "fish_add_path --path '$INSTALL_DIR'" || true
      fi
      ;;
    zsh)
      grep -qxF "export PATH=\"$INSTALL_DIR:\$PATH\"" "$HOME/.zshrc" 2>/dev/null || echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.zshrc"
      ;;
    *)
      grep -qxF "export PATH=\"$INSTALL_DIR:\$PATH\"" "$HOME/.bashrc" 2>/dev/null || echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$HOME/.bashrc"
      ;;
  esac
fi

echo
"$INSTALL_DIR/linfer" --version || true
echo "Installed: $INSTALL_DIR/linfer"
echo
echo "Open a new terminal, then run:"
echo "  linfer pull \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\" --quant q4"
