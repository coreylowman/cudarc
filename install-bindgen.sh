#!/bin/bash
apt update -y
apt upgrade -y
apt install -y curl clang
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
cargo install bindgen-cli@0.71.1
