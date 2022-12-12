#!/bin/bash
set -eu
BASEDIR=$(dirname "$0")

for filename in "$BASEDIR"/**.rs; do
    [ -e "$filename" ] || continue
    if [[ "${filename##*/}" == "sys.rs" ]]
    then
        continue
    fi
    rustfmt "$@" -l "$filename"
done