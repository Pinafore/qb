#!/usr/bin/env bash


if [[ -n "$1" ]]; then
    echo "$1" | netcat localhost 26542
fi
