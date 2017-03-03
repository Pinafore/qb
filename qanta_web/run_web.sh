#!/usr/bin/env bash

set -e

sleep 5

python manage.py makemigrations
python manage.py makemigrations leaderboard
python manage.py migrate
python manage.py runserver 0.0.0.0:8000