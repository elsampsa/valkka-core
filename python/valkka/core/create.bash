#!/bin/bash
st="s/NAME/"$1"/g"
cat template.py | sed $st > $1.py
