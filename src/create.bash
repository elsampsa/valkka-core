#!/bin/bash
st="s/NAME/"$1"/g"
cat template | sed $st > $1.cpp
