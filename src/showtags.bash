#!/bin/bash
echo
echo ----debugging switches----
grep "_DEBUG 1" *.cpp
echo
echo ----verbosity switches----
grep "_VERBOSE 1" *.cpp
echo
echo ----development tags----
grep "_DEV" *.cpp
echo
