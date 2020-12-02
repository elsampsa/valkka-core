#!/bin/bash
#
# Does the standardized debian build
# This way Petri doesn't have to remember complicated commands  :)
#
make -f debian/rules clean
make -f debian/rules build
