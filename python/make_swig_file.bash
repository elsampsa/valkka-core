#!/bin/bash
cd include
./makeswig.bash > tmp.i
cd ..
cat valkka.i.base > valkka_core.i
cat include/tmp.i >> valkka_core.i
