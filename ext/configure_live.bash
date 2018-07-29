#!/bin/bash
cp config.linux-generic live/
cd live
# ./genMakefiles linux-64bit
./genMakefiles linux-generic
cd ..
