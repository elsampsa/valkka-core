#!/bin/bash
# In order to run the test programs before system-wide installation, run this file IN THIS DIRECTORY with
#
# source test_env.bash
#
# After that the test programs in bin/ will work
echo "You did run this with"
echo "   source test_env.bash"
echo "right?"
export LD_LIBRARY_PATH=$PWD/lib
