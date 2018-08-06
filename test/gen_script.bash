#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Give test name (without .cpp)"
    exit
fi
let c=1
grep -E "\@\@.*\*" $1.cpp -o | while read -r line ; do
    echo "# $line"
    echo "echo $c"
    echo "\$valgrind ./"$1" $c \$LOGLEVEL &>> test.out"
    echo "printf \"END: $1 $c\n\n\" &>> test.out"
    echo
    let c=c+1
done
