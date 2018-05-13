#!/bin/bash
# python3 setup.py bdist_wheel -p manylinux1_x86-64 upload -r test
read -p "Check that in dist/ there is only one .whl package!  Press enter to continue"
twine upload -r test dist/valkka-*-manylinux1_x86_64.whl
