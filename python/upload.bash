#!/bin/bash
# python3 setup.py bdist_wheel -p manylinux1_x86-64 upload
read -p "Check that in dist/ there is only one .whl package!  Press enter to continue"
twine upload -r pypi dist/valkka-*-manylinux1_x86_64.whl
