#!/bin/bash
# read -p "Check that in dist/ there is only one .whl package!  Press enter to continue"
# twine upload -r pypi dist/valkka-*-manylinux1_x86_64.whl
read -p "Check that in dist/ there is only one .tar.gz package!  Press enter to continue"
twine upload -r pypi dist/valkka-*.tar.gz
