#!/bin/bash
if [ $# -ne 3 ]; then
  echo "Give major minor patch"
  exit
fi
# mod header files:
find include/ -name "*.h" -exec sed -i -r "s/@version (.*)/@version $1.$2.$3 /g" {} \;
# mod header files:
find src/ -name "*.cpp" -exec sed -i -r "s/@version (.*)/@version $1.$2.$3 /g" {} \;
# mod test files:
find test/ -name "*.cpp" -exec sed -i -r "s/@version (.*)/@version $1.$2.$3 /g" {} \;

# mod python setup script:
sed -i -r "s/version = (.*)/version = \"$1.$2.$3\", /g" python/setup.py
# mod python files:
find python/valkka/ -name "*.py" -exec sed -i -r "s/@version (.*)/@version $1.$2.$3 /g" {} \;

# mod cmake (for making debian packages) 
sed -i -r "s/SET\(MAJOR_VERSION (.*)/SET(MAJOR_VERSION \"$1\") /g" CMakeLists.txt
sed -i -r "s/SET\(MINOR_VERSION (.*)/SET(MINOR_VERSION \"$2\") /g" CMakeLists.txt
sed -i -r "s/SET\(PATCH_VERSION (.*)/SET(PATCH_VERSION \"$3\") /g" CMakeLists.txt

# mod doxygen documentation
sed -i -r "s/PROJECT_NUMBER(.*)/PROJECT_NUMBER = \"$1.$2.$3\"/g" docs/config

# mod version numbers in sizes.h
sed -i -r "s/VERSION_MAJOR(.*);/VERSION_MAJOR = $1;/g" include/constant.h
sed -i -r "s/VERSION_MINOR(.*);/VERSION_MINOR = $2;/g" include/constant.h
sed -i -r "s/VERSION_PATCH(.*);/VERSION_PATCH = $3;/g" include/constant.h

fs="git_tag.bash git_rm_tag.bash"
for f in $fs
do
  # mod version numbers in git_tag.bash
  sed -i -r "s/VERSION_MAJOR=.*/VERSION_MAJOR=$1/g" $f
  sed -i -r "s/VERSION_MINOR=.*/VERSION_MINOR=$2/g" $f
  sed -i -r "s/VERSION_PATCH=.*/VERSION_PATCH=$3/g" $f
done

#echo
#echo Updating docs
#echo
#cd docs
#./make_doc.bash
#cd ..

echo "Dont forget to use .."
echo
echo "first commit and push the code, then .."
echo "git tag -a "$1.$2.$3" -m 'my version "$version"'"
echo "git push origin --tags"
echo
echo ".. this informs git about the new version number"
echo
