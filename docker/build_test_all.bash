# all="x86_ubuntu_18 x86_ubuntu_20 x86_ubuntu_22 armv8_ubuntu_18"
all="x86_ubuntu_18 x86_ubuntu_20 x86_ubuntu_22"
echo " " > build_test.log
for tag in $all
do
    echo
    echo BUILDING $tag
    echo
    echo "BUILD>>> "$tag >> build_test.log
    # this also clears any previous builds (int the previous/different os/architecture)
    ./build.bash $tag >> build_test.log 2>&1
    echo
    echo TESTING $tag
    echo
    echo "TEST>>> "$tag >> build_test.log
    ./test.bash $tag >> build_test.log 2>&1
done
grep -i -e "VALKKA_IMPORT_TEST" -e "BUILD>>>" -e "TEST>>>" build_test.log
