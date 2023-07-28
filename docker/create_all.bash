# accepts "--no-cache" parameter
# all="x86_ubuntu_18 x86_ubuntu_20 x86_ubuntu_22 armv8_ubuntu_18"
all="x86_ubuntu_20 x86_ubuntu_22"
echo " " > create.log
for tag in $all
do
    echo
    echo CREATING $tag
    echo
    echo "CREATE>>> "$tag >> create.log
    ./create_image.bash $tag $1 >> create.log 2>&1
done
grep -i "successfully" create.log 
