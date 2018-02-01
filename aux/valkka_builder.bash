# [substitute here #!/bin/bash if you know what you're doing]
#
# A script that ..
# * Installs some packages
# * Downloads live555 and ffmpeg
# * Downloads valkka-core and compiles it
# * Downloads valkka-examples
#
# This script is tested - kind of
#
# *** Read and understand.  Use with caution.  Absolutely no warranty. ***
#
# Creates directory $HOME/C and creates subdirectories under that - things might get overwritten there
#

email="your_email"
github_user="your_username"

echo
echo "Running apt-get"
echo
sudo apt-get install yasm git swig python3-pip cmake libx11-dev libglew-dev libasound2-dev ffmpeg
echo
echo "Running pip3"
echo
pip3 install --upgrade ipython numpy
echo
echo "Getting valkka-core"
echo
mkdir C
cd C
mkdir valkka_builds
# ****** If you have not cloned valkka-core already ********
git clone https://github.com/elsampsa/valkka-core.git
# **********************************************************
cd valkka-core
git config --global push.default simple
# If you have not done this:
git config --global user.email $email
git config user.name $github_user
./new_build.bash dev
mv build_dev ../valkka_builds/
cd ~/C
echo
echo "Getting Live555"
echo
wget http://www.live555.com/liveMedia/public/live555-latest.tar.gz
gunzip live555-latest.tar.gz
tar xvf live555-latest.tar
cd live
./genMakefiles linux-64bit
make
cd ~/C/valkka-core/lib
./ln_live.bash ~/C/live
cd ~/C
echo
echo "Getting ffmpeg"
echo
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
git checkout n3.4
cp ~/C/valkka-core/lib/run_config_3_4_.bash .
./run_config_3_4_.bash
make -j 2
cd ~/C/valkka-core/lib/
./ln_ffmpeg.bash ~/C/ffmpeg
echo
echo "Preparing python bindings"
echo
cd ~/C/valkka-core/python/ext
./ln_ffmpeg.bash ~/C/ffmpeg
./ln_live.bash ~/C/live
cd ~/C/valkka-core/python
./make_links.bash
./make_swig_file.bash
echo
echo "Getting valkka-examples"
echo
cd ~
git clone https://github.com/elsampsa/valkka-examples
cd valkka-examples
git config user.name $github_user
echo
echo "1) copy run_cmake.bash to: ~/C/valkka_builds/build_dev/"
echo "2) Go there and run: ./run_cmake.bash"
echo "3) Run: make"
echo "4) Run: source test_env.bash"
echo "5) cd ~/C/valkka-core/python"
echo "6) ./compile3.bash"
echo "7) source test_env.bash"
echo
