
*some notes for myself*

## About the build system

It operates from the valkka main directory with:

    make -f debian/rules clean

The directories used in make are all relative to the valkka main dir

## Installing

Binary packages and their Python3 bindings are provided for latest Ubuntu distributions.  Subscribe to our repository with: 

    sudo apt-add repository ppa:sampsa-riikonen/valkka
    
and then do:
    
    sudo apt-get install valkka
 
## Updating ppa
 

At valkka main directory, do:
 
    clear; debuild -S -sa
    
At the upper level directory, do:
 
    dput ppa:sampsa-riikonen/valkka <source.changes> 

The resulting tar.gz archive can be unpacked and tested (i.e. with make -f etc.)
 
 
## Packages for various Ubuntu distros

The first line of file debian/changelog must be edited for each one separately:

    valkka (0.5.0-1) xenial; urgency=low
    
Use the following naming scheme:  for xenial:

    valkka (0.5.0-0ubuntu1-xenialppa1) xenial; urgency=low
    
and for bionic:

    valkka (0.5.0-0ubuntu1-bionicppa1) bionic; urgency=low

etc.

For debugging the ppa build, just change always that last number (i.e. ppa1 => ppa2, etc.)

https://askubuntu.com/questions/30145/ppa-packaging-having-versions-of-packages-for-multiple-distros

## Problems

It seems that the *i386* build could not find sys.h.  Added build-essential etc. to the build-depends field, see [here](https://ubuntuforums.org/showthread.php?t=1922575)

## Resources

An essential article for using CMake with the ppa system [here](http://simmesimme.github.io/lessons/2011/11/02/ppa-launchpad-cmake)
Creating packages for various distros, see [here](https://askubuntu.com/questions/30145/ppa-packaging-having-versions-of-packages-for-multiple-distros)

