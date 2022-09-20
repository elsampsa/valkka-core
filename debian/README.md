
## Generate entries into changelog

Jammy = 22

Focal = 20

Bionic = 18

(Xenial = 16)

```
addlog.py 0.13.2 bionic new awesome version
# copy-paste from terminal to changelog
./rundeb.bash
addlog.py 0.13.2 xenial new awesome version
# copy-paste from terminal to changelog
./rundeb.bash
```

## An example control file:

```
Source: valkka
Section: libs
Priority: optional
Maintainer: Sampsa Riikonen <sampsa.riikonen@iki.fi>
Build-Depends: build-essential, yasm, cmake, pkg-config, swig, libglew-dev, mesa-common-dev, libstdc++-5-dev, python3-dev, python3-numpy, libasound2-dev, libstdc++-6-dev, libc6-dev
Standards-Version: 3.9.7
Homepage: http://www.iki.fi/sampsa.riikonen

Package: valkka
Architecture: any
Depends: python3, mesa-utils, glew-utils, python3-numpy, v4l-utils, python3-pip, libstdc++6
Description: Valkka
 OpenSource Video Management for Linux
```

## libstdc++

Which version?

In my system:
```
build-essential => g++ => g++-7 => libstdc++-7-dev
```

So there should be no dependency on libstdc++N-dev

How about just "libstdc++N" ?

```
apt-cache depends package-name
```

Removed libstdc++ and libc dependencies from both Build-Depends & Depends.  Compiled fine.




