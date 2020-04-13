
## Generate entries into changelog

Bionic == 18
Xenial == 16

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
Build-Depends: build-essential, libc6-dev, yasm, cmake, pkg-config, swig, libglew-dev, mesa-common-dev, libstdc++-5-dev, python3-dev, python3-numpy, libasound2-dev
Standards-Version: 3.9.7
Homepage: http://www.iki.fi/sampsa.riikonen

Package: valkka
Architecture: any
Depends: python3, mesa-utils, glew-utils, python3-numpy, v4l-utils, python3-pip
Description: Valkka
 OpenSource Video Management for Linux
```

## libstdc++

That's wrong!

```
build-essential => g++ => g++-7 => libstdc++-7-dev
```

So there should be no dependency on libstdc++N-dev

How about just "libstdc++N" ?

```
apt-cache depends package-name
```

Let's change:

```
Build-Depends: build-essential, libc6-dev, yasm, cmake, pkg-config, swig, libglew-dev, mesa-common-dev, libstdc++-5-dev, python3-dev, python3-numpy, libasound2-dev
Depends: python3, mesa-utils, libstdc++5, glew-utils, python3-numpy, v4l-utils, python3-pip
```

into:

```
Build-Depends: build-essential, yasm, cmake, pkg-config, swig, libglew-dev, mesa-common-dev, python3-dev, python3-numpy, libasound2-dev
Depends: python3, mesa-utils, glew-utils, python3-numpy, v4l-utils, python3-pip
```

