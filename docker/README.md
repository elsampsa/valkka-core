# Multiarchitecture/os build testing

*Test libValkka builds for multiple architectures using docker at your linux host.*

**NOT for building official docker images**

## 1. Prepare for arm

Prepare your emulation environment for armv8 with (of course, only if you have to):
```
./prepare.bash
```

## 2. Tags
```
x86_ubuntu_18       # not maintained
x86_ubuntu_20
x86_ubuntu_22
armv8_ubuntu_22
```

## 3. Create image
```
create_image.bash [TAG]
```
- Creates an image for a certain architecture/os
- The build environment set up in the image
- Environment for using/installing libValkka is set up in the image as well
- Does _not_ compile libValkka (yet)

## 4. Build libValkka in a container

We don't want to build libValkka while creating the image: if the build fails, there's not going to be an image!

It is better to share a tmp copy of libValkka from your filesystem with a running container: this way you can edit the code
live and iteratively edit/recompile in your target system on-demand until the compilation works ok.

First, edit [run.bash](run.bash) and change the variable ``MY_LOCAL_VALKKA_CORE_TMP_COPY``, then type:
```
./run.bash [TAG]
```

Now you have an interactive shell with the container, where you can:
```
./easy_build.bash
cd build_dir
# cmake .. # only if you need to run cmake again
make package && dpkg -i Valkka-*.deb && python3 -c 'from valkka.core import *'
```
If you want to run all that in a single command, use these (again, edit therein first ``MY_LOCAL_VALKKA_CORE_TMP_COPY``):
```
./build.bash [TAG]              # cleans up earlier builds & builds
./rebuild.bash [TAG]            # no cleaning up
```

## 5. Install the package & test

Installing and testing the package is done in a docker environment where nothing extra is installed
(not for building nor for using libValkka): this way we see if the produced .deb package has the correct
dependencies & if these are resolved / fetched correctly.

For installing libValkka .deb into a bare OS with the absolute minimum dependencies (as described by the .deb package):
```
./test.bash [TAG]
```
It also runs some minimal tests in the end.

You can also run:
```
./test_it.bash [TAG]
```
which does the same *and* finally give you an interactive shell to play around with.

## 6. Grand build & test

Combines steps 4 & 5 for all tags:
```
./build_test_all.bash
```

## 7. Notes

WARNING: everything (the build, deb package, etc.) is cached into your local directory
(``MY_LOCAL_VALKKA_CORE_TMP_COPY``), so do _not_ build for, say ubuntu 18
and then run the test in ubuntu 20.

