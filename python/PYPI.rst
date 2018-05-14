Valkka - OpenSource Video Surveillance and Management for Linux
---------------------------------------------------------------

The goal of this project is to provide a library for creating open source video surveillance, management and analysis systems (VMAs) in Linux environment.  The idea is create VMA systems with graphical user interfaces (GUIs) using the combination of python3 and Qt (i.e. PyQt).

Installation instructions, demo programs and API tutorial are available at https://elsampsa.github.io/valkka-examples/

The python source package includes pre-compiled shared libraries, compiled on **Ubuntu 16.04.4 LTS, x86_64**.  There is no guarantee that they work on another linux distribution.

To compile from source and building the python package yourself, please go to https://github.com/elsampsa/valkka-core

.. This python binary (.whl) package includes a pre-compiled shared library libValkka.so, that contains statically linked (i.e. no external dependencies) live555 and the ffmpeg libraries (both with LGPL licenses).
