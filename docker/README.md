## Multiarchitecture builds

This is for build testing only.

Build libValkka for multiple architectures using docker at Ubuntu x86.

First, prepare your emulation environment with:
```
./prepare.bash
```

Scripts ```cont_*.bash``` create containers for various architectures.

Scripts ```build_*.bash``` compile libValkka in those architectures.  They bind-mount the whole valkka source code before compiling.

For creating production containers, please take a look at valkka_examples repo and it's docker/ subdirectory.

## TODO

valkka_examples/docker

- source builds for x86 & arm, including darknet-api
- TODO: test darknet-api with arm
