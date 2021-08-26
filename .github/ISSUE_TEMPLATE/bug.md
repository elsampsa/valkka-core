---
name: Bug Report
about: For bugs
title: ''
labels: ''
assignees: ''

---

## Bug report

First, make sure that:

- You are running the latest version of libValkka
- You have experimented with the Qt test suite's [test_studio_1.py](https://elsampsa.github.io/valkka-examples/_build/html/testsuite.html)
- ..and understand at least how buffering time and number of pre-reserved frames are related to each other
- You have read the [common problems](https://elsampsa.github.io/valkka-examples/_build/html/pitfalls.html) section

Your bug report should include:

1. A copy-paste of the salient features of the terminal output
    ```
    like
    this
    ```

2. Indicate the graphics driver are you using:  Intel, nvidia or opensource nvidia (aka "nouveau")
3. Indicate the kind of media stream you are dealing with (as reported by your IP camera's web-interface):
    - Resolution
    - Frames per second
    - Video codec (should be H264!)
    - Does your stream include audio?
4. If you are streaming multiple streams, what is your _total_ framerate?
5. Your architecture (arm version or amd64) and the version and name of your linux distro
6. Did you install with ``apt-get`` or did you compile yourself?

7. Ascii art of the filterchain (like in the tutorial)
8. A **minimal, single-file code that reproduces the issue**:
    - Use github gists
    - Must be a stand-alone python executable, something that can be launched (by me) from the command line, i.e.:
      ```
      python3 your_python_file.py
      ```
    - Keep it _very_ brief

Please remember that all ValkkaFS-related features are still pretty experimental.
