# New version checklist

## Basic workflow

*At dev branch*

Finally, run semiautomated tests

*At master branch*

merge from dev

Edit README.md --> log latest version

Edit Release.txt --> log latest version

Set version:
```
script/setver.bash major minor patch
```

New PPA packages:
```
cd debian
./sendall.bash major.minor.patch description
```

New git tag (launches the CI pipeline)
```
./git_tag.bash
```

checkout to dev branch & rebase to master

continue at valkka-examples:

*At master branch*

merge from dev

```
./setver.bash
./git_tag.bash
```

checkout to dev branch & rebase to master

## Bonus stuff

At valkka-examples:
```
python3 quicktest.py
python3 copy_tutorial.py
cd tmp
./run_tutorial.bash
```

Check output, say, with:
```
grep -i "seg" test.out
```

## Misc

```
problems?  check with
    gpg --verify

If keys are not working, edit file

    ~/.gnupg/gpg.conf

and add there

    default-key 284B974E9F5BC7FA
    trusted-key 284B974E9F5BC7FA
```
