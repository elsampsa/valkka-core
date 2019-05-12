## Setup

#### 1. Create a dev branch

Do this only once!
```
git branch dev
git checkout dev
# git branch --set-upstream-to=origin/dev
git push -u origin dev
```

Go to the branch:
```
git checkout dev
```


## Update main

#### 1. Commit dev branch (at dev):
```
git commit -a -m "etc"
git push
```

#### 2. Go to master:
```    
git checkout master
git merge dev
```

#### 3. Set version, recompile docs
```
./setver.bash major minor patch
docs/compile.bash
```

Now the code & docs are as they should be or the new version.


#### 4. Run tests at your build dir
```
source set_test_streams.bash
./run_tests.bash
```

Probe your test.out:
```
grep "ERROR SUMMARY" bin/test.out
grep -i "warning" bin/test.out
grep "definitely lost" bin/test.out
grep "indirectly lost" bin/test.out
```


#### 5. Run valkka-examples tests and checks
```
python3 copy_tutorial.py
```
then test the tutorial.

Test also ```test_studio_*.py``` programs.


#### 6. Edit README.md

Remember correct version number


#### 7. Commit master
```
git commit -a -m "new version"
git push
./git_tag.bash
```


#### 8. PPA update

Exclude files using "debian/source/options"

Edit "debian/changelog" (get the date stamp with "date -R")

Create debian package with
```
clear; debuild -S -sa
```
 
At the upper level directory:
```
dput ppa:sampsa-riikonen/valkka <source.changes> 
```
(dput accepts switch "-f" if you want to force upload)
   
(a comment)
```
problems?  check with
    gpg --verify

If keys are not working, edit file

    ~/.gnupg/gpg.conf

and add there

    default-key 284B974E9F5BC7FA
    trusted-key 284B974E9F5BC7FA
```



## Back to dev

#### 1. Let's reset the dev branch

```
git checkout dev
git rebase master
git commit -a -m "rebase from master" # actually, not needed
git push
```

A tip: copying a single file from dev to master (while being at master):
```
git checkout dev filename
```

