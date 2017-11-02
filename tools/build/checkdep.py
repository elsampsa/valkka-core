#!/usr/bin/python3
import subprocess
import os
import sys
import re

# fname=sys.argv[1]
fname="lib/libValkka.so"

# find out our architecture
p=subprocess.Popen(["dpkg","--print-architecture"],stdout=subprocess.PIPE)
st=p.stdout.read()
arch=st.decode("utf-8").strip()
#print(arch)
#stop

p=subprocess.Popen(["objdump","-p",fname],stdout=subprocess.PIPE)
st=p.stdout.read()

dynlibs=[]

for line in st.decode("utf-8").split("\n"):
  # print(line)
  if (line.find("NEEDED")>-1):
    # print(">",line)
    st=line.split()[1].strip()
    # print(">",st)
    dynlibs.append(st)
    
# now we have the dynamic library names

libpkg={}
for dynlib in dynlibs:
  # print(dynlib)
  p=subprocess.Popen(["dpkg","-S",dynlib],stdout=subprocess.PIPE)
  lines=p.stdout.read().decode("utf-8").split("\n")
  found=False
  for line in lines:
    st=line.strip()
    #print(st)
    #print(arch)
    #print(">>",st.find("amd64"))
    if (found==False and st.find(arch)>-1):
      # print(st)
      found=True
      pkgname=st.split(":")[0]
      libpkg[dynlib]=pkgname
      
    
sst=""
for key in libpkg:
  print("shared object",key,"==>","package",libpkg[key])
  p=subprocess.Popen(["apt-cache","show",libpkg[key]],stdout=subprocess.PIPE)
  st=p.stdout.read().decode("utf-8")
  reg=re.compile("Version: (\S*)-")
  try:
    res=reg.findall(st)[0]
  except e:
    res="?"
    print(str(e))
  print("   version:",res)
  sst+=libpkg[key]+"(>= "+res+"), "
  
sst='"'+sst[0:-2]+'"'
print("\n",sst,"\n")
