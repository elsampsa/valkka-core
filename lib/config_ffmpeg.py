#!/usr/bin/python3
"""
* Creates script "run_configure.bash" that launches ffmpeg's "configure" script with correct parameters (enabling/disabling stuff)
* Run in the same directory where you have ffmpeg's configure script
(C) Sampsa Riikonen / 2017
"""
import subprocess
import os
import re

def features(switch, adstring="", remove=[]):
  p=subprocess.Popen(["./configure",switch],stdout=subprocess.PIPE)
  st=p.stdout.read()
  fst=""
  for s in st.split():
    ss=s.decode("utf-8")
    ok=True
    for rem in remove:
      if (ss.find(rem)!=-1):
        ok=False
    if ok: fst+=adstring+ss+" "
  return fst
  

def disable_external():  
  p=subprocess.Popen(["./configure","-h"],stdout=subprocess.PIPE)
  st=p.stdout.read().decode("utf-8")
  # print(st)
  # stop
  # find some text tags from the configure output:
  # i1=st.find("External library support:") # older ffmpeg .. version 3.0, etc.
  i1=st.find("themselves, not all their features will necessarily be usable by FFmpeg.") # main branch, latest
  # themselves, not all their features will necessarily be usable by FFmpeg.
  i2=st.find("The following")
  # i2=st.find("Toolchain options:")
  if (i1<0 or i2<0): 
    print("Could not find text tags from ./configure -h : edit this python script")
    return
  st=st[i1:i2]
  # """ # debugging ..
  # print(i1,i2)
  # stop
  # """
  p=re.compile('--(enable|disable)-(\S*)')
  switches=[]
  for sw in p.findall(st):
    if (sw[1] not in switches):
      # print(sw[1]) # debugging
      switches.append(sw[1])
  fst=""
  for sw in switches:
    fst+="--disable-"+sw+" "
  return fst

st ="./configure "

def create_script():
  # # use this to get statically linked executables: (but then your archive .a files don't have position independent code, i.e. compiled with "-fPIC")
  # st+='--disable-everything --disable-doc --disable-gpl --disable-pthreads --pkg-config-flags="--static" --disable-shared --enable-static '
  # # you might want to use this in order to get an ffmpeg executable that is consistent with the (static) libraries you are using in valkka

  # # use this to get statically linked arhive .a files , compiled with "-fPIC" :
  st+='--disable-everything --disable-doc --disable-gpl --disable-pthreads --pkg-config-flags="--static" --enable-shared --enable-static '
  # # these can be used to "bake ffmpeg into" valkka

  st+="--enable-protocol=file --enable-protocol=unix --enable-protocol=data "
  st+="--enable-encoder=rawvideo "

  # st+="--disable-asm " # WARNING: only for debugging
  # st+="--enable-ffplay " # won't get build if external libraries missing

  # WARNING: if you set --disable-shader the code is not compiled with "-fPIC" => can't include .a archive files into shared library
  # And remember, commands can be checked with "make -n"
  st+= disable_external()
  st+="--disable-crystalhd --disable-v4l2_m2m --disable-alsa " # these did not get disabled..? no ***king way to remove alsa
  st+="--disable-avdevice --disable-avfilter "

  # disable all hw decoders
  # st+="--disable-d3d11va --disable-dxva2 --disable-vaapi --disable-vda --disable-vdpau --disable-videotoolbox "
  st+="--disable-d3d11va --disable-dxva2 --disable-vaapi --disable-vdpau --disable-videotoolbox "

  # print(">>",st)
  st+= features("--list-decoders",adstring="--enable-decoder=", remove=["vdpau","crystalhd","zlib"])
  # st+= features("--list-decoders",adstring="--disable-decoder=") # WARNING: just for debugging .. disable all decoders

  st+= features("--list-muxers",  adstring="--enable-muxer=")
  st+= features("--list-demuxers",adstring="--enable-demuxer=")
  st+= features("--list-parsers", adstring="--enable-parser=")

  f=open("run_configure.bash","w")
  f.write("#!/bin/bash\n")
  f.write(st+"\n")
  f.close()
  os.system("chmod a+x run_configure.bash")
  print("\nNext run ./run_configure.bash\n")

  """
  For cleaning up .a and .so files, use
  find -name *.a -exec ls {} \;
  find -name *.so* -exec ls {} \;
  """


# *** uncomments one of the following ***

# create_script() # this creates a "run_configure.bash" .. run it in the same directory where you have the ffmpeg "configure" script

# print(disable_external()) # prints switches to disable external libraries

print(features("--list-encoders",adstring="--disable-encoder=")) # prints a switch to disable all encoders

