#!/bin/bash
# #
arch=$1
echo "CONFIGURE_FFMPEG: ARCH "$arch

# # WARNING!  This ffmpeg configuration MUST follow the ffmpeg version
release="3.4"
# # run this script in the directory where you have ffmpeg's "configure" script

# # BUG: even with this, "alsa" and "crystalhd" won't get disabled !
# everything="--disable-everything"

# basic="--disable-gpl --disable-audiotoolbox --disable-cuda --disable-cuda-sdk --disable-cuvid --disable-nvenc --disable-vaapi --disable-vdpau --enable-static --enable-shared --disable-pthreads"
# # a side note here: we don't want ffmpeg to use threads.  Valkka will reserve one thread per stream - that's enough (the idea is to have massive number of streams - not massive number of threads per stream)

# # allow multithreading decoder.  It's up to the user to control single vs. multithreading
basic="--disable-gpl --disable-audiotoolbox --enable-static --enable-shared"

# # BUG: with this, alsa does not get disabled !
ext_libs="--disable-alsa --disable-appkit --disable-avfoundation --disable-avisynth --disable-bzlib --disable-coreimage --disable-chromaprint --disable-frei0r --disable-gcrypt --disable-gmp --disable-gnutls --disable-iconv --disable-jack --disable-jni --disable-ladspa --disable-libass --disable-libbluray --disable-libbs2b --disable-libcaca --disable-libcelt --disable-libcdio --disable-libdc1394 --disable-libfdk-aac --disable-libflite --disable-libfontconfig --disable-libfreetype --disable-libfribidi --disable-libgme --disable-libgsm --disable-libiec61883 --disable-libilbc --disable-libkvazaar --disable-libmodplug --disable-libmp3lame --disable-libopencore-amrnb --disable-libopencore-amrwb --disable-libopencv --disable-libopenh264 --disable-libopenjpeg --disable-libopenmpt --disable-libopus --disable-libpulse --disable-librsvg --disable-librubberband --disable-librtmp --disable-libshine --disable-libsmbclient --disable-libsnappy --disable-libsoxr --disable-libspeex --disable-libssh --disable-libtesseract --disable-libtheora --disable-libtwolame --disable-libv4l2 --disable-libvidstab --disable-libvmaf --disable-libvo-amrwbenc --disable-libvorbis --disable-libvpx --disable-libwavpack --disable-libwebp --disable-libx264 --disable-libx265 --disable-libxavs --disable-libxcb --disable-libxcb-shm --disable-libxcb-xfixes --disable-libxcb-shape --disable-libxvid --disable-libxml2 --disable-libzimg --disable-libzmq --disable-libzvbi --disable-lzma --disable-decklink --disable-libndi_newtek --disable-mediacodec --disable-libmysofa --disable-openal --disable-opencl --disable-opengl --disable-openssl --disable-sndio --disable-schannel --disable-sdl2 --disable-securetransport --disable-xlib --disable-zlib"

# # disable all encoders
encoders="--disable-encoder=a64multi --disable-encoder=libgsm --disable-encoder=pcm_s32be --disable-encoder=a64multi5 --disable-encoder=libgsm_ms --disable-encoder=pcm_s32le --disable-encoder=aac --disable-encoder=libilbc --disable-encoder=pcm_s32le_planar --disable-encoder=aac_at --disable-encoder=libkvazaar --disable-encoder=pcm_s64be --disable-encoder=ac3 --disable-encoder=libmp3lame --disable-encoder=pcm_s64le --disable-encoder=ac3_fixed --disable-encoder=libopencore_amrnb --disable-encoder=pcm_s8 --disable-encoder=adpcm_adx --disable-encoder=libopenh264 --disable-encoder=pcm_s8_planar --disable-encoder=adpcm_g722 --disable-encoder=libopenjpeg --disable-encoder=pcm_u16be --disable-encoder=adpcm_g726 --disable-encoder=libopus --disable-encoder=pcm_u16le --disable-encoder=adpcm_g726le --disable-encoder=libshine --disable-encoder=pcm_u24be --disable-encoder=adpcm_ima_qt --disable-encoder=libspeex --disable-encoder=pcm_u24le --disable-encoder=adpcm_ima_wav --disable-encoder=libtheora --disable-encoder=pcm_u32be --disable-encoder=adpcm_ms --disable-encoder=libtwolame --disable-encoder=pcm_u32le --disable-encoder=adpcm_swf --disable-encoder=libvo_amrwbenc --disable-encoder=pcm_u8 --disable-encoder=adpcm_yamaha --disable-encoder=libvorbis --disable-encoder=pcx --disable-encoder=alac --disable-encoder=libvpx_vp8 --disable-encoder=pgm --disable-encoder=alac_at --disable-encoder=libvpx_vp9 --disable-encoder=pgmyuv --disable-encoder=alias_pix --disable-encoder=libwavpack --disable-encoder=png --disable-encoder=amv --disable-encoder=libwebp --disable-encoder=ppm --disable-encoder=apng --disable-encoder=libwebp_anim --disable-encoder=prores --disable-encoder=ass --disable-encoder=libx262 --disable-encoder=prores_aw --disable-encoder=asv1 --disable-encoder=libx264 --disable-encoder=prores_ks --disable-encoder=asv2 --disable-encoder=libx264rgb --disable-encoder=qtrle --disable-encoder=avrp --disable-encoder=libx265 --disable-encoder=r10k --disable-encoder=avui --disable-encoder=libxavs --disable-encoder=r210 --disable-encoder=ayuv --disable-encoder=libxvid --disable-encoder=ra_144 --disable-encoder=bmp --disable-encoder=ljpeg --disable-encoder=rawvideo --disable-encoder=cinepak --disable-encoder=roq --disable-encoder=cljr --disable-encoder=mjpeg --disable-encoder=roq_dpcm --disable-encoder=comfortnoise --disable-encoder=mjpeg_vaapi --disable-encoder=rv10 --disable-encoder=dca --disable-encoder=mlp --disable-encoder=rv20 --disable-encoder=dnxhd --disable-encoder=movtext --disable-encoder=s302m --disable-encoder=dpx --disable-encoder=mp2 --disable-encoder=sgi --disable-encoder=dvbsub --disable-encoder=mp2fixed --disable-encoder=snow --disable-encoder=dvdsub --disable-encoder=mpeg1video --disable-encoder=sonic --disable-encoder=dvvideo --disable-encoder=mpeg2_qsv --disable-encoder=sonic_ls --disable-encoder=eac3 --disable-encoder=mpeg2_vaapi --disable-encoder=srt --disable-encoder=ffv1 --disable-encoder=mpeg2video --disable-encoder=ssa --disable-encoder=ffvhuff --disable-encoder=mpeg4 --disable-encoder=subrip --disable-encoder=fits --disable-encoder=mpeg4_v4l2m2m --disable-encoder=sunrast --disable-encoder=flac --disable-encoder=msmpeg4v2 --disable-encoder=svq1 --disable-encoder=flashsv --disable-encoder=msmpeg4v3 --disable-encoder=targa --disable-encoder=flashsv2 --disable-encoder=msvideo1 --disable-encoder=text --disable-encoder=flv --disable-encoder=nellymoser --disable-encoder=tiff --disable-encoder=g723_1 --disable-encoder=nvenc --disable-encoder=truehd --disable-encoder=gif --disable-encoder=nvenc_h264 --disable-encoder=tta --disable-encoder=h261 --disable-encoder=nvenc_hevc --disable-encoder=utvideo --disable-encoder=h263 --disable-encoder=opus --disable-encoder=v210 --disable-encoder=h263_v4l2m2m --disable-encoder=pam --disable-encoder=v308 --disable-encoder=h263p --disable-encoder=pbm --disable-encoder=v408 --disable-encoder=h264_nvenc --disable-encoder=pcm_alaw --disable-encoder=v410 --disable-encoder=h264_omx --disable-encoder=pcm_alaw_at --disable-encoder=vc2 --disable-encoder=h264_qsv --disable-encoder=pcm_f32be --disable-encoder=vorbis --disable-encoder=h264_v4l2m2m --disable-encoder=pcm_f32le --disable-encoder=vp8_v4l2m2m --disable-encoder=h264_vaapi --disable-encoder=pcm_f64be --disable-encoder=vp8_vaapi --disable-encoder=h264_videotoolbox --disable-encoder=pcm_f64le --disable-encoder=vp9_vaapi --disable-encoder=hap --disable-encoder=pcm_mulaw --disable-encoder=wavpack --disable-encoder=hevc_nvenc --disable-encoder=pcm_mulaw_at --disable-encoder=webvtt --disable-encoder=hevc_qsv --disable-encoder=pcm_s16be --disable-encoder=wmav1 --disable-encoder=hevc_v4l2m2m --disable-encoder=pcm_s16be_planar --disable-encoder=wmav2 --disable-encoder=hevc_vaapi --disable-encoder=pcm_s16le --disable-encoder=wmv1 --disable-encoder=huffyuv --disable-encoder=pcm_s16le_planar --disable-encoder=wmv2 --disable-encoder=ilbc_at --disable-encoder=pcm_s24be --disable-encoder=wrapped_avframe --disable-encoder=jpeg2000 --disable-encoder=pcm_s24daud --disable-encoder=xbm --disable-encoder=jpegls --disable-encoder=pcm_s24le --disable-encoder=xface --disable-encoder=libfdk_aac --disable-encoder=pcm_s24le_planar --disable-encoder=xsub --disable-encoder=xwd --disable-encoder=yuv4 --disable-encoder=zmbv --disable-encoder=y41p --disable-encoder=zlib"

# # architecture-dependent configuration:
# # 
# # let's add some libav hw accelerators # https://trac.ffmpeg.org/wiki/HWAccelIntro
# # 1. VAAPI: using requires "i965-va-driver", compilation requires "libva-dev", for monitoring "intel-gpu-tools" --> intel_gpu_top

# by default, enable vaapi
hw_decoders="--enable-vaapi --disable-cuda --disable-cuda-sdk --disable-cuvid --disable-nvenc --disable-vdpau"
if [[ $arch = *"arm"* ]]; then
    # 1. VAAPI
    echo "FFMPEG: disabling VAAPI for arm architecture"
    hw_decoders="--disable-vaapi --disable-cuda --disable-cuda-sdk --disable-cuvid --disable-nvenc --disable-vdpau"
fi

# # These libraries are not listed with "./configure -h".  This finally disables crystalhd..
ext_libs2="--disable-crystalhd"

# # not under external libraries section of "./configure -h"
extras="--disable-v4l2_m2m"

# # works ok with: basic, ext_libs, ext_libs2, extras

cd ffmpeg; ./configure $everything $basic $ext_libs $ext_libs2 $encoders $extras $hw_decoders
if [ $? -ne 0 ]
then
  echo "configure_ffmpeg.bash: Could not configure!"
  exit 1
fi

## fix config.h so that it doesn't use sysctl
sed -i -r "s/#define HAVE_SYSCTL 1/#define HAVE_SYSCTL 0/g" config.h
