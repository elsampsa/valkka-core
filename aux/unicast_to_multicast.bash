#!/bin/bash
# # Make a copy of this script and edit for your particular case

form="udp://"

# # *** EDIT THESE SECTIONS ***
# # cam1
# user="admin"
# passwd="12345"
# ip="192.168.0.157"
# # *************************
# # cam1
# user="admin"
# passwd="nordic12345"
# ip="192.168.0.24"
# # *************************
# # cam3
#user="admin"
#passwd="123456"
#ip="192.168.0.134"
# # *************************
# # cam4
user="admin"
passwd="nordic12345"
ip="192.168.1.41"
# # *******

target="224.1.168.91:50000" # multicast comes from here

com="ffmpeg -i rtsp://"$user":"$passwd"@"$ip" -c:v copy -map 0:0 -f rtp "$form""$target
$com
