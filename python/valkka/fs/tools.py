"""
tools.py : tools for valkka.fs

 * Copyright 2017-2021 Valkka Security Ltd. and Sampsa Riikonen
 *
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 *
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    tools.py
@author  Sampsa Riikonen
@date    2021
@version 1.2.2 

@brief   tools for valkka.fs
"""
import time
import json
import os
import glob
import datetime

# https://en.wikipedia.org/wiki/GUID_Partition_Table#Partition_type_GUIDs
guid_linux_swap="0657FD6D-A4AB-43C4-84E5-0933C84B4F4F" # GPT partition table
mbr_linux_swap="82" # MBR partition table



def formatMstimestamp(mstime):
    t = datetime.datetime.fromtimestamp(mstime/1000)
    return "%s:%s" % (t.minute, t.second)


def findBlockDevices():
    """Scans all block devices for Linux swap partitions, returns key-value pairs like this:
    
    {'626C5523-2979-FD4D-A169-43D818FB0FFE': ('/dev/sda1', 500107862016)}
    
    Key is the uuid of the partition, value is a tuple of the device and its size
    
    To wipe out partition tables from a block device, use:
    
    ::
    
        wipefs -a /dev/devicename
        
    Block devices can be found in
    
    ::
    
        /sys/block/
        
        
    Find block device size in bytes:
        
    ::
    
        blockdev --getsize64 /dev/sdb
        
        
    Show block device id
        
    ::
    
        blkid /dev/sda1
        
    """
    from subprocess import Popen, PIPE
    devs={}
    # devs=[]
    block_devices = glob.glob("/sys/block/sd*")
    block_devices += glob.glob("/dev/nvme*")
    # block_devices = glob.glob("/sys/block/*")
    for block_device in block_devices:
        # print("block_device", block_device)
        devname = os.path.join("/dev", block_device.split("/")[-1]) # e.g. "/dev/sda"
        lis=["sfdisk", devname, "-J"]
        # print("lis>", lis)
        p = Popen(lis, stdout=PIPE, stderr=PIPE)
        st = p.stdout.read().decode("utf-8").strip()
        # print(">"+st+"<")
        if (len(st)>0):
            dic=json.loads(st)
            # print(dic)
            if "partitiontable" in dic:
                ptable = dic["partitiontable"]
                if "partitions" in ptable:
                    for partition in ptable["partitions"]:
                        if (partition["type"]==guid_linux_swap): # GTP partition table 
                            p = Popen(["blockdev", "--getsize64", devname], stdout=PIPE, stderr=PIPE)
                            st = p.stdout.read().decode("utf-8").strip()
                            devs[partition["uuid"].lower()]=(partition["node"], int(st))
                            """
                                "size"  : int(st),
                                "uuid"  : partition["uuid"]
                                }
                            """
                            # devs.append((partition["uuid"], partition["node"], int(st)))
                        """
                        elif (partition["type"]==mbr_linux_swap): # MBR partition table
                            p = Popen(["blockdev", "--getsize64", devname], stdout=PIPE, stderr=PIPE)
                            st = p.stdout.read().decode("utf-8").strip()
                            devs[partition["node"].lower()]=(partition["node"], int(st))
                        """
    return devs



 
