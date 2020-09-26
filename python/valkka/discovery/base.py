"""base.py : Discovery module for onvif cameras, using wsdiscovery and brute-force scan

Copyright 2017-2019 Valkka Security Ltd. and Sampsa Riikonen.

Authors: Sampsa Riikonen (sampsa.riikonen@iki.fi)

This particular file, referred below as "Software", is licensed under the MIT LICENSE:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

@file    base.py
@author  Sampsa Riikonen
@date    2019
@version 1.0.0 

@brief   Discovery module for onvif cameras, using wsdiscovery and brute-force scan
"""
import sys
import os
import inspect
import asyncio
import re
import traceback
from subprocess import Popen, PIPE 

from valkka.discovery.wsdiscovery import WSDiscovery, QName, Scope

"""
- Run WSDiscovery
- For each discovered device, poke port 554 with an RTSP OPTIONS request
- ..but do that in parallel, using asyncio
- For each found IP camera, try a few user & password combos
- In fact, no need to do that brute-force port 554 scan if we filter
  ws-discovery results having /onvif/ in the device list
"""

options_str = """OPTIONS rtsp://%s:%i RTSP/1.0\r
CSeq: 1\r
User-Agent: libValkka\r
\r
\r"""

def parse_http_resp(resp: str):
    reg = re.compile("(^\S*):(.*)")
    fields = resp.split("\r\n")
    output = {}
    for field in fields[1:]: # ignore "GET / HTTP/1.1" and the like
        try:
            m = reg.match(field)
            if m is None:
                continue
            key = field[m.start(1):m.end(1)]
            value = field[m.start(2):m.end(2)]
        except IndexError:
            continue
        else:
            output[key] = value
    return fields[0], output



async def probe(ip, port):
    """Send an RTSP OPTIONS request to ip & port
    """
    verbose = False
    # verbose = True

    if verbose:
        print("arp-scan probe", ip)

    writer = None
    ok = True
    st = options_str % (ip, port)
    st_ = st.encode("utf-8")
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout = 1)
    except asyncio.TimeoutError:
        # traceback.print_exc()
        ok = False
    except ConnectionRefusedError:
        # traceback.print_exc()
        ok = False
    except Exception as e:
        # traceback.print_exc()
        ok = False
    
    if not ok:
        if writer is not None: writer.close()
        return None

    writer.write(st_)
    try:
        res = await asyncio.wait_for(reader.read(1024), timeout = 1)
    except asyncio.TimeoutError:
        ok = False
    except Exception as e:
        traceback.print_exc()
        ok = False
    else:
        header, res = parse_http_resp(res.decode("utf-8").lower())
        if header.find("200 ok") > -1:
            if verbose:
                print("\nSuccess at %s\n" % (ip))
            pass
        else:
            ok = False

    writer.close()
    if ok:
        return ip
        

def runWSDiscovery():
    reg = re.compile("^http:\/\/(\d*\.\d*\.\d*\.\d*).*\/onvif")
    wsd = WSDiscovery()
    wsd.start()
    ws_devices = wsd.searchServices()
    ip_addresses = []
    for ws_device in ws_devices:
        for http_address in ws_device.getXAddrs():
            m = reg.match(http_address)
            if m is None:
                continue
            else:
                ip_address = http_address[m.start(1):m.end(1)]
                # print(ip_address)
                ip_addresses.append(ip_address)

        """ws_device.
        getXAddrs
            returns a list like this:
            ['http://192.168.1.57/onvif/device_service', 'http://[fe8d::8ef7:48ee:feed:a84d]/onvif/device_service']

        getEPR
        getInstanceId
        getMessageNumber
        getMetadataVersion
        getScopes
        getTypes
        """
    wsd.stop()
    return ip_addresses


def runARPScan(exclude_list = []):
    """brute-force port 554 scan & RTSP OPTIONS test ping in parallel
    """
    reg = re.compile("^(\d*\.\d*.\d*.\d*)") # match ipv4 address
    lis = "arp-scan --localnet".split()
    try:
        p = Popen(lis, stderr = PIPE, stdout = PIPE)
    except FileNotFoundError:
        print("arp-scan failed.  You need extra rights to run it, try: 'sudo chmod u+s /usr/sbin/arp-scan'")
        return []
    stdout, stderr = p.communicate()
    lis = []
    
    # parse the output of arp-scan
    for line in stdout.decode("utf-8").split("\n"):
        l = line.split()
        # print(">>", l)
        if len(l) > 2:
            col = l[0]
            m = reg.match(col)
            if m is None:
                continue
            else:
                ip = col[m.start(1):m.end(1)]
                # print(ip)
                if ip not in exclude_list:
                    lis.append(ip)
    
    coros = [probe(ip, 554) for ip in lis]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    """
    finished, unfinished = asyncio.get_event_loop().run_until_complete(
        asyncio.wait(coros)
        )
    """
    finished, unfinished = asyncio.get_event_loop().run_until_complete(
        asyncio.wait(coros)
    )

    ips = []
    for task in finished:
        ip = task.result()
        if ip is not None:
            ips.append(ip)

    loop.close()
    return ips


if __name__ == "__main__":
    # ips = runWSDiscovery()
    # print(ips)
    ips = runARPScan()
    print(ips)
