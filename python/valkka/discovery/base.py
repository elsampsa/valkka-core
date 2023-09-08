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
@version 1.5.4 

@brief   Discovery module for onvif cameras, using wsdiscovery and brute-force scan
"""
import sys, time
import signal
import os, errno, select
import inspect
import asyncio
import re
import traceback
from subprocess import Popen, PIPE, check_output, CalledProcessError, STDOUT
import pty

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

    timeout = 3

    if verbose:
        print("arp-scan probe: trying", ip)

    writer = None
    ok = True
    st = options_str % (ip, port)
    st_ = st.encode("utf-8")
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout = timeout)
    except asyncio.TimeoutError:
        # traceback.print_exc()
        if verbose: print("arp-scan probe: timeout", ip)
        ok = False
    except ConnectionRefusedError:
        if verbose: print("arp-scan probe: connection refused", ip)
        # traceback.print_exc()
        ok = False
    except Exception as e:
        if verbose: print("arp-scan probe: exception", e)
        # traceback.print_exc()
        ok = False
    
    if not ok:
        if writer is not None: writer.close()
        return None

    if verbose: print("arp-scan probe: writing", ip)
    writer.write(st_)
    try:
        res = await asyncio.wait_for(reader.read(1024), timeout = timeout)
    except asyncio.TimeoutError:
        if verbose: print("arp-scan probe: timeout at port 554")
        ok = False
    except Exception as e:
        traceback.print_exc()
        ok = False
    else:
        if verbose: print("arp-scan probe: parsing response for", ip)
        header, res = parse_http_resp(res.decode("utf-8").lower())
        if verbose: print("arp-scan probe: reply for", ip, ":", header, res)
        # if header.find("200 ok") > -1: # most likely not authorized!
        if header.find("rtsp") > -1:
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



def parse_ip_address_output(output) -> dict:
    """Sample output:
    
    ::
    
        {'br-2ad50d95054a': {'attributes': 'NO-CARRIER,BROADCAST,MULTICAST,UP',
                         'subnets': [('172.19.0.1', '16')]},
         'br-2cddc3bcd5fe': {'attributes': 'NO-CARRIER,BROADCAST,MULTICAST,UP',
                             'subnets': [('172.23.0.1', '16')]},
         'br-5100bab9ac22': {'attributes': 'NO-CARRIER,BROADCAST,MULTICAST,UP',
                             'subnets': [('172.24.0.1', '16')]},
         'br-52b32b08932e': {'attributes': 'NO-CARRIER,BROADCAST,MULTICAST,UP',
                             'subnets': [('172.20.0.1', '16')]},
         'br-7d96055bc7ee': {'attributes': 'NO-CARRIER,BROADCAST,MULTICAST,UP',
                             'subnets': [('172.18.0.1', '16')]},
         'docker0': {'attributes': 'NO-CARRIER,BROADCAST,MULTICAST,UP',
                     'subnets': [('172.17.0.1', '16')]},
         'enx4865ee147a39': {'attributes': 'BROADCAST,MULTICAST,UP,LOWER_UP',
                             'subnets': []},
         'lo': {'attributes': 'LOOPBACK,UP,LOWER_UP', 'subnets': [('127.0.0.1', '8')]},
         'wlp2s0': {'attributes': 'BROADCAST,MULTICAST,UP,LOWER_UP',
                    'subnets': [('192.168.1.135', '24')]}}
    
    P. S. Thanks chat-gpt!
    """
    interfaces = {}
    current_interface = None

    for line in output.splitlines():
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if it's a new interface line
        if line[0].isdigit():
            parts = line.split(": ")
            interface_name = parts[1].split()[0]
            current_interface = interface_name
            interfaces[current_interface] = {'subnets': [], 'attributes': ''}
            attributes = re.findall(r'<([^>]+)>', line)[0]
            interfaces[current_interface]['attributes'] = attributes

        
        # Check for subnets information within the interface
        elif line.startswith('inet '):
            if current_interface is not None:
                subnet_info = re.findall(r'inet ([\d.]+)/(\d+)', line)
                for subnet, mask in subnet_info:
                    interfaces[current_interface]['subnets'].append((subnet, mask))

    return interfaces


def getInterfaces() -> dict:
    """Returns all 'normal' interfaces (excluding NO-CARRIER and LOOPBACK interfaces)
    
    Example output:
    
    ::
    
        {'wlp2s0': [('192.168.1.135', '24')], 'enx4865ee147a39': []}
    
    """
    try:
        ip_output = check_output(['ip', 'address'], universal_newlines=True)
        interfaces = parse_ip_address_output(ip_output)
        # pprint(interfaces)
    except CalledProcessError as e:
        print(f"Error executing 'ip address' command: {e}")
        return {}
    
    dic={}
    for name, interface in interfaces.items():
        # print(">",name, interface["attributes"])
        if "NO-CARRIER" not in interface["attributes"] and "LOOPBACK" not in interface["attributes"]:
            # print(name, interface["subnets"])
            dic[name] = interface["subnets"]
    return dic



class Alarm(Exception):
    pass

def alarm_handler(signum, frame):
    raise Alarm

def dummy_handler(signum, frame):
    pass


def runARPScan(exclude_list = [], exclude_interfaces = [], max_time_per_interface=10, verbose=False) -> list:
    """brute-force port 554 & 8554 scan & RTSP OPTIONS test ping in parallel

    Returns a list of ip addresses

    :param exclude_list: ip addresses to be exluded
    :param exclude_interfaces: interfaces to be excluded from the scan
    :param max_time_per_interface: (int) how many secs to be spent max in each interface (default: 10)

    quicktest:

    ::

        python3 -c "from valkka.discovery import runARPScan; print(runARPScan(verbose=False))"

    """
    # verbose = True
    # verbose = False
    reg = re.compile("^(\d*\.\d*.\d*.\d*)") # match ipv4 address
    lis = []
    # signal.signal(signal.SIGALRM, alarm_handler) # only works in main thread and this might run in a multithread
    for name, subnets in getInterfaces().items():
        if name in exclude_interfaces:
            continue
        print("runARPScan: scanning interface", name)
        for ip, iplen in subnets:
            # print(name+" "+ip+"/"+iplen) # stdbuf -oL 
            comst = f"arp-scan -g -retry=2 --interface={name} {ip}/{iplen}" # e.g. eno1 192.168.30.149/24
            if verbose: print("runARPScan: starting for", comst)
            stdout = ""
            """
            try:
                with Popen(comst.split(), stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True) as p:
                    for line in p.stdout:
                        if verbose: print(">", line.strip())
                        stdout += line
            except FileNotFoundError:
                print(f"arp-scan failed for '{comst}'.  Install with 'sudo apt-get install arp-scan'.  You also need extra rights to run it: 'sudo chmod u+s /usr/sbin/arp-scan'")
            except Alarm:
                print(f"arp-scan for '{comst}' took more than n secs - aborting")
                continue
            if p.returncode > 0:
                print(f"arp-scan failed for  '{comst}'.  You might need extra rights to run it: 'sudo chmod u+s /usr/sbin/arp-scan'")
                # print("arp-scan error:", stderr.decode("utf-8"))
            # parse the output of arp-scan
            # for line in stdout.decode("utf-8").split("\n"):
            """
            # as per: https://stackoverflow.com/questions/12419198/python-subprocess-readlines-hangs
            # the ONLY way to read process output on-the-fly
            master_fd, slave_fd = pty.openpty()
            # signal.alarm(max_time_per_interface) # in secs
            proc = Popen(comst.split(), stdin=slave_fd, stdout=slave_fd, stderr=STDOUT, close_fds=True)
            os.close(slave_fd)
            timecount = 0
            try:
                while 1:
                    if timecount > max_time_per_interface:
                        print(f"WARNING: arp-scan '{comst}' took more than {max_time_per_interface} secs - aborting")
                        break        
                    t0 = time.time()
                    try:
                        r, w, e = select.select([ master_fd ], [], [], 1)
                        dt = time.time() - t0
                        timecount += dt
                        if verbose: print("timecount>", timecount)
                        if master_fd in r:
                            data = os.read(master_fd, 512)
                        else:
                            continue
                    except OSError as e:
                        if e.errno != errno.EIO:
                            raise
                        break # EIO means EOF on some systems
                    else:
                        if not data: # EOF
                            break
                        # if verbose: print('>' + repr(data))
                        if verbose: print('>', data.decode("utf-8"))
                        stdout += data.decode("utf-8")
            finally:
                os.close(master_fd)
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            # cancel the alarm:
            # print("cancel alarm")
            # signal.alarm(0)

            for line in stdout.split("\n"):
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
                            if verbose: print("runARPScan: appending", ip)
                            lis.append(ip)
            if verbose: print("runARPScan: finished for", comst)

    if len(lis) < 1:
        if verbose: print("arp-scan: did not find anything")
        return []

    coros = [probe(ip, 554) for ip in lis]
    coros += [probe(ip, 8554) for ip in lis]

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


def runARPScanParallel(exclude_list = [], exclude_interfaces = [], max_time_per_interface=10, verbose=False) -> list:
    pass
    # TODO!    


if __name__ == "__main__":
    # ips = runWSDiscovery()
    # print(ips)
    ips = runARPScan()
    print(ips)
