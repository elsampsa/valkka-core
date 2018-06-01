import time
from valkka.valkka_core import TestThread
from valkka.api2 import setValkkaLogLevel, loglevel_debug, loglevel_crazy

setValkkaLogLevel(loglevel_crazy)

def cb1():
  print("      callback in python>")

def cb2(i):
  print("      callback in python>",i)

def cb3(*args):
  print("      callback in python>",args[0])


t=TestThread("eka")

# t.setCallback(cb1)
t.setCallback(cb2)
# t.setCallback(cb3)

t.startCall()
time.sleep(3)
print(">>waiting thread to stop")
t.stopCall()
print(">>thread stopped")
# time.sleep(3)
