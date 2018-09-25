"""
logging.py : Interface to cpp-level logging

 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    logging.py
@author  Sampsa Riikonen
@date    2017
@version 0.5.4 
  
@brief Interface to cpp-level logging
"""

# can't get the namespace from cpp (logging.h) with swig, so redefine here
loglevel_silent =0
loglevel_normal =1
loglevel_debug  =2
loglevel_crazy  =3

from valkka import valkka_core as core

def setFFmpegLogLevel(i):
  core.ffmpeg_av_log_set_level(i)
  
  
def setValkkaLogLevel(i=loglevel_normal):
  if   (i==loglevel_silent):   # just fatal messages
    core.fatal_log_all()
    core.ffmpeg_av_log_set_level(-8)
  elif (i==loglevel_normal):   # just necessary messages
    core.normal_log_all()
    core.ffmpeg_av_log_set_level(8)
  elif (i==loglevel_debug):    # verbose
    core.debug_log_all()
    core.ffmpeg_av_log_set_level(8)
  elif (i==loglevel_crazy):    # loglevel crazy here at the python side means more verbosity to ffmpeg
    core.debug_log_all()
    core.ffmpeg_av_log_set_level(32)
    
  else:
    raise(AssertionError("Invalid loglevel"))


"""
Individual loggers can be set like this:

core.setLogLevel_livelogger(loglevel_crazy);

"""
  

def test1():
  st=""" Empty test
  """
  pre=pre_mod+"test1 :"
  print(pre,st)


def test2():
  st=""" Empty test
  """
  pre=pre_mod+"test2 :"
  print(pre,"st")
  

def main():
  pre=pre_mod+"main :"
  print(pre,"main: arguments: ",sys.argv)
  if (len(sys.argv)<2):
    print(pre,"main: needs test number")
  else:
    st="test"+str(sys.argv[1])+"()"
    exec(st)
  
  
if (__name__=="__main__"):
  main()
