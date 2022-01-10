"""
multi.py : api 2 level encapsulation for ValkkaFS having multiple streams in single file

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

@file    multi.py
@author  Sampsa Riikonen
@date    2021
@version 1.3.0 

@brief   api 2 level encapsulation for ValkkaFS having multiple streams in single file
"""
import time
import json
import os
import glob
import numpy
import logging
from valkka import core
# from valkka.api2.tools import *
from valkka.api2.tools import parameterInitCheck
from valkka.fs.tools import findBlockDevices
from valkka.api2.exceptions import ValkkaFSLoadError
from valkka.fs.base import ValkkaFS
import datetime
import traceback

"""Blocktable format:

::

    k-max   max



"""

def getTimeRange(blocktable = None):
    """Returns tuple timerange.  If BT empty, returns None
    """
    assert(blocktable is not None)
    # print(blocktable)
    nonzero = blocktable[:,0] > 0
    if nonzero.sum() < 1: # empty blocktable
        return None
    return (int(blocktable[nonzero,0].min()), int(blocktable[nonzero,1].max()))
    

def getInd(times, blocktable = None):
    """Times is a tuple of mstimestamps.  Returns blocks between mstimestamps
    
    This implementation of searching the blocks might seem overly complicated ..
    .. but its not: there are many things to take into account, for example, the fact
    that block numbers are wrapped over device boundaries

    returns correct block indices
    """
    # verbose=True
    verbose=False
    assert(blocktable is not None)

    t0 = times[0]
    t1 = times[1]

    if (t0>t1):
        return []
        
    # 1: max among keyframes
    # 2: min among all frames
        
    ftab = blocktable[:,0]                 # FTAB: max values  
    filt = blocktable[:,0]>0               # we're only interested in blocks that are used (i.e. > 0)
    
    # ***** FROM LOWER LIMIT TO INF ****
    
    aux = ftab[(ftab<=t0)*filt]            # All blocks before the requested keyframe (including a block that has the keyframe)
    """
    We dont want timestamps that are >= t0, but the last one in the list of values with t>=0 
    
    t0 = 10.5
    t1 = 15.5
    
    15 16 17 7 8 9 10 11 12 13 14
                . . . .  
                    |
                    take this (max) = tt
    """

    if (verbose): print("BlockTable: aux 1=",aux)
    
    if (aux.shape[0]>0):
        tt=aux[aux==aux.max()][0]          # target time found ok
    else:
        aux=ftab[(ftab>=t0)*filt]          # the next best option.. 
        """
        t0 = 10.5
        
        15 16 17 0 0  11 12 13 14
        .  .  .       .  .  .  .
                        |
                        take this (min) = tt
        """
        if (aux.shape[0]==0):             # there is nothing corresponding to t0 !
            return []
        else:
            if (verbose): print("BlockTable: aux 1b=",aux)
            tt=aux.min()
    
    if (verbose): print("BlockTable: target time 1=",tt)
    
    eka = (ftab>=tt)*filt                  # all times greater or equal to final target time tt
    """
    eka:
    
    15 16 17 7 8 9 10 11 12 13 14
    .  .  .         .  .  .  .  . 
    """
    
    if (verbose): print("BlockTable: eka=",eka)
    
    
    # **** FROM -INF TO UPPER LIMIT ****
    aux=ftab[(ftab>t1)*filt]             # ("target time 2" =tt) is just above t1, i.e. the first timestamp in the blocktable > t1
    """
    We don't want timestamps <= t1, but the smallest in the list of values > t1
    
    
    t0 = 10.5
    t1 = 15.5
    
    15 16 17 7 8 9 10 11 12 13 14
        .  .                     
        | 
        take this == tt
    """
    
    if (aux.shape[0]==0):                # so there's nothing greater than t1
        aux=ftab[(ftab>=t1)*filt]        # .. equal to, maybe?
                
    if (aux.shape[0]==0):                # nothing, nothing
        toka=eka
    else:
        tt=aux[aux==aux.min()][0]        # target time found
        toka=(ftab<=tt)*filt             # all times less or equal to target times (compared to FTAB: max values)
        """
        15 16 17 7 8 9 10 11 12 13 14
        .  .     . . .  .  .  .  .  .
        """
    if (verbose): print("BlockTable: target time 2=",tt)

    if (verbose): print("BlockTable: toka=",toka)
    
    # *** COMBINE (LOWER, INF) and (-INF, UPPER) ***
    inds=numpy.where(eka*toka)[0]        # AND
    
    order=blocktable[inds,0].argsort()   # put the block indices into time order
    if (verbose): print("BlockTable: order:",order)
    if (len(order)>0):
        t0=blocktable[order[0],0]
    else:
        t0=None
        
    final = inds[order]                 # return sorted indices
    return final.tolist()
    

def getIndNeigh(n=1, time=0, blocktable = None):
    verbose=False
    # verbose=True
    assert(blocktable is not None)
    
    ftab=blocktable[:,0]                                # row index 1 = max timestamp of all devices in the block 
    
    inds=numpy.where( ( (ftab<time) & (ftab>0) ) )[0]   # => ftab indices
    if (verbose): print("BlockTable: inds:",inds)
    # print(ftab[inds])
    order=ftab[inds].argsort()                          # ordering array that puts those indices into time order ..
    if (verbose): print("BlockTable: order:",order)
    inds=inds[order]                                    # sorted indices ..
    # if (verbose): print("Sorted blocks:",ftab[inds])
    if (verbose): print("BlockTable: sorted indices",inds)
    inds=numpy.flipud(inds)                             # .. in descending order
    if (verbose): print("BlockTable: indices, des",inds)
    inds=inds[0:min(len(inds),n)]                       # take nearest neighbour to target block.. and the target block (its the first index .. not necessarily!)
    finalinds=inds.copy()
    if (verbose): print("BlockTable: final indices",finalinds,"\n")
    
    inds=numpy.where(ftab>=time)[0]                     # => ftab indices
    if (verbose): print("BlockTable: inds2:",inds)
    order=ftab[inds].argsort()                          # ordering array that puts those indices into time order ..
    if (verbose): print("BlockTable: order2:",order)
    inds=inds[order]                                    # sorted indices ..
    if (verbose): print("BlockTable: sorted indices2:",inds)
    inds=inds[0:min(len(inds),n+1)]                     # take nearest neighbour to target block..
    if (verbose): print("BlockTable: final indices2",inds,"\n")
    finalinds=numpy.hstack((finalinds,inds))
    if (verbose): print("BlockTable: ** final indices",finalinds,"\n")
    
    order=ftab[finalinds].argsort()                          # final blocks should be in time order
    finalinds = finalinds[order]
    # finalinds.sort()
    return finalinds.tolist()
    

class ValkkaMultiFS(ValkkaFS):
    """Multiple streams per one file - not that great idea after all
    """
    core_valkkafs_class = core.ValkkaFS

    def getTimeRange(self):
        return getTimeRange(self.blocktable)

    def getInd(self, times):
        return getInd(times, self.blocktable)

    def getIndNeigh(self, n=1, time=0):
        return getIndNeigh(n=n, time=time, blocktable = self.blocktable)

