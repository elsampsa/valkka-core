/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// Copyright (c) 1996-2020, Live Networks, Inc.  All rights reserved
// "Group Endpoint Id"
// Implementation

#include "GroupEId.hh"


GroupEId::GroupEId(struct in_addr const& groupAddr,
		   portNumBits portNum, u_int8_t ttl) {
  struct in_addr sourceFilterAddr;
  sourceFilterAddr.s_addr = ~0; // indicates no source filter

  init(groupAddr, sourceFilterAddr, portNum, ttl);
}

GroupEId::GroupEId(struct in_addr const& groupAddr,
		   struct in_addr const& sourceFilterAddr,
		   portNumBits portNum) {
  init(groupAddr, sourceFilterAddr, portNum, 255);
}

struct in_addr const& GroupEId::groupAddress() const {
  struct sockaddr_in const& groupAddress4 = (struct sockaddr_in const&)fGroupAddress;
      // later fix to allow for IPv6
  return groupAddress4.sin_addr;
}

struct in_addr const& GroupEId::sourceFilterAddress() const {
  struct sockaddr_in const& sourceFilterAddress4
    = (struct sockaddr_in const&)fSourceFilterAddress;
      // later fix to allow for IPv6
  return sourceFilterAddress4.sin_addr;
}

Boolean GroupEId::isSSM() const {
  struct sockaddr_in const& sourceFilterAddress4
    = (struct sockaddr_in const&)fSourceFilterAddress;
      // later fix to allow for IPv6
  return sourceFilterAddress4.sin_addr.s_addr != netAddressBits(~0);
}

portNumBits GroupEId::portNum() const {
  struct sockaddr_in const& groupAddress4 = (struct sockaddr_in const&)fGroupAddress;
      // later fix to allow for IPv6
  return groupAddress4.sin_port;
}

void GroupEId::init(struct in_addr const& groupAddr,
		    struct in_addr const& sourceFilterAddr,
		    portNumBits portNum,
		    u_int8_t ttl) {
  fGroupAddress.ss_family = AF_INET; // later update to support IPv6
  ((sockaddr_in&)fGroupAddress).sin_addr = groupAddr;
  ((sockaddr_in&)fGroupAddress).sin_port = portNum;
  
  fSourceFilterAddress.ss_family = AF_INET; // later update to support IPv6
  ((sockaddr_in&)fSourceFilterAddress).sin_addr = sourceFilterAddr;

  fTTL = ttl;
}
