#ifndef enumiter_HEADER_GUARD
#define enumiter_HEADER_GUARD
/*
 * enumiter.h : iterate over enum class
 * 
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

/** 
 *  @file    enumiter.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.5.3 
 *  
 *  @brief   iterate over enum class
 */ 

// ripped from: https://stackoverflow.com/questions/8498300/allow-for-range-based-for-with-enum-classes

#include <iostream>

template< typename T >
class Enum
{
public:
   class Iterator
   {
   public:
      Iterator( int value ) :
         m_value( value )
      { }

      T operator*( void ) const
      {
         return (T)m_value;
      }

      void operator++( void )
      {
         ++m_value;
      }

      bool operator!=( Iterator rhs )
      {
         return m_value != rhs.m_value;
      }

   private:
      int m_value;
   };

};

template< typename T >
typename Enum<T>::Iterator begin( Enum<T> )
{
   return typename Enum<T>::Iterator( (int)T::First );
}

template< typename T >
typename Enum<T>::Iterator end( Enum<T> )
{
   return typename Enum<T>::Iterator( ((int)T::Last) + 1 );
}

/* use like this:
for( auto e: Enum<FrameClass>() ) {
  std::cout << ((int)e) << std::endl;
}
*/
  

#endif
