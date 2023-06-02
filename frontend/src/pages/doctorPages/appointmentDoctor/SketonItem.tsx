import { Skeleton } from '@mui/material'
import React from 'react'

type Props = {}

export default function SketonItem({}: Props) {
  return (
    <div className='appointment__item'>
      <div className='profile__infor'>
        <div className='avatar__patient'>
          <Skeleton
            animation='wave'
            variant='rounded'
            width='100%'
            height='100%'
          />
        </div>
        <div className='infor__patient'>
          <Skeleton
            animation='wave'
            variant='text'
            sx={{ fontSize: '2.5rem' }}
            width='124px'
          />

          <ul className='infor__list'>
            <Skeleton
              animation='wave'
              variant='text'
              sx={{ fontSize: '1rem' }}
              width='200px'
            />
            <Skeleton
              animation='wave'
              variant='text'
              sx={{ fontSize: '1rem' }}
              width='200px'
            />
            <Skeleton
              animation='wave'
              variant='text'
              sx={{ fontSize: '1rem' }}
              width='200px'
            />
            <Skeleton
              animation='wave'
              variant='text'
              sx={{ fontSize: '1rem' }}
              width='200px'
            />
          </ul>
        </div>
      </div>

      <div className='appointment__action'>
        <Skeleton
          animation='wave'
          variant='rounded'
          sx={{ width: '115px', height: '42px', borderRadius: '85px' }}
        />
      </div>
    </div>
  )
}
