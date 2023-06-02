import ChevronRightIcon from '@mui/icons-material/ChevronRight'
import HomeOutlinedIcon from '@mui/icons-material/HomeOutlined'
import NotificationsNoneIcon from '@mui/icons-material/NotificationsNone'
import IconButton from '@mui/material/IconButton'
import React from 'react'
import Avatar from 'react-avatar'
import { useSelector } from 'react-redux'
import { NavLink } from 'react-router-dom'

import { RootState } from '../../../redux/configStore'
import BreadCrumbs from './BreadCrumbs'

type Props = {}

export default function NavBar({}: Props) {
  const { currentUser } = useSelector((state: RootState) => state.auths)

  return (
    <nav className='container__nav'>
      <div className='container__nav-content'>
        <div className='nav__left'>
          <BreadCrumbs />
        </div>
        <div className='nav__right'>
          {/* <div className='avatar'>
            {currentUser.profilePicture ? (
              <img src={currentUser.profilePicture} alt='' />
            ) : (
              <Avatar facebookId='100008343750912' size='120' />
            )}
          </div> */}
        </div>
      </div>
    </nav>
  )
}
