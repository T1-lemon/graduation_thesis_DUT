import { Box } from '@mui/material'
import React from 'react'
import { NavLink } from 'react-router-dom'

import '../../assets/css/components/header_bar.css'
import {
  NavListInterface,
  NavListStatus,
  PagesInteface
} from '../../interface/HeaderInterface'

type Props = {
  pages?: PagesInteface[]
  navList?: NavListInterface[]
  onClickNavItem?: (status: NavListStatus) => void
}

function HeaderNav(props: Props) {
  const { pages, navList, onClickNavItem } = props
  return (
    <Box className='header__nav--container'>
      <ul className='header__nav--list'>
        {pages ? (
          pages.map((page) => {
            return (
              <li className='header__nav--item' key={page.pageName}>
                <NavLink
                  to={page.link}
                  className={({ isActive }) =>
                    isActive ? 'header__nav--active' : 'header__nav--link'
                  }
                >
                  {page.pageName}
                </NavLink>
              </li>
            )
          })
        ) : (
          <></>
        )}
        {navList ? (
          navList.map((navItem) => {
            return (
              <li className='header__nav--item' key={navItem.name}>
                <p
                  className={
                    navItem.isActive
                      ? 'header__nav--active'
                      : 'header__nav--link'
                  }
                  onClick={() => {
                    if (onClickNavItem) {
                      onClickNavItem(navItem.status)
                    }
                  }}
                >
                  {navItem.name}
                </p>
              </li>
            )
          })
        ) : (
          <></>
        )}
      </ul>
    </Box>
  )
}

export default HeaderNav
