import React from 'react'

import AppointmentNotification from './AppointmentNotification '
import BreadCrumbs from './BreadCrumbs'
import ChatNotification from './ChatNotification'

type Props = {}

export default function NavBar({}: Props) {
  return (
    <nav className='container__nav'>
      <div className='container__nav-content'>
        <div className='nav__left'>
          <BreadCrumbs />
        </div>
        <div className='nav__right'>
          <AppointmentNotification />
          <ChatNotification />
        </div>
      </div>
    </nav>
  )
}
