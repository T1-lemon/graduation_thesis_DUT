import React from 'react'
import { Outlet } from 'react-router-dom'

import HeaderBar from '../components/header/HeaderBar'

export default function HomeTemplates() {
  return (
    <div>
      <HeaderBar />
      <Outlet />
      <footer></footer>
    </div>
  )
}
