import React from 'react'
import { Outlet } from 'react-router-dom'

import HeaderBar from '../components/header/HeaderBar'
import ContextProviderPatient from '../pages/patientPages/context/ContextProviderPatient'

export default function HomeTemplates() {
  return (
    <ContextProviderPatient>
      <div>
        <HeaderBar />
        <Outlet />
        <footer></footer>
      </div>
    </ContextProviderPatient>
  )
}
