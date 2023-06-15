import React, { useEffect, useRef, useState } from 'react'
import { Outlet } from 'react-router-dom'

import '../../../assets/css/pages/doctorPage/homeDoctor/home_doctor.css'
import ContextProviderDoctor from '../context/ContextProviderDoctor'
import NavBar from './NavBar'
import SideBar from './SideBar'

type Props = {}

export default function MainDoctor(props: Props) {
  return (
    <ContextProviderDoctor>
      <div className='container__doctor-page'>
        <SideBar />
        <div className='container__doctor-content'>
          <NavBar />
          <div className='container__outlet'>
            <Outlet />
          </div>
        </div>
      </div>
    </ContextProviderDoctor>
  )
}
