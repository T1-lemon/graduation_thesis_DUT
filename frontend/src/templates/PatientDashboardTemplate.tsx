import React from 'react'
import { Outlet } from 'react-router-dom'

import '../assets/css/templates/patient_dashboard_template.css'
import PatientSidebar from '../components/PatientSidebar'
import HeaderBar from '../components/header/HeaderBar'

function PatientDashboardTemplate() {
  return (
    <div>
      <HeaderBar />
      <div className='patient__dashboard--container'>
        <PatientSidebar />
        <Outlet />
      </div>
    </div>
  )
}

export default PatientDashboardTemplate
