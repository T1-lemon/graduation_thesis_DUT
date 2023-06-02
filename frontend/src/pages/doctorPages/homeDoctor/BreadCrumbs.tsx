import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import { Link } from '@mui/material'
import Breadcrumbs from '@mui/material/Breadcrumbs'
import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'

import '../../../assets/css/components/breadcrumb.css'

type Props = {}

export default function BreadCrumbs({}: Props) {
  const breadcrumbs = [{ content: 'Home', url: '/doctor/appointment' }]

  const location = useLocation()

  let currentLink = ''
  let check = location.pathname.includes('appointment')

  if (check) {
    breadcrumbs.push({ content: 'Appointment', url: '/doctor/appointment' })
  } else {
    location.pathname
      .split('/')
      .filter((crumb) => crumb !== '')
      .forEach((crumb, index) => {
        currentLink += `/${crumb}`

        if (index !== 0) {
          let content = crumb.split('-').join(' ')
          breadcrumbs.push({ content: content, url: currentLink })
        }
      })
  }

  const renderCrumb = () => {
    return breadcrumbs.map((crumb) => {
      return (
        <div className='crumb' key={crumb.content}>
          <NavLink to={crumb.url}>{crumb.content}</NavLink>
        </div>
      )
    })
  }

  return <div className='breadcumbs'>{renderCrumb()}</div>
}
