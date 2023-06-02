import { useEffect } from 'react'
import { useLocation, useNavigate, useRoutes } from 'react-router-dom'

import { useAppSelector } from '../redux/hooks'
import Page404 from '../utils/Page404'
import { Role } from '../utils/roles'
import adminRoutes from './AdminRoutes'
import doctorRoutes from './DoctorRoutes'
import globalRoutes from './GlobalRoutes'
import patientRoutes from './PatientRoutes'

function ApplicationRoute() {
  const navigate = useNavigate()
  const { role, isCheckInitialStatus } = useAppSelector((state) => state.auths)
  const location = useLocation()
  useEffect(() => {
    if (!isCheckInitialStatus && location.pathname === '/') {
      if (role === Role.Doctor) {
        navigate('/doctor/appointment')
        return
      }

      if (!isCheckInitialStatus) {
        navigate('/home')
      }
    }
  }, [])

  const routes = [
    ...adminRoutes,
    ...doctorRoutes,
    ...patientRoutes,
    ...globalRoutes,
    {
      path: '*',
      element: <Page404 />
    }
  ]

  return useRoutes(routes)
}

export default ApplicationRoute
