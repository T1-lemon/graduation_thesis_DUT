import { RouteObject } from 'react-router-dom'

import BookAppointment from '../pages/patientPages/bookAppointment/BookAppointment'
import PatientBookedAppoitment from '../pages/patientPages/dashboardPage/PatientBookedAppoitment'
import PatientProfile from '../pages/patientPages/dashboardPage/PatientProfile'
import HomeTemplates from '../templates/HomeTemplates'
import PatientDashboardTemplate from '../templates/PatientDashboardTemplate'
import AuthRoute from '../utils/AuthRoute'
import { Role } from '../utils/roles'

const patientRoutes: RouteObject[] = [
  {
    path: '/user',
    element: (
      <AuthRoute roles={Role.Patient}>
        <PatientDashboardTemplate />
      </AuthRoute>
    ),
    children: [
      {
        path: 'profile-settings',
        element: <PatientProfile />
      },
      {
        path: 'appointments',
        element: <PatientBookedAppoitment />
      }
    ]
  },
  {
    path: '',
    element: <HomeTemplates />,
    children: [
      {
        path: 'book-appointment/:id',
        element: (
          <AuthRoute roles={Role.Patient}>
            <BookAppointment />
          </AuthRoute>
        )
      }
    ]
  }
]

export default patientRoutes
