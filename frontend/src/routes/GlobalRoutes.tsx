import { RouteObject } from 'react-router'

import LoginPage from '../pages/guestPages/authPage/LoginPage'
import RegisterPage from '../pages/guestPages/authPage/RegisterPage'
import RegisterVerifyEmail from '../pages/guestPages/authPage/RegisterVerifyEmail'
import RegisterVerifyStatus from '../pages/guestPages/authPage/RegisterVerifyStatus'
import DetailExamination from '../pages/guestPages/detailExamination/DetailExamination'
import MainFindDoctor from '../pages/guestPages/findDoctorPage/MainFindDoctor'
import HomePage from '../pages/guestPages/homePage/HomePage'
import AuthTemplate from '../templates/AuthTemplate'
import HomeTemplates from '../templates/HomeTemplates'

const globalRoutes: RouteObject[] = [
  {
    path: 'home',
    element: <HomeTemplates />,
    children: [
      {
        path: '',
        element: <HomePage />
      }
    ]
  },
  {
    path: '',
    element: <AuthTemplate />,
    children: [
      {
        path: '/login',
        element: <LoginPage />
      },
      {
        path: '/register',
        element: <RegisterPage />
      },
      {
        path: '/process_register',
        element: <RegisterVerifyEmail />
      },
      {
        path: '/verify/:token',
        element: <RegisterVerifyStatus />
      }
    ]
  },
  {
    path: '',
    element: <AuthTemplate />,
    children: [
      {
        path: '/login',
        element: <LoginPage />
      },
      {
        path: '/register',
        element: <RegisterPage />
      }
    ]
  },
  {
    path: '',
    element: <HomeTemplates />,
    children: [
      {
        path: 'search-doctor',
        element: <MainFindDoctor />
      },
      {
        path: 'examination/detail/:id',
        element: <DetailExamination />
      }
    ]
  }
]

export default globalRoutes
