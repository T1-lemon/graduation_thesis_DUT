import { Navigate, useLocation } from 'react-router-dom'

import { useAppSelector } from '../redux/hooks'
import Page404 from './Page404'
import { Role } from './roles'

const AuthRoute = ({
  children,
  roles
}: {
  children: JSX.Element
  roles: Role
}) => {
  const location = useLocation()
  const { isAuth, isCheckInitialStatus, role } = useAppSelector(
    (state) => state.auths
  )

  if (!isAuth && !isCheckInitialStatus) {
    return <Navigate to='/login' state={{ from: location }} />
  }

  const userHasRequiredRole = roles.includes(`${role}`)

  if (isAuth && !userHasRequiredRole && !isCheckInitialStatus) {
    return <Page404 />
  }

  return children
}

export default AuthRoute
