import CloseIcon from '@mui/icons-material/Close'
import { Alert, Box } from '@mui/material'
import React, { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { Outlet, useNavigate } from 'react-router-dom'

import '../assets/css/templates/auth_template.css'
import ButtonCustomize from '../components/ButtonCustomize'
import HeaderBoxLogo from '../components/header/HeaderBoxLogo'
import { useAppSelector } from '../redux/hooks'
import { clearMessage } from '../redux/slices/authSlice'
import { checkResponseSuccess } from '../utils/checkResponseStatus'
import { Role } from '../utils/roles'

function AuthTemplate() {
  const navigate = useNavigate()
  const dispatch = useDispatch()

  const { status, message, isAuth, role } = useAppSelector(
    (state) => state.auths
  )

  useEffect(() => {
    if (isAuth && role === Role.Patient) {
      navigate('/home')
    }

    if (isAuth && role === Role.Doctor) {
      navigate('/doctor/appointment')
    }

    if (isAuth && role === Role.Admin) {
      navigate('/admin/appointments')
    }
  }, [isAuth])

  const handleCloseAlert = () => {
    dispatch(clearMessage())
  }

  return (
    <>
      <div className='header'>
        <HeaderBoxLogo />
      </div>
      {message && !checkResponseSuccess(status) ? (
        <Box className='box__alert'>
          <Alert severity='error'>{message}</Alert>
          <ButtonCustomize
            icon={<CloseIcon fontSize='small' />}
            className='btn__alert--close'
            onClickBtn={handleCloseAlert}
          />
        </Box>
      ) : (
        <></>
      )}
      <div className={`${!message ? 'auth__outlet' : 'auth__outlet--alert'}`}>
        <Outlet />
      </div>
    </>
  )
}

export default AuthTemplate
