import { Box } from '@mui/material'
import React from 'react'
import { Link } from 'react-router-dom'
import * as Yup from 'yup'

import { ProgressListener } from '../../../components/Progress'
import { FormRegisterValues } from '../../../interface/ValidateInterface'
import { useAppDispatch, useAppSelector } from '../../../redux/hooks'
import { clearMessage } from '../../../redux/slices/authSlice'
import { registerUser } from '../../../redux/thunk/authThunk'
import { customFontLoginTheme } from '../../../themes/authTheme'
import FormikCustomize from '../../../utils/formik/FormikCustomize'
import { EMAIL_REGEX, NAME_REGEX, PASSWORD_REGEX } from '../../../utils/regex'
import { Role } from '../../../utils/roles'
import {
  CHECK_EMAIL_EMPTY,
  CHECK_EMAIL_MATCH_REGEX,
  CHECK_FIRST_NAME_EMPTY,
  CHECK_LAST_NAME_EMPTY,
  CHECK_NAME_EMPTY,
  CHECK_NAME_MATCH_REGEX,
  CHECK_PASSWORD_CONFIRM_EMPTY,
  CHECK_PASSWORD_CONFIRM_MATCH_PASSWORD,
  CHECK_PASSWORD_EMPTY,
  CHECK_PASSWORD_MATCH_REGEX
} from '../../../utils/validateInform'
import AuthBoxHeader from './AuthBoxHeader'

const initialRegisterValues = {
  email: '',
  password: '',
  firstName: '',
  lastName: '',
  confirmPassword: '',
  role: ''
}

export interface RegisterInputField {
  id: string
  label: string
  autoFoucus?: boolean
}

const registerInputFields: RegisterInputField[] = [
  {
    id: 'firstName',
    label: 'First Name',
    autoFoucus: true
  },
  {
    id: 'lastName',
    label: 'Last Name'
  },
  {
    id: 'email',
    label: 'Email'
  },
  {
    id: 'password',
    label: 'Password'
  },
  {
    id: 'confirmPassword',
    label: 'ConfirmPassword'
  }
]

const RegisterPage: React.FC = () => {
  const dispatch = useAppDispatch()
  const { role } = useAppSelector((state) => state.auths)

  const handleRegisterSubmit = (values: FormRegisterValues) => {
    const fetchApiRegister = async (dataRegister: FormRegisterValues) => {
      ProgressListener.emit('start')
      await dispatch(registerUser(dataRegister))
      ProgressListener.emit('stop')
    }
    const { confirmPassword, ...dataRegister } = values
    role === Role.Admin
      ? (dataRegister.role = Role.Doctor)
      : (dataRegister.role = Role.Patient)

    fetchApiRegister(dataRegister)
  }

  const handleRegisterValidationSchema = () => {
    return Yup.object().shape({
      email: Yup.string()
        .required(CHECK_EMAIL_EMPTY)
        .matches(EMAIL_REGEX, CHECK_EMAIL_MATCH_REGEX),
      password: Yup.string()
        .trim()
        .required(CHECK_PASSWORD_EMPTY)
        .matches(PASSWORD_REGEX, CHECK_PASSWORD_MATCH_REGEX),
      confirmPassword: Yup.string()
        .trim()
        .required(CHECK_PASSWORD_CONFIRM_EMPTY)
        .oneOf([Yup.ref('password'), ''], CHECK_PASSWORD_CONFIRM_MATCH_PASSWORD)
    })
  }

  const handleRegisterValidation = (values: FormRegisterValues) => {
    const errors = {
      firstName: '',
      lastName: ''
    }

    errors.firstName = !values.firstName
      ? !values.lastName
        ? CHECK_NAME_EMPTY
        : CHECK_FIRST_NAME_EMPTY
      : !NAME_REGEX.test(values.firstName)
      ? !values.lastName
        ? ''
        : CHECK_NAME_MATCH_REGEX
      : !values.lastName
      ? ''
      : !NAME_REGEX.test(values.lastName)
      ? CHECK_NAME_MATCH_REGEX
      : ''

    errors.lastName = !values.lastName
      ? !values.firstName
        ? ' '
        : CHECK_LAST_NAME_EMPTY
      : !NAME_REGEX.test(values.lastName)
      ? !values.firstName
        ? ''
        : ' '
      : !values.firstName
      ? ''
      : !NAME_REGEX.test(values.lastName)
      ? ' '
      : ''

    const responseErrors = !errors.firstName && !errors.lastName ? {} : errors
    return responseErrors
  }

  return (
    <div className='container__auth'>
      <AuthBoxHeader
        title='Get Started Now'
        detail='Enter your credentials to access your account'
      />
      <Box className='auth__box--body'>
        <FormikCustomize
          initialFormikValues={initialRegisterValues}
          inputFields={registerInputFields}
          onValidationSchema={handleRegisterValidationSchema}
          onValidate={handleRegisterValidation}
          onSubmitFormik={handleRegisterSubmit}
          btnText='Sign up'
          theme={customFontLoginTheme}
        />
      </Box>
      <Box className='auth__footer'>
        <p>Already have an account?</p>
        <Link
          to='/login'
          className='auth__link'
          onClick={() => dispatch(clearMessage())}
        >
          Sign in
        </Link>
      </Box>
    </div>
  )
}

export default RegisterPage
