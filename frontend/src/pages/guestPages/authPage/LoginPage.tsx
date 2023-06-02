import { Box } from '@mui/material'
import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'react-toastify'
import * as Yup from 'yup'

import { ProgressListener } from '../../../components/Progress'
import { FormLoginValues } from '../../../interface/ValidateInterface'
import { useAppDispatch, useAppSelector } from '../../../redux/hooks'
import { clearMessage } from '../../../redux/slices/authSlice'
import { loginUser } from '../../../redux/thunk/authThunk'
import { customFontLoginTheme } from '../../../themes/authTheme'
import FormikCustomize from '../../../utils/formik/FormikCustomize'
import { EMAIL_REGEX } from '../../../utils/regex'
import {
  CHECK_EMAIL_EMPTY,
  CHECK_EMAIL_MATCH_REGEX,
  CHECK_PASSWORD_EMPTY
} from '../../../utils/validateInform'
import AuthBoxHeader from './AuthBoxHeader'

const initialLoginValues = {
  email: '',
  password: ''
}

export interface LoginInputField {
  id: string
  label: string
  autoFoucus?: boolean
}

const loginInputFields: LoginInputField[] = [
  {
    id: 'email',
    label: 'Your Email',
    autoFoucus: true
  },
  {
    id: 'password',
    label: 'Password'
  }
]

export default function LoginPage() {
  const dispatch = useAppDispatch()
  const { isAuth } = useAppSelector((state) => state.auths)

  useEffect(() => {
    if (isAuth) {
      toast.success('Login successfullly')
    }
  }, [isAuth])
  const handleLoginSubmit = (values: FormLoginValues) => {
    const fetchApiLoign = async (dataLogin: FormLoginValues) => {
      ProgressListener.emit('start')
      await dispatch(loginUser(dataLogin))
      ProgressListener.emit('stop')
    }
    fetchApiLoign(values)
  }

  const handleLoginValidationSchema = () => {
    return Yup.object().shape({
      email: Yup.string()
        .required(CHECK_EMAIL_EMPTY)
        .matches(EMAIL_REGEX, CHECK_EMAIL_MATCH_REGEX),
      password: Yup.string().required(CHECK_PASSWORD_EMPTY)
    })
  }

  return (
    <>
      <div className='container__auth'>
        <AuthBoxHeader
          title='Welcome to Hospital Care'
          detail='To get started, please enter your details'
        />
        <Box className='auth__box--body'>
          <FormikCustomize
            initialFormikValues={initialLoginValues}
            inputFields={loginInputFields}
            onValidationSchema={handleLoginValidationSchema}
            onSubmitFormik={handleLoginSubmit}
            btnText='Sign in'
            theme={customFontLoginTheme}
          />
        </Box>
        <Box className='auth__footer'>
          <p>Don't have an account?</p>
          <Link
            to='/register'
            className='auth__link'
            onClick={() => dispatch(clearMessage())}
          >
            Sign up
          </Link>
        </Box>
      </div>
    </>
  )
}
