import { createAsyncThunk } from '@reduxjs/toolkit'

import { AuthErrResponse } from '../../interface/AuthInterface'
import { UserResponse } from '../../interface/UsersInterface'
import {
  FormLoginValues,
  FormRegisterValues,
  RegisterSuccessResponse
} from '../../interface/ValidateInterface'
import {
  loginServices,
  logoutService,
  registerService
} from '../../services/authService'
import { getCurrentUserService } from '../../services/userServices'
import { checkResponseSuccess } from '../../utils/checkResponseStatus'

export const loginUser = createAsyncThunk(
  'auth/login',
  async (dataLogin: FormLoginValues, { rejectWithValue }) => {
    const response: UserResponse & AuthErrResponse = await loginServices(
      dataLogin
    )
    if (checkResponseSuccess(response.status)) {
      return response.user
    }
    return rejectWithValue(response)
  }
)

export const logoutUser = createAsyncThunk(
  'logout',
  async (_, { rejectWithValue }) => {
    const response = await logoutService()
    const { status } = response
    if (!checkResponseSuccess(status)) {
      return rejectWithValue('')
    }
  }
)

export const registerUser = createAsyncThunk(
  'auth/register',
  async (dataRegister: FormRegisterValues, { rejectWithValue }) => {
    const response: RegisterSuccessResponse & AuthErrResponse =
      await registerService(dataRegister)
    if (checkResponseSuccess(response.status)) {
      return response
    }
    return rejectWithValue(response)
  }
)

export const updateAuth = createAsyncThunk(
  'auth/update',
  async (_, { rejectWithValue }) => {
    const response: UserResponse = await getCurrentUserService()

    if (checkResponseSuccess(response.status)) {
      return response.user
    }
    return rejectWithValue('')
  }
)
