import { createSlice } from '@reduxjs/toolkit'

import { UserInformation } from '../../interface/UsersInterface'
import {
  loginUser,
  logoutUser,
  registerUser,
  updateAuth
} from '../thunk/authThunk'

interface UserState {
  isAuth: boolean
  message: String
  status: number
  loading: boolean
  currentUser: UserInformation
  role?: String
  isCheckInitialStatus: boolean
}

const initialState: UserState = {
  isAuth: false,
  message: '',
  status: 0,
  loading: false,
  currentUser: {},
  role: '',
  isCheckInitialStatus: true
}

export const authSlice = createSlice({
  name: 'auths',
  initialState,
  reducers: {
    clearMessage: (state) => {
      state.message = ''
      state.status = 0
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(loginUser.rejected, (state, action: any) => {
        state.message = action.payload.message
        state.status = action.payload.status
      })
      .addCase(loginUser.fulfilled, (state, action: any) => {
        state.currentUser = action.payload
        state.role = action.payload.role
        state.isAuth = true
        state.message = ''
      })
      .addCase(logoutUser.pending, (state) => {
        state.loading = true
      })
      .addCase(logoutUser.fulfilled, (state) => {
        state.isAuth = false
        state.loading = false
        state.currentUser = {}
        state.role = ''
        state.isCheckInitialStatus = false
      })
      .addCase(updateAuth.pending, (state) => {
        state.loading = true
      })
      .addCase(updateAuth.fulfilled, (state, action: any) => {
        state.currentUser = action.payload
        state.role = action.payload.role
        state.isAuth = true
        state.message = ''
      })
      .addCase(registerUser.fulfilled, (state, action: any) => {
        state.message = action.payload.message
        state.isAuth = false
        state.status = action.payload.status
      })
      .addCase(registerUser.rejected, (state, action: any) => {
        state.message = action.payload.message
        state.status = action.payload.status
      })
      .addMatcher(
        (action) =>
          action.type.startsWith('auth/') && action.type.endsWith('/rejected'),
        (state) => {
          state.loading = false
          state.isAuth = false
          state.isCheckInitialStatus = false
        }
      )
      .addMatcher(
        (action) =>
          action.type.startsWith('auth/') && action.type.endsWith('/fulfilled'),
        (state) => {
          state.loading = false
          state.isCheckInitialStatus = false
        }
      )
      .addMatcher(
        (action) =>
          action.type.startsWith('user/') && action.type.endsWith('/fulfilled'),
        (state, action: any) => {
          state.currentUser = action.payload
        }
      )
  }
})

export const { clearMessage } = authSlice.actions
const authReducer = authSlice.reducer
export default authReducer
