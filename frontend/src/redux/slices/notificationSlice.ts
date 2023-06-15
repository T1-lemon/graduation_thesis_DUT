import { PayloadAction, createSlice } from '@reduxjs/toolkit'

import {
  INotificationDoctorApi,
  INotificationPatientApi
} from '../../interface/NotificationInterface'
import {
  getNotificationsDoctorService,
  getNotificationsPatientService,
  readNotificationService
} from '../../services/notificationService'
import { DispatchType } from '../configStore'

export type NotificationState = {
  listNotificationDoctor: INotificationDoctorApi[]
  listNotificationPatient: INotificationPatientApi[]
}

const initialState: NotificationState = {
  listNotificationDoctor: [],
  listNotificationPatient: []
}

const notificationSlice = createSlice({
  name: 'notificationReducer',
  initialState,
  reducers: {
    getNotificationsDoctorAction: (
      state: NotificationState,
      action: PayloadAction<INotificationDoctorApi[]>
    ) => {
      state.listNotificationDoctor = action.payload
    },
    getNotificationsPatientAction: (
      state: NotificationState,
      action: PayloadAction<INotificationPatientApi[]>
    ) => {
      state.listNotificationPatient = action.payload
    }
  }
})

export const { getNotificationsDoctorAction, getNotificationsPatientAction } =
  notificationSlice.actions

export default notificationSlice.reducer

export const getNotificationsDoctorThunk = () => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await getNotificationsDoctorService()

      dispatch(getNotificationsDoctorAction(data))
    } catch (error) {
      return error
    }
  }
}

export const getNotificationsPatientThunk = () => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await getNotificationsPatientService()

      dispatch(getNotificationsPatientAction(data))
    } catch (error) {
      return error
    }
  }
}

export const readNotificationsDoctorThunk = (appointmentId: number) => {
  return async (dispatch: DispatchType) => {
    try {
      await readNotificationService(appointmentId)

      dispatch(getNotificationsDoctorThunk())
    } catch (error) {
      return error
    }
  }
}

export const readNotificationsPatientThunk = (appointmentId: number) => {
  return async (dispatch: DispatchType) => {
    try {
      await readNotificationService(appointmentId)

      dispatch(getNotificationsPatientThunk())
    } catch (error) {
      return error
    }
  }
}
