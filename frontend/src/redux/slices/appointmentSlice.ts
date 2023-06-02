import { PayloadAction, createSlice } from '@reduxjs/toolkit'

import {
  AppointmentInforInterface,
  IAppointmentPageable
} from '../../interface/AppointmentInterface'
import {
  changeStatusAppointmentService,
  getAllAppointmentDoctorPageableService
} from '../../services/appointmentService'
import { DispatchType } from '../configStore'
import { getListAppointment } from '../thunk/appointmentThunk'

interface AppointmentState {
  listAppointments: AppointmentInforInterface[]
  loadingAppointment: boolean
  appointmentPageable: IAppointmentPageable | null
}
const initialState: AppointmentState = {
  listAppointments: [],
  loadingAppointment: false,
  appointmentPageable: null
}

export const appointmentSlice = createSlice({
  name: 'appointments',
  initialState,
  reducers: {
    getAllAppointmentDoctorPageableAction: (
      state: AppointmentState,
      action: PayloadAction<IAppointmentPageable>
    ) => {
      state.appointmentPageable = action.payload
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(getListAppointment.pending, (state) => {
        state.loadingAppointment = true
      })
      .addCase(getListAppointment.fulfilled, (state, action: any) => {
        state.loadingAppointment = false
        state.listAppointments = action.payload
      })
  }
})

export const { getAllAppointmentDoctorPageableAction } =
  appointmentSlice.actions

const appointmentReducer = appointmentSlice.reducer
export default appointmentReducer

export const changeStatusAppointmentThunk = (
  appointmentId: number,
  appointmentStatus: number
) => {
  return async (dispatch: DispatchType) => {
    try {
      await changeStatusAppointmentService(appointmentId, appointmentStatus)

      dispatch(getAllAppointmentDoctorPageableThunk(1, 4, 0))
    } catch (error) {
      return error
    }
  }
}

export const getAllAppointmentDoctorPageableThunk = (
  pageIndex: number,
  limit: number,
  appointmentStatus: number
) => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await getAllAppointmentDoctorPageableService(
        pageIndex,
        limit,
        appointmentStatus
      )

      dispatch(getAllAppointmentDoctorPageableAction(data))
    } catch (error) {
      return error
    }
  }
}
