import { createAsyncThunk } from '@reduxjs/toolkit'

import { AppointmentInforInterface } from '../../interface/AppointmentInterface'
import { getListAppointmentService } from '../../services/appointmentService'

export const getListAppointment = createAsyncThunk(
  'appointments/getList',
  async () => {
    const response: AppointmentInforInterface =
      await getListAppointmentService()
    return response
  }
)
