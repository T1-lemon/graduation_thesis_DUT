import { PayloadAction, createSlice } from '@reduxjs/toolkit'

import {
  ITimeSlotRequest,
  ITimeSlotResponse
} from '../../interface/TimeSlotInterfaces'
import {
  addArrTimeSlotService,
  deleteTimeSlotService,
  editTimeSlotService,
  getAllTimeSlotOfDoctorService
} from '../../services/timeSlotService'
import { DispatchType } from '../configStore'

export type TimeSlotState = {
  arrTimeSlotOfDoctorInCurrentWeek: ITimeSlotResponse[]
}

const initialState: TimeSlotState = {
  arrTimeSlotOfDoctorInCurrentWeek: []
}

const timeSlotSlice = createSlice({
  name: 'timeSlotsReducer',
  initialState,
  reducers: {
    getAllTimeSlotOfDoctorAction: (
      state: TimeSlotState,
      action: PayloadAction<ITimeSlotResponse[]>
    ) => {
      state.arrTimeSlotOfDoctorInCurrentWeek = action.payload
    }
  }
})

export const { getAllTimeSlotOfDoctorAction } = timeSlotSlice.actions

export default timeSlotSlice.reducer

export const getAllTimeSlotOfDoctorThunk = () => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await getAllTimeSlotOfDoctorService()

      dispatch(getAllTimeSlotOfDoctorAction(data))
    } catch (error) {
      return error
    }
  }
}

export const addArrTimeSlotThunk = (arrTimeSlot: ITimeSlotRequest[]) => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await addArrTimeSlotService(arrTimeSlot)
      dispatch(getAllTimeSlotOfDoctorThunk())
    } catch (error) {
      return error
    }
  }
}

export const deleteTimeSlotThunk = (timeSlotId: number) => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await deleteTimeSlotService(timeSlotId)
      dispatch(getAllTimeSlotOfDoctorThunk())
    } catch (error) {
      return error
    }
  }
}

export const editTimeSlotThunk = (
  timeSlotId: number,
  timeSlotUpdate: ITimeSlotRequest
) => {
  return async (dispatch: DispatchType) => {
    try {
      await editTimeSlotService(timeSlotId, timeSlotUpdate)
      dispatch(getAllTimeSlotOfDoctorThunk())
    } catch (error) {
      return error
    }
  }
}
