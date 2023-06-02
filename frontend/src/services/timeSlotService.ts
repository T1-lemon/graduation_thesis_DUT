import { ITimeSlotRequest } from '../interface/TimeSlotInterfaces'
import requestApi from '../utils/requestApi/requestApi'
import requestAuthApi from '../utils/requestApi/requestAuthApi'

export const getAllTimeSlotOfDoctorService = async () => {
  try {
    const response = await requestAuthApi({
      url: `/time_slots/doctor`,
      method: 'get'
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const getDetailTimeSlotService = async (id: string) => {
  try {
    const response = await requestApi({
      url: `/time_slots/${id}`,
      method: 'get'
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const deleteTimeSlotService = async (timeSlotId: number) => {
  try {
    const response = await requestAuthApi({
      url: `/time_slots/${timeSlotId}`,
      method: 'delete'
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const editTimeSlotService = async (
  timeSlotId: number,
  timeSlotUpdate: ITimeSlotRequest
) => {
  try {
    const response = await requestAuthApi({
      url: `/time_slots/${timeSlotId}`,
      method: 'patch',
      data: timeSlotUpdate
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const addArrTimeSlotService = async (
  arrTimeSlot: ITimeSlotRequest[]
) => {
  try {
    const response = await requestAuthApi({
      url: `/time_slots`,
      method: 'post',
      data: arrTimeSlot
    })
    return response.data
  } catch (error) {
    return error
  }
}
