import requestAuthApi from '../utils/requestApi/requestAuthApi';


export const getNotificationsDoctorService = async () => {
  try {
    const response = await requestAuthApi({
      url: `/notifications/doctor`,
      method: 'get'
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const getNotificationsPatientService = async () => {
  try {
    const response = await requestAuthApi({
      url: `/notifications/patient`,
      method: 'get'
    })
    return response.data
  } catch (error) {
    return error
  }
}

export const readNotificationService = async (appointmentId: number) => {
  try {
    const response = await requestAuthApi({
      url: `/notifications/read-notification`,
      method: 'patch',
      data: { appointmentId }
    })
    return response.data
  } catch (error) {
    return error
  }
}