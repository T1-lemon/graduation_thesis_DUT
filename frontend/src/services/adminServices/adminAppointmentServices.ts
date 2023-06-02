import requestAuthApi from '../../utils/requestApi/requestAuthApi'

export const getAllAppointmentService = async (
  pageIndex: number,
  limit: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: `/appointments/pageable?pageIndex=${pageIndex}&limit=${limit}`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const getDetailAppointmentService = async (
  apptId: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: `/appointments/${apptId}`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}
