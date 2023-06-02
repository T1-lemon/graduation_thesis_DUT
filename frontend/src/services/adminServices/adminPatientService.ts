import requestAuthApi from '../../utils/requestApi/requestAuthApi'

export const getAllPatientByAdminService = async (): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: '/user/patient'
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const getDetailPatientByAdminService = async (
  patientId: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: `/user/patient/${patientId}`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}
