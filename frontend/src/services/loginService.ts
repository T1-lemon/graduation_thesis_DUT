import { FormLoginValues } from '../interface/ValidateInterface'
import requestApi from '../utils/requestApi/requestApi'

export const loginServices = async (
  dataLogin: FormLoginValues
): Promise<any> => {
  try {
    const response = await requestApi({
      method: 'post',
      url: '/auth/authenticate',
      data: {
        email: `${dataLogin.email}`,
        password: `${dataLogin.password}`
      }
    })
    return response.data
  } catch (error: any) {
    const responseErr = error.response
    return responseErr.data
  }
}
export const refeshTokenService = async (): Promise<any> => {
  try {
    const response = await requestApi({
      method: 'post',
      url: `/auth/refresh-token`
    })
    return response.data
  } catch (error: any) {
    return error
  }
}
