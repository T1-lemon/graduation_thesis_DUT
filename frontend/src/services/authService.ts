import {
  FormLoginValues,
  FormRegisterValues
} from '../interface/ValidateInterface'
import requestApi from '../utils/requestApi/requestApi'

export const loginServices = async (
  dataLogin: FormLoginValues
): Promise<any> => {
  try {
    const response = await requestApi({
      method: 'post',
      url: '/auth/authenticate',
      data: {
        ...dataLogin
      }
    })
    return response.data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const logoutService = async (): Promise<any> => {
  try {
    const response = await requestApi({
      method: 'post',
      url: `/auth/logout`
    })
    return response
  } catch (error: any) {
    return error
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

export const registerService = async (
  dataRegister: FormRegisterValues
): Promise<any> => {
  try {
    const response = await requestApi({
      method: 'post',
      url: `/auth/register`,
      data: {
        ...dataRegister
      }
    })
    return response.data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}
