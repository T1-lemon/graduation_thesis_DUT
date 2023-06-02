import axios from 'axios'

import { AuthErrResponse } from '../../interface/AuthInterface'
import { refeshTokenService } from '../../services/authService'

const requestAuthApi = axios.create({
  withCredentials: true,
  baseURL: `${process.env.REACT_APP_AUTH_URL_API}`
})

const handleAccessTokenExpired = async (error: any) => {
  try {
    await refeshTokenService()
    return axios(error.config)
  } catch (error: any) {
    throw error
  }
}

// Add a response interceptor
requestAuthApi.interceptors.response.use(
  function (response) {
    return response
  },
  function (error) {
    const { response } = error
    const { data } = response
    const responseErr: AuthErrResponse = data
    if (responseErr.status === 401 || responseErr.status === 403) {
      return handleAccessTokenExpired(error)
    }
    return Promise.reject(error)
  }
)

export default requestAuthApi
