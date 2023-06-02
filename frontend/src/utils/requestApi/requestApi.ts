import axios from 'axios'

const requestApi = axios.create({
  withCredentials: true,
  baseURL: `${process.env.REACT_APP_BASE_URL_API}`
})

export default requestApi
