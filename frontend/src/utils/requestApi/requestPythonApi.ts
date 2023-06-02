import axios from 'axios'

const requestPythonApi = axios.create({
  baseURL: `${process.env.REACT_APP_PYTHON_URL_API}`
})

export default requestPythonApi
