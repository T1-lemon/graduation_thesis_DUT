import requestApi from '../utils/requestApi/requestApi'

export const getAllCategoryService = async () => {
  try {
    const response = await requestApi({
      url: '/categories',
      method: 'get'
    })
    return response.data
  } catch (error) {
    return error
  }
}
