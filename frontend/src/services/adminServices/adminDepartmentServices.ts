import requestAuthApi from '../../utils/requestApi/requestAuthApi'

export const createDepartmentByAdminService = async (
  dataDepartment: any
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'post',
      url: `/department`,
      data: dataDepartment,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const deleteDepartmentByAdminService = async (
  departmentId: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'delete',
      url: `/department/${departmentId}`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const editDepartmentByAdminService = async (
  dataDepartment: any,
  departmentId: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'post',
      url: `/department/${departmentId}`,
      data: dataDepartment,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const getAllDepartmentByAdminService = async (): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: `/department`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}
