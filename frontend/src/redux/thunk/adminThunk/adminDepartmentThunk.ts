import { createAsyncThunk } from '@reduxjs/toolkit'

import {
  createDepartmentByAdminService,
  deleteDepartmentByAdminService,
  editDepartmentByAdminService
} from '../../../services/adminServices/adminDepartmentServices'
import { checkResponseFailed } from '../../../utils/checkResponseStatus'

export const createDepartmentByAdmin = createAsyncThunk(
  'admin/department/create',
  async (dataDepartment: any) => {
    const response = await createDepartmentByAdminService(dataDepartment)
    return response
  }
)

export const deleteDepartmentByAdmin = createAsyncThunk(
  'admin/department/delete',
  async (departmentId: number, { rejectWithValue }) => {
    const response = await deleteDepartmentByAdminService(departmentId)

    if (checkResponseFailed(response.status)) {
      return rejectWithValue('')
    }
    return response
  }
)

export const editDepartmentByAdmin = createAsyncThunk(
  'admin/department/edit',
  async (params: [any, number]) => {
    const response = await editDepartmentByAdminService(params[0], params[1])
    return response
  }
)
