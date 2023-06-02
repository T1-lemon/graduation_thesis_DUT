import { PayloadAction, createAsyncThunk, createSlice } from '@reduxjs/toolkit'

import { IMedicalExaminationTime } from '../../interface/MedicalExaminationInterfaces'
import {
  filterMedicalExaminationTimeByCategoryAndPriceService,
  filterMedicalExaminationTimeByCategoryService,
  getAllMedicalExaminationTimeService,
  getDetailMedicalExaminationTimeService,
  getListPythonDepartmentIdApi
} from '../../services/medicalExaminationService'
import { DispatchType } from '../configStore'

export type MedicalExaminationState = {
  arrMedicalExaminations: IMedicalExaminationTime[]
  arrDeparmentId: number[]
  medicalExaminationDetail: IMedicalExaminationTime | null
}

const initialState: MedicalExaminationState = {
  arrMedicalExaminations: [],
  arrDeparmentId: [],
  medicalExaminationDetail: null
}

const medicalExaminationReducer = createSlice({
  name: 'medicalExaminationReducer',
  initialState,
  reducers: {
    getAllMedicalExaminationTimeAction: (
      state: MedicalExaminationState,
      action: PayloadAction<IMedicalExaminationTime[]>
    ) => {
      state.arrMedicalExaminations = action.payload
    },
    getDetailMedicalExaminationTimeAction: (
      state: MedicalExaminationState,
      action: PayloadAction<IMedicalExaminationTime>
    ) => {
      state.medicalExaminationDetail = action.payload
    },
    clearListDepartmentId: (state) => {
      state.arrDeparmentId = []
    }
  },
  extraReducers: (buider) => {
    buider.addCase(
      getListPythonDepartmentId.fulfilled,
      (state, action: any) => {
        state.arrDeparmentId = action.payload
      }
    )
  }
})

export const {
  getAllMedicalExaminationTimeAction,
  getDetailMedicalExaminationTimeAction,
  clearListDepartmentId
} = medicalExaminationReducer.actions

export default medicalExaminationReducer.reducer

export const filterMedicalExaminationTimeThunkByCategory = (
  categories: string[]
) => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await filterMedicalExaminationTimeByCategoryService(
        categories
      )

      dispatch(getAllMedicalExaminationTimeAction(data))
    } catch (error) {
      return error
    }
  }
}

export const filterMedicalExaminationTimeThunkByCategoryAndPrice = (
  categories: string[],
  minPrice: number,
  maxPrice: number
) => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await filterMedicalExaminationTimeByCategoryAndPriceService(
        categories,
        minPrice,
        maxPrice
      )

      dispatch(getAllMedicalExaminationTimeAction(data))
    } catch (error) {
      return error
    }
  }
}

export const getAllMedicalExaminationTimeThunk = () => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await getAllMedicalExaminationTimeService()

      dispatch(getAllMedicalExaminationTimeAction(data))
    } catch (error) {
      return error
    }
  }
}

export const getDetailMedicalExaminationTimeThunk = (id: string) => {
  return async (dispatch: DispatchType) => {
    try {
      const data = await getDetailMedicalExaminationTimeService(id)

      dispatch(getDetailMedicalExaminationTimeAction(data))
    } catch (error) {
      return error
    }
  }
}

export const getListPythonDepartmentId = createAsyncThunk(
  'python/departmentId',
  async (searchText: string) => {
    const listDepartmentID: string[] = await getListPythonDepartmentIdApi(
      searchText
    )
    return listDepartmentID
  }
)
