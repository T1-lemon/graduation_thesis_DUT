import { createSlice } from '@reduxjs/toolkit'

import {
  AdminDepartmentInterface,
  AdminDoctorInterface,
  AdminUserInterface
} from '../../interface/AdminInformationInterface'
import { TableAppointment } from '../../interface/AdminTableInterface'
import {
  createDepartmentByAdmin,
  deleteDepartmentByAdmin,
  editDepartmentByAdmin
} from '../thunk/adminThunk/adminDepartmentThunk'
import {
  editMedicalExamination,
  getAllDepartmentByAdmin,
  getDetailDoctorByAdmin,
  updateDoctorProfile,
  updateDoctorProfilePicture
} from '../thunk/adminThunk/adminDoctorThunk'
import { getAllPatientByAdmin } from '../thunk/adminThunk/adminPatientThunk'

interface UserState {
  listAppointment?: TableAppointment[]
  listDepartment?: AdminDepartmentInterface[]
  doctorDetail: Partial<AdminDoctorInterface>
  listPatient?: AdminUserInterface[]
  messageDeleteDepartment: string
}

const initialState: UserState = {
  listAppointment: [],
  listDepartment: [],
  doctorDetail: {},
  listPatient: [],
  messageDeleteDepartment: ''
}

export const adminSlice = createSlice({
  name: 'admin',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder.addCase(getAllPatientByAdmin.fulfilled, (state, action) => {
      state.listPatient = action.payload
    })
    builder.addCase(getDetailDoctorByAdmin.fulfilled, (state, action: any) => {
      state.doctorDetail = action.payload
    })
    builder.addCase(editMedicalExamination.fulfilled, (state, action: any) => {
      state = { ...state, doctorDetail: action.payload }
    })
    builder.addCase(
      updateDoctorProfilePicture.fulfilled,
      (state, action: any) => {
        state.doctorDetail = {
          ...state.doctorDetail,
          doctorInfor: action.payload
        }
      }
    )
    builder.addCase(updateDoctorProfile.fulfilled, (state, action: any) => {
      state.doctorDetail = {
        ...state.doctorDetail,
        doctorInfor: action.payload.doctorInfor,
        department: action.payload.department
      }
    })
    builder.addCase(getAllDepartmentByAdmin.fulfilled, (state, action: any) => {
      state.listDepartment = action.payload
    })
    builder.addCase(createDepartmentByAdmin.fulfilled, (state, action: any) => {
      if (state.listDepartment) {
        state.listDepartment = [...state.listDepartment, action.payload]
      }
    })
    builder.addCase(editDepartmentByAdmin.fulfilled, (state, action: any) => {
      if (state.listDepartment) {
        const newListDepartment = state.listDepartment.map((department) => {
          if (department.id === action.payload.id) {
            return action.payload
          }
          return department
        })

        state.listDepartment = newListDepartment
      }
    })
    builder.addCase(deleteDepartmentByAdmin.pending, (state) => {
      state.messageDeleteDepartment = 'Delete Successfully!'
    })
    builder.addCase(deleteDepartmentByAdmin.fulfilled, (state, action: any) => {
      if (state.listDepartment) {
        state.listDepartment = action.payload
        state.messageDeleteDepartment = 'Delete Successfully!'
      }
    })
    builder.addCase(deleteDepartmentByAdmin.rejected, (state, action: any) => {
      if (state.listDepartment) {
        state.messageDeleteDepartment = "Can't delete this department"
      }
    })
  }
})

// export const { addProfilePictureDoctor } = adminSlice.actions
const adminReducer = adminSlice.reducer
export default adminReducer
