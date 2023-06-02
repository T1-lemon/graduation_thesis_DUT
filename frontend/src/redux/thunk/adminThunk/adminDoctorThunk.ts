import { createAsyncThunk } from '@reduxjs/toolkit'

import {
  AdminMedicalExaminationInterface,
  DataAdminSetDoctorProfile
} from '../../../interface/AdminInformationInterface'
import { DataUserProfilePicture } from '../../../interface/UsersInterface'
import { getAllDepartmentByAdminService } from '../../../services/adminServices/adminDepartmentServices'
import {
  editMedicalExaminationService,
  getDetailDoctorByAdminService,
  updateAdminProfileService
} from '../../../services/adminServices/adminDoctorService'
import { updateUserProfilePictureService } from '../../../services/userServices'

export const editMedicalExamination = createAsyncThunk(
  'admin/medical/edit',
  async (examinationData: AdminMedicalExaminationInterface) => {
    const response = await editMedicalExaminationService(examinationData)
    return response
  }
)

export const getAllDepartmentByAdmin = createAsyncThunk(
  'admin/department/getAll',
  async () => {
    const response = await getAllDepartmentByAdminService()
    return response
  }
)

export const getDetailDoctorByAdmin = createAsyncThunk(
  'admin/doctor/getDetail',
  async (dotorId: number) => {
    const response = await getDetailDoctorByAdminService(dotorId)
    const { department } = response.medicalExamination
    const doctor = {
      doctorInfor: response.doctor,
      medicalExamination: response.medicalExamination,
      department: department
    }
    return doctor
  }
)

export const updateDoctorProfile = createAsyncThunk(
  'admin/doctor/profile/update',
  async (dataUserProfile: DataAdminSetDoctorProfile) => {
    const response = await updateAdminProfileService(dataUserProfile)
    const {doctor, medicalExamination} = response
    const {department} = medicalExamination
    const dataResponse = {
      doctorInfor: doctor, 
      departmentId: department.id
    }
    console.log(dataResponse)
    return dataResponse
  }
)

export const updateDoctorProfilePicture = createAsyncThunk(
  'admin/doctor/profilePicture/update',
  async (data: DataUserProfilePicture) => {
    const response = await updateUserProfilePictureService(data)
    return response
  }
)
