import { createAsyncThunk } from '@reduxjs/toolkit'

import { getAllPatientByAdminService } from '../../../services/adminServices/adminPatientService'

export const getAllPatientByAdmin = createAsyncThunk(
  'admin/patient/getAll',
  async () => {
    const response = await getAllPatientByAdminService()
    const listPatient = response.map((patient: any) => {
      const {
        createdBy,
        createdDate,
        modifiedBy,
        modifiedDate,
        role,
        ...patientInfor
      } = patient
      return patientInfor
    })

    return listPatient
  }
)
