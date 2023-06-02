import {
  AdminMedicalExaminationInterface,
  DataAdminSetDoctorProfile
} from '../../interface/AdminInformationInterface'
import requestAuthApi from '../../utils/requestApi/requestAuthApi'

export const addAdminSetDoctorProfileService = async (
  dataDoctorProfile: any
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'post',
      url: `/user/doctor`,
      data: dataDoctorProfile,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const archiveDoctorByAdminService = async (
  doctorId: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'patch',
      url: `/medical_examinations/archive/${doctorId}`,
      data: {
        statusArchive: 1
      }
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const checkValidDoctorInforService = async (
  email: string
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'post',
      url: '/user/doctor/check-email',
      data: {
        email
      }
    })
    return response.data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const editMedicalExaminationService = async (
  examniationData: AdminMedicalExaminationInterface
): Promise<any> => {
  try {
    const { title, shortDescription, description, examinationPrice } =
      examniationData
    const response = await requestAuthApi({
      method: 'patch',
      url: `/medical_examinations/${examniationData.id}`,
      data: {
        title,
        shortDescription,
        description,
        examinationPrice
      }
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const getAllDoctorByAdminService = async (
  statusArchive: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: `/user/doctor?statusArchive=${statusArchive}`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const getDetailDoctorByAdminService = async (
  doctorId: number
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: `/user/doctor/${doctorId}`
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const updateAdminProfileService = async (
  dataAdminSetDoctorProfile: DataAdminSetDoctorProfile
): Promise<any> => {
  try {
    const { doctorId, ...dataApi } = dataAdminSetDoctorProfile
    const response = await requestAuthApi({
      method: 'put',
      url: `/user/doctor/update-profile/${doctorId}`,
      data: {
        ...dataApi
      }
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}
