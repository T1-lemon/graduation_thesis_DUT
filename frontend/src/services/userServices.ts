import {
  DataUserProfile,
  DataUserProfilePicture
} from '../interface/UsersInterface'
import requestAuthApi from '../utils/requestApi/requestAuthApi'

export const getCurrentUserService = async (): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'get',
      url: '/user'
    })
    const { data } = response
    return data
  } catch (error: any) {
    const { data } = error.response
    return data
  }
}

export const updateUserProfilePictureService = async (
  dataUpdateProfilePicture: DataUserProfilePicture
): Promise<any> => {
  try {
    const response = await requestAuthApi({
      method: 'post',
      url: `/user/${dataUpdateProfilePicture.userId}/profile-picture`,
      data: {
        profilePicture: dataUpdateProfilePicture.profilePicture
      },
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

export const updateUserProfileService = async (
  dataUserProfile: DataUserProfile
): Promise<any> => {
  try {
    const { userId, ...dataApi } = dataUserProfile
    const response = await requestAuthApi({
      method: 'put',
      url: `/user/${userId}`,
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
