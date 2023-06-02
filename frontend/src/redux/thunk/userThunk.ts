import { createAsyncThunk } from '@reduxjs/toolkit'

import {
  DataUserProfile,
  DataUserProfilePicture,
  UserInformation
} from '../../interface/UsersInterface'
import {
  updateUserProfilePictureService,
  updateUserProfileService
} from '../../services/userServices'

export const updateUserProfile = createAsyncThunk(
  'user/profile/update',
  async (data: DataUserProfile) => {
    const response: UserInformation = await updateUserProfileService(data)
    return response
  }
)

export const updateUserProfilePicture = createAsyncThunk(
  'user/profilePicture/update',
  async (data: DataUserProfilePicture) => {
    const response: UserInformation = await updateUserProfilePictureService(
      data
    )
    return response
  }
)
