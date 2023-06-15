import { UserInformation } from './UsersInterface'

export interface IFeedback {
  id: number
  createdBy: string
  modifiedBy: string
  createdDate: Date
  modidiedDate: Date
  commentText: string
  patient: UserInformation
}

export interface ICheckConditionFeedback {
  statusCode: number
  message: string
  booked: boolean
}
