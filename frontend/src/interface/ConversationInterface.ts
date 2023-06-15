import { UserInformation } from './UsersInterface'

export interface IConversationReceiver {
  id: number
  createdBy: null
  modifiedBy: null
  createdDate: Date
  modifiedDate: Date
  // name: string
  // backgroundImage: string
  receiver: UserInformation
}
