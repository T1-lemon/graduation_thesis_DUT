import { IMessage } from '../interface/MessageInterface'
import requestAuthApi from '../utils/requestApi/requestAuthApi'

export const addMessageService = async (values: IMessage) => {
  try {
    const respone = await requestAuthApi({
      method: 'post',
      url: `messages`,
      data: { ...values }
    })
    return respone.data
  } catch (error) {
    console.log(error)
    // return error.response;
  }
}
export const getAllMessageTwoUserService = async (
  senderId: number,
  receiverId: number
) => {
  try {
    const respone = await requestAuthApi({
      method: 'get',
      url: `messages?senderId=${senderId}&receiverId=${receiverId}`
    })
    return respone.data
  } catch (error) {
    console.log(error)
    // return error.response;
  }
}
