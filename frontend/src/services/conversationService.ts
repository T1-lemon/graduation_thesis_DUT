import requestAuthApi from '../utils/requestApi/requestAuthApi'

export const getAllConversationUserService = async (senderId: number) => {
  try {
    const respone = await requestAuthApi({
      method: 'get',
      url: `conversations/receivers?senderId=${senderId}`
    })
    return respone.data
  } catch (error) {
    console.log(error)
    // return error;
  }
}

export const addConversationService = async (receiverId: number) => {
  try {
    const respone = await requestAuthApi({
      method: 'post',
      url: `conversations?receiverId=${receiverId}`
    })
    return respone.data
  } catch (error) {
    console.log(error)
    // return error;
  }
}
