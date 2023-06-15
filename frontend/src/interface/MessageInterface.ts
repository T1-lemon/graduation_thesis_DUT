export interface IMessage {
  text: string
  conversationId: number
}

export interface IMessageResponse {
  fromSelf: boolean
  message: string
}
