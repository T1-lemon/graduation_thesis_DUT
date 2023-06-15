import React, { useEffect, useRef, useState } from 'react'
import { Client } from 'stompjs'

import '../../assets/css/components/chat/chat_container.css'
import { IConversationReceiver } from '../../interface/ConversationInterface'
import { IMessageResponse } from '../../interface/MessageInterface'
import { UserInformation } from '../../interface/UsersInterface'
import { getAllMessageTwoUserService } from '../../services/messageService'
import { addMessageService } from '../../services/messageService'
import ChatInput from './ChatInput'

type Props = {
  currentChat: IConversationReceiver
  currentUser: UserInformation
  stompClient?: Client | null
}

export default function ChatContainer(props: Props) {
  const { currentChat, stompClient, currentUser } = props
  const { receiver, ...conversation } = currentChat

  // lưu tin nhắn gửi
  const [messages, setMessages] = useState<IMessageResponse[]>([])
  // dom đến div ở dưới
  const scrollRef = useRef<HTMLDivElement>(null)
  // lưu tin nhắn nhận
  const [arrivalMessage, setArrivalMessage] = useState<IMessageResponse>()

  useEffect(() => {
    const recieveMessageApi = async () => {
      if (currentUser && currentUser.id) {
        const listMessage: IMessageResponse[] =
          await getAllMessageTwoUserService(
            currentUser.id as number,
            receiver.id as number
          )

        setMessages(listMessage)
      }
    }

    recieveMessageApi()
  }, [currentChat])

  const handleSendMsg = async (msg: string) => {
    if (stompClient) {
      let chatMessage = {
        senderName: currentUser.email,
        receiverName: receiver.email,
        message: msg,
        status: 'MESSAGE'
      }
      stompClient.send('/app/private-message', {}, JSON.stringify(chatMessage))
    }
    // gọi api gửi messenge
    await addMessageService({ conversationId: conversation.id, text: msg })

    const msgs = [...messages]
    msgs.push({ fromSelf: true, message: msg })
    setMessages(msgs)
  }

  useEffect(() => {
    receiveMess()
  }, [])

  const receiveMess = () => {
    if (stompClient) {
      stompClient.subscribe(`/user/${currentUser.email}/private`, (payload) => {
        let payloadData = JSON.parse(payload.body)
        const { message } = payloadData
        setArrivalMessage({ fromSelf: false, message: message })
      })
    }
  }

  useEffect(() => {
    arrivalMessage && setMessages((prev) => [...prev, arrivalMessage])
  }, [arrivalMessage])

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className='chat_content_container'>
      <div className='chat-header'>
        <div className='user-details'>
          <div className='avatar'>
            <img src={`${receiver.profilePicture}`} alt='' />
          </div>
          <div className='username'>
            <h4>
              {receiver.firstName} {receiver.lastName}
            </h4>
          </div>
        </div>
      </div>

      <div className='chat-messages'>
        {messages.map((message, index) => {
          return (
            <div ref={scrollRef} key={message.message + index}>
              <div
                className={`message ${
                  message.fromSelf ? 'sended' : 'recieved'
                }`}
              >
                {!message.fromSelf ? (
                  <div className='avatar_receiver'>
                    <img src={`${receiver.profilePicture}`} alt='' />
                  </div>
                ) : (
                  ''
                )}

                <div className='content '>
                  <p>{message.message}</p>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <ChatInput handleSendMsg={handleSendMsg} />
    </div>
  )
}
