import React, { useContext, useEffect, useRef, useState } from 'react'
import SockJS from 'sockjs-client'
import { Client, over } from 'stompjs'

import '../../../assets/css/pages/doctorPage/chatDoctor/chat_doctor.css'
import ChatContainer from '../../../components/chat/ChatContainer'
import Contacts from '../../../components/chat/Contacts'
import Welcome from '../../../components/chat/Welcome'
import { IConversationReceiver } from '../../../interface/ConversationInterface'
import { useAppSelector } from '../../../redux/hooks'
import { getAllConversationUserService } from '../../../services/conversationService'
import { host } from '../../../utils/contant'
import {
  DoctorContextType,
  doctorContext
} from '../context/ContextProviderDoctor'

type Props = {}

export default function ChatDoctor({}: Props) {
  // const stompClient = useRef<Client>()
  const [conversations, setConversations] = useState<IConversationReceiver[]>(
    []
  )

  const [currentChat, setCurrentChat] = useState<IConversationReceiver>()

  const { currentUser } = useAppSelector((state) => state.auths)
  // const { stompDoctor } = useAppSelector((state) => state.socketReducer)
  const { stompDoctor } = useContext(doctorContext) as DoctorContextType

  // const connect = () => {
  //   let Sock = new SockJS(host)

  //   stompClient.current = over(Sock)
  //   // stompClient.current.debug = (str) => {}

  //   stompClient.current.connect({}, onConnected)
  // }

  // const onConnected = (frame: any) => {
  //   console.log(frame)
  // }

  // useEffect(() => {
  //   if (currentUser && currentUser.id) {
  //     connect()
  //   }
  // }, [currentUser])

  // get all conversation
  useEffect(() => {
    const getAllConversationUserApi = async () => {
      if (currentUser) {
        const listConversation: IConversationReceiver[] =
          await getAllConversationUserService(currentUser.id as number)
        setConversations(listConversation)
      }
    }

    getAllConversationUserApi()
  }, [currentUser])

  const handleChatChange = (chat: IConversationReceiver) => {
    setCurrentChat(chat)
  }
  return (
    <>
      <div className='container_chat-doctor'>
        <div className='container'>
          {/* hiển thị các user */}
          <Contacts
            conversations={conversations}
            changeChat={handleChatChange}
            currentUser={currentUser}
          />
          {currentChat === undefined ? (
            <Welcome />
          ) : (
            <ChatContainer
              currentChat={currentChat}
              currentUser={currentUser}
              stompClient={stompDoctor}
            />
          )}
        </div>
      </div>
    </>
  )
}
