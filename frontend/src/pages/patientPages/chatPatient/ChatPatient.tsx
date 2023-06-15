import React, { useContext, useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'

import '../../../assets/css/pages/patientPage/chat/chat_patient.css'
import ChatContainer from '../../../components/chat/ChatContainer'
import Contacts from '../../../components/chat/Contacts'
import Welcome from '../../../components/chat/Welcome'
import { IConversationReceiver } from '../../../interface/ConversationInterface'
import { useAppSelector } from '../../../redux/hooks'
import { getAllConversationUserService } from '../../../services/conversationService'
import {
  PatientContextType,
  patientContext
} from '../context/ContextProviderPatient'

type Props = {}

export default function ChatPatient({}: Props) {
  const [conversations, setConversations] = useState<IConversationReceiver[]>(
    []
  )

  const { stompPatient } = useContext(patientContext) as PatientContextType

  const [currentChat, setCurrentChat] = useState<IConversationReceiver>()

  const { currentUser } = useAppSelector((state) => state.auths)

  const { state } = useLocation()

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

  //check conversation first
  useEffect(() => {
    if (state && conversations.length > 0) {
      const currentConversation = conversations.find(
        (item) => item.receiver.id === state.doctorId
      )
      setCurrentChat(currentConversation)
    }
  }, [conversations])

  const handleChatChange = (chat: IConversationReceiver) => {
    setCurrentChat(chat)
  }
  return (
    <>
      <div className='container_chat-patient'>
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
              stompClient={stompPatient}
            />
          )}
        </div>
      </div>
    </>
  )
}
