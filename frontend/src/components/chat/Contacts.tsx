import React, { useEffect, useState } from 'react'

import '../../assets/css/components/chat/contacts.css'
import { IConversationReceiver } from '../../interface/ConversationInterface'
import { UserInformation } from '../../interface/UsersInterface'
import { Role } from '../../utils/roles'

type Props = {
  conversations: IConversationReceiver[]
  currentUser: UserInformation
  changeChat: (type: IConversationReceiver) => void
}

export default function Contacts(props: Props) {
  const { conversations, changeChat, currentUser } = props
  const [currentSelected, setCurrentSelected] = useState<number>()

  const changeCurrentChat = (
    index: number,
    conversation: IConversationReceiver
  ) => {
    setCurrentSelected(index)
    changeChat(conversation)
  }

  return (
    <>
      {currentUser && (
        <div className='container_conversation'>
          {currentUser.role == Role.Doctor ? <p>Patients</p> : <p>Doctors</p>}

          <div className='contacts'>
            {conversations.map((conversation, index) => {
              const { id, receiver } = conversation
              return (
                <div
                  key={id}
                  className={`contact ${
                    index === currentSelected ? 'selected' : ''
                  }`}
                  onClick={() => changeCurrentChat(index, conversation)}
                >
                  <div className='avatar'>
                    <img src={`${receiver.profilePicture}`} alt='' />
                  </div>
                  <div className='username'>
                    <p>
                      {receiver.firstName} {receiver.lastName}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>

          {/* <div className='current-user'>
            <div className='avatar'>
              <img src={`${currentUser.profilePicture}`} alt='avatar' />
            </div>
            <div className='username'>
              <h2>
                {currentUser.firstName} {currentUser.lastName}
              </h2>
            </div>
          </div> */}
        </div>
      )}
    </>
  )
}
