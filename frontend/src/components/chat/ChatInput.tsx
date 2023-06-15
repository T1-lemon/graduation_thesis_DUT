import Picker, { EmojiClickData } from 'emoji-picker-react'
import React, { useState } from 'react'
import { BsEmojiSmile } from 'react-icons/bs'
import { IoMdSend } from 'react-icons/io'

import '../../assets/css/components/chat/chat_input.css'

type Props = {
  handleSendMsg: (type: string) => void
}

export default function ChatInput({ handleSendMsg }: Props) {
  const [msg, setMsg] = useState('')
  const [showEmojiPicker, setShowEmojiPicker] = useState(false)
  const handleEmojiPickerhideShow = () => {
    setShowEmojiPicker(!showEmojiPicker)
  }

  const handleEmojiClick = (emojiObject: EmojiClickData) => {
    let message = msg
    message += emojiObject.emoji
    setMsg(message)
  }

  const sendChat = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (msg.length > 0) {
      handleSendMsg(msg)
      setMsg('')
    }
  }

  return (
    <div className='chat_input-contrainer'>
      <div className='button-container'>
        <div className='emoji'>
          <BsEmojiSmile onClick={handleEmojiPickerhideShow} />
          {showEmojiPicker && <Picker onEmojiClick={handleEmojiClick} />}
        </div>
      </div>
      <form className='input-container' onSubmit={(event) => sendChat(event)}>
        <input
          type='text'
          placeholder='Type your message ...'
          onChange={(e) => setMsg(e.target.value)}
          value={msg}
        />
        <button type='submit'>
          <IoMdSend className='icon_send' />
        </button>
      </form>
    </div>
  )
}
