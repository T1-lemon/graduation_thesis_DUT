import React, { useEffect, useRef, useState } from 'react'
import { useSelector } from 'react-redux'
import SockJS from 'sockjs-client'
import { Client, over } from 'stompjs'

import { RootState } from '../../../redux/configStore'
import { host } from '../../../utils/contant'

type Props = { children: React.ReactNode }
export type DoctorContextType = {
  stompDoctor?: Client
}

export const doctorContext = React.createContext<DoctorContextType | null>(null)

export default function ContextProviderDoctor({ children }: Props) {
  const stompClient = useRef<Client>()
  const [stompDoctor, setStompDoctor] = useState<Client>()
  const { currentUser } = useSelector((state: RootState) => state.auths)

  const connect = () => {
    let Sock = new SockJS(host)

    stompClient.current = over(Sock)
    // stompClient.current.debug = (str) => {}

    stompClient.current.connect({}, onConnected)
  }

  const onConnected = (frame: any) => {
    setStompDoctor(stompClient.current)
  }

  useEffect(() => {
    if (currentUser && currentUser.id) {
      connect()
    }
  }, [currentUser])

  return (
    <doctorContext.Provider value={{ stompDoctor }}>
      {children}
    </doctorContext.Provider>
  )
}
