import React, { useEffect, useRef, useState } from 'react'
import { useSelector } from 'react-redux'
import SockJS from 'sockjs-client'
import { Client, over } from 'stompjs'

import { RootState } from '../../../redux/configStore'
import { host } from '../../../utils/contant'

type Props = { children: React.ReactNode }
export type PatientContextType = {
  stompPatient?: Client
}

export const patientContext = React.createContext<PatientContextType | null>(
  null
)

export default function ContextProviderPatient({ children }: Props) {
  const stompClient = useRef<Client>()
  const [stompPatient, setStompPatient] = useState<Client>()
  const { currentUser } = useSelector((state: RootState) => state.auths)

  const connect = () => {
    let Sock = new SockJS(host)

    stompClient.current = over(Sock)
    // stompClient.current.debug = (str) => {}

    stompClient.current.connect({}, onConnected)
  }

  const onConnected = (frame: any) => {
    // console.log(frame)
    setStompPatient(stompClient.current)
  }

  useEffect(() => {
    if (currentUser && currentUser.id) {
      connect()
    }
  }, [currentUser])

  return (
    <patientContext.Provider value={{ stompPatient }}>
      {children}
    </patientContext.Provider>
  )
}
