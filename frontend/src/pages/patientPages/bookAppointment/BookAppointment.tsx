import React, { useContext, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import '../../../assets/css/pages/patientPage/bookAppointment/book_appointment.css'
import {
  ITimeSlot,
  ITimeSlotResponse
} from '../../../interface/TimeSlotInterfaces'
import { getDetailTimeSlotService } from '../../../services/timeSlotService'
import {
  PatientContextType,
  patientContext
} from '../context/ContextProviderPatient'
import BookingSummary from './BookingSummary'
import FormAppointment from './FormAppointment'

type Props = {}

export default function BookAppointment(props: Props) {
  const params = useParams()
  const { stompPatient } = useContext(patientContext) as PatientContextType

  const [timeSlotResponse, setTimeSlotResponse] = useState<ITimeSlotResponse>()

  const getDetailTimeSlot = async () => {
    const id: string | undefined = params.id
    const timeSlotDetail: ITimeSlotResponse = await getDetailTimeSlotService(
      id as string
    )
    setTimeSlotResponse(timeSlotDetail)
  }

  useEffect(() => {
    getDetailTimeSlot()
  }, [params.id])

  return (
    <div className='book__appointment'>
      <FormAppointment
        stompClient={stompPatient}
        timeSlotResponse={timeSlotResponse}
      />

      <BookingSummary timeSlot={timeSlotResponse?.timeSlotDTO} />
    </div>
  )
}
