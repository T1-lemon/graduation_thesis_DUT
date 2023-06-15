import { Grid } from '@mui/material'
import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import '../../../assets/css/pages/doctorPage/appointmentDoctor/appointment_patient_of_doctor.css'
import { IAppointment } from '../../../interface/AppointmentInterface'
import { getAllAppointmentPatientForDoctorService } from '../../../services/appointmentService'
import TableAppointment from './TableAppointment'

type Props = {}

export default function AppointmentPatientOfDoctor({}: Props) {
  const [appointments, setAppointments] = useState<IAppointment[]>([])

  const params = useParams()

  const getAllAppointmentPatientForDoctor = async () => {
    const patientId: string | undefined = params.patientId
    const listAppointment: IAppointment[] =
      await getAllAppointmentPatientForDoctorService(patientId as string)

    setAppointments(listAppointment)
  }

  useEffect(() => {
    getAllAppointmentPatientForDoctor()
  }, [params.id, params.patientId])

  const arrInforPatient = [
    {
      title: 'Gender:',
      value: appointments[0]?.patient.gender ? 'Male' : 'Female'
    },
    {
      title: 'Age:',
      value: appointments[0]?.patient.age
    },
    {
      title: 'Phone:',
      value: appointments[0]?.patient.phoneNumber
    },
    {
      title: 'Address:',
      value: appointments[0]?.patient.address
    }
  ]

  return (
    <div className='container__patient--doctor'>
      <div className='card widget__profile-paient'>
        <div className='card__header'>
          <div className='profile__info-widget'>
            <img src={appointments[0]?.patient.profilePicture} alt='' />
            <h3>
              {appointments[0]?.patient.firstName}{' '}
              {appointments[0]?.patient.lastName}
            </h3>
          </div>
        </div>
        <div className='card__body'>
          <ul>
            {arrInforPatient.map((item, index) => {
              return (
                <li key={index}>
                  <p className='title__information'>{item.title}</p>
                  <span className='value'>{item.value}</span>
                </li>
              )
            })}
          </ul>
        </div>
      </div>

      <div className='table__appointment'>
        <TableAppointment
          appointments={appointments}
          getAllAppointmentPatientForDoctor={getAllAppointmentPatientForDoctor}
        />
      </div>
    </div>
  )
}
