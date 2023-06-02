import { useEffect, useState } from 'react'

import '../../../assets/css/pages/adminPage/adminAppointment/admin_appointment_detail.css'
import {
  AdminMedicalExaminationInterface,
  AdminUserInterface
} from '../../../interface/AdminInformationInterface'
import { getDetailAppointmentService } from '../../../services/adminServices/adminAppointmentServices'
import { convertGenderToString } from '../../../utils/convertGenderToString'
import { addHoursToDate, formatDate, getTimeZone } from '../../../utils/date'
import AdminAppoitnmentDetailBox from './AdminAppoitnmentDetailBox'

type Props = {
  apptId: number
}

export interface DataAdminDetailAppointment {
  title: string
  data: string
}

interface ResponseDetailAppointmentInterface {
  status: number
  patient: AdminUserInterface
  doctor: AdminUserInterface
  timeSlot: {
    startTime: string
    duration: number
    medicalExamination: AdminMedicalExaminationInterface
  }
}

function AdminAppointmentDetail(props: Props) {
  const { apptId } = props
  const [dataApi, setDataApi] = useState<
    Partial<ResponseDetailAppointmentInterface>
  >({})
  const [patientInfor, setPatientInfor] = useState<
    DataAdminDetailAppointment[]
  >([])
  const [doctorInfor, setDoctorInfor] = useState<DataAdminDetailAppointment[]>(
    []
  )
  const [appointmentInfor, setAppointmentInfor] = useState<
    DataAdminDetailAppointment[]
  >([])

  useEffect(() => {
    const fetchGetDetailAppointment = async () => {
      const response = await getDetailAppointmentService(apptId as number)
      const { patient, timeSlot, doctor, status } = response
      setDataApi({ patient, timeSlot, doctor, status })
    }

    fetchGetDetailAppointment()
  }, [apptId])

  useEffect(() => {
    console.log(dataApi)
    if (dataApi) {
      const { patient, timeSlot, doctor, status } = dataApi

      const patientInforApi: DataAdminDetailAppointment[] = [
        { title: 'gmail', data: `${patient?.email}` },
        {
          title: 'gender',
          data: convertGenderToString(patient?.gender as boolean)
        },
        { title: 'age', data: `${patient?.age}` },
        { title: 'phone number', data: `${patient?.phoneNumber}` },
        { title: 'address', data: `${patient?.address}` }
      ]

      const doctorInforApi: DataAdminDetailAppointment[] = [
        { title: 'gmail', data: `${doctor?.email}` },
        {
          title: 'gender',
          data: convertGenderToString(patient?.gender as boolean)
        },
        { title: 'age', data: `${doctor?.age}` },
        { title: 'phone number', data: `${doctor?.phoneNumber}` },
        { title: 'address', data: `${doctor?.address}` }
      ]

      if (timeSlot?.startTime) {
        const date = new Date(timeSlot.startTime as string)
        const appointmentInforApi: DataAdminDetailAppointment[] = [
          { title: 'appointment date', data: `${formatDate(date)}` },
          {
            title: 'time',
            data: `${getTimeZone(
              timeSlot?.startTime as string
            )} - ${getTimeZone(
              addHoursToDate(date, timeSlot?.duration as number)
            )}`
          },
          {
            title: 'price',
            data: `${timeSlot?.medicalExamination.examinationPrice}$`
          },
          { title: 'status', data: `${status}` }
        ]
        setAppointmentInfor(appointmentInforApi)
      }

      setPatientInfor(patientInforApi)
      setDoctorInfor(doctorInforApi)
    }
  }, [dataApi])

  return (
    <div className='admin__apptdetail--container'>
      <AdminAppoitnmentDetailBox
        dataInfor={patientInfor}
        title='Appointment From: '
        name={`${dataApi.patient?.firstName} ${dataApi.patient?.lastName}`}
        profilePicture={`${dataApi.patient?.profilePicture}`}
        status={true}
      />
      <AdminAppoitnmentDetailBox
        dataInfor={doctorInfor}
        title='Appointment To: '
        name={`Dr. ${dataApi.doctor?.firstName} ${dataApi.doctor?.lastName}`}
        profilePicture={`${dataApi.doctor?.profilePicture}`}
        status={true}
      />
      <AdminAppoitnmentDetailBox
        dataInfor={appointmentInfor}
        title='Appointment Information: '
        name='doctor name'
        status={false}
      />
    </div>
  )
}

export default AdminAppointmentDetail
