import { CircularProgress } from '@mui/material'
import React, { useEffect, useState } from 'react'

import '../../../assets/css/pages/adminPage/adminPatient/admin_patient_detail.css'
import { AdminUserInterface } from '../../../interface/AdminInformationInterface'
import { getDetailPatientByAdminService } from '../../../services/adminServices/adminPatientService'
import { DataAdminDetailAppointment } from '../adminAppointment/AdminAppointmentDetail'
import AdminAppointmnetDetailBoxAvatar from '../adminAppointment/AdminAppointmnetDetailBoxAvatar'
import AdminAppoitnmentDetailBox from '../adminAppointment/AdminAppoitnmentDetailBox'

type Props = {
  patientId: number
}

interface PatientDetailInterface {
  patientInfor?: DataAdminDetailAppointment[]
  patientName: string
  patientProfilePicture: string
}
function AdminPatientDetail(props: Props) {
  const { patientId } = props
  const [patientDetail, setPatientDetail] = useState<PatientDetailInterface>({
    patientInfor: [],
    patientName: '',
    patientProfilePicture: ''
  })
  const [loading, setLoading] = useState<boolean>(false)

  useEffect(() => {
    const fetchApiGetAllPatientByAdmin = async () => {
      setLoading(true)
      const response: AdminUserInterface = await getDetailPatientByAdminService(
        patientId
      )
      setLoading(false)

      const patientInfor: DataAdminDetailAppointment[] = [
        { title: 'email', data: `${response.email}` },
        { title: 'gender', data: `${response.gender}` },
        { title: 'age', data: `${response.age}` },
        { title: 'phone number', data: `${response.phoneNumber}` },
        { title: 'address', data: `${response.address}` }
      ]
      const patientName = `${response.firstName} ${response.lastName}`
      const patientProfilePicture = response.profilePicture
      setPatientDetail({
        patientInfor,
        patientName,
        patientProfilePicture
      })
    }
    fetchApiGetAllPatientByAdmin()
    return
  }, [])

  return (
    <div className='admin__patient--detail'>
      {loading ? (
        <div className='modal__loading'>
          <CircularProgress color='inherit' />
        </div>
      ) : (
        <>
          {' '}
          <div className='patient__detail--header'>
            <AdminAppointmnetDetailBoxAvatar
              name={patientDetail.patientName}
              profilePicture={patientDetail.patientProfilePicture}
            />
          </div>
          <div className='patient__detail--body'>
            <AdminAppoitnmentDetailBox
              dataInfor={
                patientDetail.patientInfor ? patientDetail.patientInfor : []
              }
              title='About: '
              name='doctor name'
              status={false}
            />
          </div>
        </>
      )}
    </div>
  )
}

export default AdminPatientDetail
