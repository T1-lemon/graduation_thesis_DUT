import Grid from '@mui/material/Grid'
import React from 'react'

import { RenderStatus } from '../../patientPages/dashboardPage/TableBookedAppointment'
import { DataAdminDetailAppointment } from './AdminAppointmentDetail'
import AdminAppointmnetDetailBoxAvatar from './AdminAppointmnetDetailBoxAvatar'

type Props = {
  dataInfor: DataAdminDetailAppointment[]
  name: string
  title: string
  status: boolean
  profilePicture?: string
}

const AdminAppoitnmentDetailBox = (props: Props) => {
  const { dataInfor, name, title, status, profilePicture } = props
  return (
    <Grid
      container
      className={`apptdetail__grid--container ${
        !status ? 'apptdetail__grid--last' : ''
      }`}
    >
      <Grid item xs={12} className='admin__apptdetail--header'>
        <p>{title}</p>
      </Grid>
      {status ? (
        <Grid item xs={3}>
          <AdminAppointmnetDetailBoxAvatar
            name={name}
            profilePicture={profilePicture as string}
          />
        </Grid>
      ) : (
        <></>
      )}

      <Grid item xs={status ? 9 : 12} className='admin__apptdetail--bodyright'>
        {dataInfor.map((item, index) => {
          return (
            <div className='bodyright__div' key={index}>
              {item.title === 'status' ? (
                <div className='bodyright__div--status'>
                  <p className='bodyright__p'>{item.title}</p>
                  <span className='bodyright__span--status'>:</span>{' '}
                  <RenderStatus status={parseInt(item.data)} />
                </div>
              ) : (
                <>
                  <p className='bodyright__p'>{item.title}</p>
                  <span className='bodyright__span'>{`: ${item.data}`}</span>
                </>
              )}
            </div>
          )
        })}
      </Grid>
    </Grid>
  )
}

export default AdminAppoitnmentDetailBox
