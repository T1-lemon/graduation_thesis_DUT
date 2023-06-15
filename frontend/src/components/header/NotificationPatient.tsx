import NotificationsIcon from '@mui/icons-material/Notifications'
import { Badge, Popover } from '@mui/material'
import React, { useContext, useEffect, useRef, useState } from 'react'
import Moment from 'react-moment'
import { useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'

import {
  INotificationPatientApi,
  INotificationPatientSocket
} from '../../interface/NotificationInterface'
import {
  PatientContextType,
  patientContext
} from '../../pages/patientPages/context/ContextProviderPatient'
import { RootState } from '../../redux/configStore'
import { useAppDispatch, useAppSelector } from '../../redux/hooks'
import {
  getNotificationsPatientThunk,
  readNotificationsPatientThunk
} from '../../redux/slices/notificationSlice'
import { getListAppointment } from '../../redux/thunk/appointmentThunk'
import { addHoursToDate, formatDate, getTimeZone } from '../../utils/date'

type Props = {}

export default function NotificationPatient({}: Props) {
  const navigate = useNavigate()
  const dispatch = useAppDispatch()
  const [anchorEl, setAnchorEl] = React.useState<HTMLButtonElement | null>(null)

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleClose = () => {
    setAnchorEl(null)
  }

  const open = Boolean(anchorEl)
  const id = open ? 'simple-popover' : undefined

  const { isAuth, currentUser } = useAppSelector((state) => state.auths)

  const notificationsPatientApi = useSelector(
    (state: RootState) => state.notificationReducer.listNotificationPatient
  )

  const [listNotificationPatient, setListNotificationPatient] = useState<
    INotificationPatientSocket[]
  >([])

  const { stompPatient } = useContext(patientContext) as PatientContextType

  useEffect(() => {
    const getNotificationPatientApi = async () => {
      await dispatch(getNotificationsPatientThunk())
    }
    if (currentUser && currentUser.id) getNotificationPatientApi()
  }, [currentUser])

  useEffect(() => {
    if (notificationsPatientApi.length > 0) {
      const listNotification = notificationsPatientApi.map(
        (item: INotificationPatientApi) => {
          const { isRead, inforNotification } = item
          return {
            patientId: inforNotification.patientId,
            avatarDoctor: inforNotification.profilePictureDoctor,
            doctorName:
              inforNotification.firstNameDoctor +
              ' ' +
              inforNotification.lastNameDoctor,
            startTime: inforNotification.startTime,
            duration: inforNotification.duration,
            status: inforNotification.status,
            appointmentId: inforNotification.appointmentId,
            isRead,
            modifiedDate: inforNotification.modifiedDate
          }
        }
      )

      setListNotificationPatient(listNotification)
    }
  }, [notificationsPatientApi])

  useEffect(() => {
    receivePatientNotifications()
  }, [stompPatient])

  const receivePatientNotifications = () => {
    if (stompPatient) {
      stompPatient.subscribe(
        `/user/${currentUser.id}/status-appointment`,
        (payload) => {
          let payloadData = JSON.parse(payload.body)
          setListNotificationPatient((prev) => [payloadData, ...prev])
        }
      )
    }
  }

  useEffect(() => {
    if (stompPatient) getAppointmentOfPatientApi()
  }, [listNotificationPatient])

  const getAppointmentOfPatientApi = async () => {
    await dispatch(getListAppointment())
  }

  const readNotification = async (appointmentId: number) => {
    await dispatch(readNotificationsPatientThunk(appointmentId))
    navigate('/user/appointments')
  }

  const renderPatientNotification = () => {
    return listNotificationPatient.map((notification, index) => {
      return (
        <div
          className='patient_notification'
          key={index}
          onClick={() => {
            readNotification(notification.appointmentId)
          }}
        >
          <div className='avatar_doctor'>
            <img src={notification.avatarDoctor} alt='' />
          </div>
          <div
            className={`info_appointment ${
              notification.isRead ? 'isRead' : ''
            }`}
          >
            <p>
              <strong>{notification.doctorName}</strong> has{' '}
              {notification.status === 1 ? 'approved' : 'canceled'} your
              appointment
            </p>
            <div className='time_appointment'>
              <strong>{formatDate(new Date(notification.startTime))}</strong>
              <strong>
                {getTimeZone(notification.startTime)} -
                {getTimeZone(
                  addHoursToDate(
                    new Date(notification.startTime),
                    notification.duration
                  )
                )}
              </strong>
            </div>
            <div className='modified_date'>
              <Moment fromNow>{notification.modifiedDate}</Moment>
            </div>
          </div>
          {notification.isRead ? '' : <span className='dot_read'></span>}
        </div>
      )
    })
  }

  return (
    <>
      {isAuth ? (
        <div className='notification_patient-container'>
          <Badge
            badgeContent={
              listNotificationPatient.filter((item) => !item.isRead).length
            }
            color='primary'
            className='badge_notification'
            onClick={handleClick}
          >
            <NotificationsIcon className='icon_notification' />
          </Badge>

          <Popover
            id={id}
            open={open}
            anchorEl={anchorEl}
            onClose={handleClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'left'
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'center'
            }}
          >
            <div className='appointment_notification-container'>
              {listNotificationPatient.length === 0 ? (
                <p>No announcements yet</p>
              ) : (
                renderPatientNotification()
              )}
            </div>
          </Popover>
        </div>
      ) : (
        ''
      )}
    </>
  )
}
