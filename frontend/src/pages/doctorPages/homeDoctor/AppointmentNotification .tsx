import NotificationsIcon from '@mui/icons-material/Notifications'
import Badge from '@mui/material/Badge'
import Popover from '@mui/material/Popover'
import React, { useContext, useEffect, useRef, useState } from 'react'
import Moment from 'react-moment'
import { useSelector } from 'react-redux'
import { NavLink, useNavigate } from 'react-router-dom'

import '../../../assets/css/pages/doctorPage/homeDoctor/appointment_notification.css'
import { INotificationDoctorSocket } from '../../../interface/NotificationInterface'
import { RootState } from '../../../redux/configStore'
import { useAppDispatch, useAppSelector } from '../../../redux/hooks'
import { getNotificationsDoctorThunk } from '../../../redux/slices/notificationSlice'
import { addHoursToDate, formatDate, getTimeZone } from '../../../utils/date'
import {
  DoctorContextType,
  doctorContext
} from '../context/ContextProviderDoctor'

type Props = {}

export default function AppointmentNotification({}: Props) {
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

  const { currentUser } = useSelector((state: RootState) => state.auths)

  const { listNotificationDoctor } = useSelector(
    (state: RootState) => state.notificationReducer
  )

  const { stompDoctor } = useContext(doctorContext) as DoctorContextType

  const [notifications, setNotifications] = useState<
    INotificationDoctorSocket[]
  >([])

  useEffect(() => {
    const getNotificationApi = async () => {
      await dispatch(getNotificationsDoctorThunk())
    }
    getNotificationApi()
  }, [currentUser])

  useEffect(() => {
    if (listNotificationDoctor.length > 0) {
      const listNotification = listNotificationDoctor.map((notification) => {
        const { isRead, appointment } = notification
        const { patient, timeSlot } = appointment

        return {
          patientId: patient.id,
          doctorId: currentUser.id,
          avatarPatient: patient.profilePicture,
          startTime: timeSlot.startTime,
          duration: timeSlot.duration,
          patientName: patient.firstName + ' ' + patient.lastName,
          isRead,
          createdDate: appointment.createdDate
        }
      })

      setNotifications(listNotification)
    }

    if (listNotificationDoctor.length === 0) {
      setNotifications([])
    }
  }, [listNotificationDoctor])

  useEffect(() => {
    receiveNotifications()
  }, [stompDoctor])

  const receiveNotifications = () => {
    if (stompDoctor) {
      stompDoctor.subscribe(
        `/user/${currentUser.id}/appointment`,
        (payload) => {
          let payloadData = JSON.parse(payload.body)
          setNotifications((prev) => [payloadData, ...prev])
        }
      )
    }
  }

  const clickNotification = (
    notification: INotificationDoctorSocket,
    indexNotification: number
  ) => {
    handleClose()
    navigate(
      `/doctor/appointment/appointment-patient/${notification.patientId}`
    )
  }

  const renderAppointmentNotification = () => {
    return notifications.map((notification, index) => {
      if (!notification.isRead) {
        return (
          <div
            className='notification'
            onClick={() => {
              clickNotification(notification, index)
            }}
            key={index}
          >
            <div className='avatar_patient'>
              <img src={notification.avatarPatient} alt='' />
            </div>
            <div className='info_appointment'>
              <p>
                <strong>{notification.patientName}</strong> has booked an
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
              <div className='created_date'>
                <Moment fromNow>{notification.createdDate}</Moment>
              </div>
            </div>
            <span className='dot_read'></span>
          </div>
        )
      }
    })
  }

  return (
    <>
      <Badge
        badgeContent={notifications.filter((item) => !item.isRead).length}
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
          horizontal: 'right'
        }}
      >
        <div className='appointment_notification-container'>
          {notifications.filter((item) => !item.isRead).length === 0 ? (
            <p>No announcements yet</p>
          ) : (
            renderAppointmentNotification()
          )}
        </div>
      </Popover>
    </>
  )
}
